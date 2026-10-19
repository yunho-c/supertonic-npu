"""Isolated frontend layout probes; temporal convolutions stay in BCT order."""
import math

import torch
from torch.nn import functional as F

try:
    from .qnn_transformations import MaskAwareConvNeXtBlock
    from .research_transformations import GatherBoundaryBlock
except ImportError:
    from qnn_transformations import MaskAwareConvNeXtBlock
    from research_transformations import GatherBoundaryBlock

MODES = ('baseline', 'gather-boundary', 'pack8', 'pack8-norm-tail', 'compact-relative', 'packed-compact')


class DurationBucket96(torch.nn.Module):
    """Public T128 ABI; host must route valid lengths above96 to baseline."""
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, text_ids, style_dp, text_mask):
        return self.source(text_ids[..., :96], style_dp, text_mask[..., :96])


class TextBucket96(torch.nn.Module):
    """Compact valid frontend with explicit zero extension for fixed T128 flow."""
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, text_ids, style_ttl, text_mask):
        return F.pad(self.source(text_ids[..., :96], style_ttl, text_mask[..., :96]), (0, 32))


def pack(value, rows=8):
    batch, channels, length = value.shape
    aligned = (length + rows - 1) // rows * rows
    if aligned != length:
        value = F.pad(value, (0, aligned - length))
    return value.transpose(1, 2).reshape(1, rows, batch * aligned // rows, channels).permute(0, 3, 1, 2)


def unpack(value, batch, channels, length, rows=8):
    aligned = (length + rows - 1) // rows * rows
    return value.permute(0, 2, 3, 1).reshape(batch, aligned, channels).transpose(1, 2)[..., :length]


class FrontendPackedBlock(torch.nn.Module):
    """Pad only pointwise work to a tile multiple, then crop every batch.

    Duration T128 has 129 internal positions. Its temporal convolution and
    boundary extension still see exactly those 129 positions. Seven zeros are
    appended separately per batch only after the temporal convolution. Channel
    normalization, pointwise projections and the residual cannot mix positions.
    """
    def __init__(self, block, norm_tail=False):
        super().__init__()
        self.block, self.norm_tail = block, norm_tail
        for conv in (block.pwconv1, block.pwconv2):
            if (not isinstance(conv, torch.nn.Conv1d) or conv.kernel_size != (1,)
                    or conv.stride != (1,) or conv.padding != (0,)
                    or conv.dilation != (1,) or conv.groups != 1):
                raise ValueError('Packing requires ungrouped pointwise convolutions')
        if not isinstance(block.norm.norm, torch.nn.LayerNorm):
            raise ValueError('Packing requires channel LayerNorm')
        if tuple(block.norm.norm.normalized_shape) != (block.gamma.shape[1],):
            raise ValueError('LayerNorm must normalize channels only')
        self.w1 = torch.nn.Parameter(block.pwconv1.weight.detach().unsqueeze(-1), requires_grad=False)
        self.w2 = torch.nn.Parameter(block.pwconv2.weight.detach().unsqueeze(-1), requires_grad=False)

    def forward(self, inputs, mask=None):
        block = self.block
        batch, channels, length = inputs.shape
        residual = inputs if mask is None else inputs * mask
        extended = residual
        if mask is not None:
            terminal = mask - torch.cat((mask[..., 1:], torch.zeros_like(mask[..., :1])), -1)
            extended = residual + (residual * terminal).sum(-1, keepdim=True) * (1 - mask)
        hidden = block.dwconv(block.pad(extended))
        if mask is not None:
            hidden = hidden * mask
        if self.norm_tail:
            hidden = block.norm.norm(pack(hidden).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            hidden = pack(block.norm(hidden))
        hidden = F.conv2d(hidden, self.w1, block.pwconv1.bias)
        hidden = F.conv2d(block.act(hidden), self.w2, block.pwconv2.bias)
        if self.norm_tail:
            output = pack(residual) + block.gamma.reshape(1, channels, 1, 1) * hidden
            if mask is not None:
                output = output * pack(mask)
            return unpack(output, batch, channels, length)
        output = residual + block.gamma * unpack(hidden, batch, channels, length)
        return output if mask is None else output * mask


class CompactRelativeAttention(torch.nn.Module):
    """Global content attention plus compact learned relative offsets.

    Q@relativeK uses nine columns. Gather expands those scores to the global
    T-by-T softmax domain, zeroing offsets outside the learned window. After
    global softmax, only the nine band weights contribute to relative V.
    Neither the global QK product nor global weights@V is approximated.
    """
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, inputs, attention_mask):
        source = self.source
        batch, _, length = inputs.shape
        heads, width = source.num_heads, source.head_channels
        query = source.conv_q(inputs).reshape(batch, heads, width, length).transpose(2, 3) / math.sqrt(width)
        key = source.conv_k(inputs).reshape(batch, heads, width, length).transpose(2, 3)
        value = source.conv_v(inputs).reshape(batch, heads, width, length).transpose(2, 3)
        scores = torch.matmul(query, key.transpose(-2, -1))
        window = source.window_size
        band = 2 * window + 1
        positions = torch.arange(length, device=inputs.device)
        offset = positions.unsqueeze(0) - positions.unsqueeze(1)
        in_band = (offset >= -window) & (offset <= window)
        safe = torch.where(in_band, offset + window, torch.zeros_like(offset))
        indices = positions.unsqueeze(1) * band + safe
        relative = torch.matmul(query, source.emb_rel_k.unsqueeze(0).transpose(-2, -1))
        relative = torch.index_select(relative.reshape(batch, heads, -1), -1, indices.reshape(-1))
        scores = scores + relative.reshape(batch, heads, length, length) * in_band.reshape(1, 1, length, length)
        weights = torch.softmax(scores.masked_fill(attention_mask == 0, -10000.0), -1)
        attended = torch.matmul(weights, value)
        offsets = torch.arange(band, device=inputs.device) - window
        keys = positions.unsqueeze(1) + offsets.unsqueeze(0)
        valid = (keys >= 0) & (keys < length)
        indices = positions.unsqueeze(1) * length + torch.where(valid, keys, torch.zeros_like(keys))
        relative_weights = torch.index_select(weights.reshape(batch, heads, -1), -1, indices.reshape(-1))
        relative_weights = relative_weights.reshape(batch, heads, length, band) * valid.reshape(1, 1, length, band)
        attended = attended + torch.matmul(relative_weights, source.emb_rel_v.unsqueeze(0))
        return source.conv_o(attended.transpose(2, 3).reshape(batch, source.channels, length))


def apply_frontend(model, mode, metadata=None):
    from examples.models.supertonic.source_transformations.mlx import MLXRelativeMultiHeadAttention
    if mode not in MODES:
        raise ValueError(mode)
    if mode == 'packed-compact':
        packed, compact = {}, {}
        model = apply_frontend(model, 'pack8-norm-tail', packed)
        model = apply_frontend(model, 'compact-relative', compact)
        if metadata is not None:
            metadata.update(packed)
            metadata.update(frontend_mode=mode,
                            changed_modules=packed['changed_modules']+compact['changed_modules'],
                            composition=['pack8-norm-tail','compact-relative'],
                            attention_contract=compact['attention_contract'])
        return model
    replacements = []
    for name, child in model.named_modules():
        if type(child) is MaskAwareConvNeXtBlock and mode in ('gather-boundary', 'pack8', 'pack8-norm-tail'):
            replacement = (GatherBoundaryBlock(child.block) if mode == 'gather-boundary'
                           else FrontendPackedBlock(child.block, mode == 'pack8-norm-tail'))
            replacements.append((name, replacement))
        elif type(child) is MLXRelativeMultiHeadAttention and mode == 'compact-relative':
            replacements.append((name, CompactRelativeAttention(child)))
    if mode != 'baseline' and not replacements:
        raise ValueError('No eligible frontend modules')
    for name, replacement in replacements:
        if not name:
            model = replacement
        else:
            parent, _, child_name = name.rpartition('.')
            setattr(model.get_submodule(parent), child_name, replacement)
    if metadata is not None:
        metadata.update(frontend_mode=mode, changed_modules=[n for n, _ in replacements],
                        source_parameters_preserved=True, temporal_layout='unchanged BCT',
                        packing_alignment='per-batch zero pad after temporal convolution, crop before next block',
                        attention_contract='full global content attention; nine learned relative offsets')
    return model
