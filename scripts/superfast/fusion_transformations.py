"""Independent inference-only ConvNeXt fusion probes for the combined baseline.

The caller first applies mlp-pack8 (except blocks 24..27) and gather-boundary.
No source parameters are modified. All masks follow the exporter's valid-domain
contract: finite inputs/intermediates and binary, nonempty prefix masks. Temporal convolution
always remains in BCL; pointwise packing never defines temporal neighbours.
"""

import hashlib

import torch
from torch.nn import functional as F

try:
    from .research_transformations import GatherBoundaryBlock, PackedPointwisePair, named_convnext_blocks
except ImportError:
    from research_transformations import GatherBoundaryBlock, PackedPointwisePair, named_convnext_blocks


FUSIONS = ("packed-norm-tail", "fold-gamma", "drop-hidden-mask")


def _pack(value, rows):
    batch, channels, length = value.shape
    if batch * length % rows:
        raise ValueError("Batch times bucket length must be divisible by packing rows")
    return value.transpose(1, 2).reshape(1, rows, batch * length // rows, channels).permute(0, 3, 1, 2)


def _unpack(value, batch, channels, length):
    return value.permute(0, 2, 3, 1).reshape(batch, length, channels).transpose(1, 2)


def _boundary(block, inputs, mask):
    residual = inputs if mask is None else inputs * mask
    extended = residual
    if mask is not None:
        index = (mask.sum(dim=-1, keepdim=True).to(torch.int64) - 1).expand(-1, inputs.shape[1], -1)
        last_valid = torch.gather(residual, 2, index)
        extended = residual + last_valid * (1 - mask)
    return residual, block.dwconv(block.pad(extended))


def _packed_pair(pair, inputs, weight, bias):
    batch, channels, length = inputs.shape
    hidden = F.conv2d(_pack(inputs, pair.rows), pair.w1, pair.b1)
    hidden = F.conv2d(pair.activation(hidden), weight, bias)
    return _unpack(hidden, batch, channels, length)


def _validate_block(block):
    pair = block.pwconv1
    if isinstance(pair, PackedPointwisePair):
        if not isinstance(block.act, torch.nn.Identity) or not isinstance(block.pwconv2, torch.nn.Identity):
            raise ValueError("Packed fusion requires the complete PackedPointwisePair rewrite")
        channels = pair.w2.shape[0]
    else:
        conv = block.pwconv2
        if (not isinstance(conv, torch.nn.Conv1d) or conv.kernel_size != (1,)
                or conv.stride != (1,) or conv.padding != (0,) or conv.dilation != (1,) or conv.groups != 1):
            raise ValueError("Fusion requires a packed pair or ungrouped pointwise second projection")
        channels = conv.out_channels
    if tuple(block.gamma.shape) != (1, channels, 1):
        raise ValueError("Fusion requires per-output-channel gamma with shape (1, C, 1)")


class PackedNormTailBlock(GatherBoundaryBlock):
    """Pack before channel LayerNorm and keep the entire pointwise tail packed.

    LayerNorm's final dimension is still exactly C, independently at every
    original (batch, time) position. Its NHWC output feeds the packed Conv2d pair.
    Gamma, residual and final mask retain that pair's rank-four geometry.
    """

    def __init__(self, block):
        _validate_block(block)
        if not isinstance(block.pwconv1, PackedPointwisePair):
            raise ValueError("Packed normalization requires an already packed MLP")
        norm = getattr(block.norm, "norm", None)
        if not isinstance(norm, torch.nn.LayerNorm) or tuple(norm.normalized_shape) != (block.pwconv1.w1.shape[1],):
            raise ValueError("Packed normalization requires channel-only LayerNorm")
        super().__init__(block)

    def forward(self, inputs, mask=None):
        block, pair = self.block, self.block.pwconv1
        batch, channels, length = inputs.shape
        if batch * length % pair.rows:
            raise ValueError("Batch times bucket length must be divisible by packing rows")
        residual, hidden = _boundary(block, inputs, mask)
        if mask is not None:
            hidden = hidden * mask
        hidden = hidden.transpose(1, 2).reshape(1, pair.rows, batch * length // pair.rows, channels)
        hidden = block.norm.norm(hidden).permute(0, 3, 1, 2)
        hidden = F.conv2d(hidden, pair.w1, pair.b1)
        hidden = F.conv2d(pair.activation(hidden), pair.w2, pair.b2)
        output = _pack(residual, pair.rows) + block.gamma.reshape(1, channels, 1, 1) * hidden
        if mask is not None:
            output = output * _pack(mask, pair.rows)
        return _unpack(output, batch, channels, length)


class FoldedGammaBlock(GatherBoundaryBlock):
    """Replace gamma * (W2 @ h + b2) with derived W2 and b2 constants.

    Derived tensors are independent storage. Original block weights, bias and
    gamma remain registered and unmodified for provenance; only the derived
    tensors participate in forward/export. Rounding differs from the baseline.
    This wrapper is an inference snapshot, not a training transformation.
    """

    def __init__(self, block):
        _validate_block(block)
        super().__init__(block)
        pair = block.pwconv1
        weight, bias = (pair.w2, pair.b2) if isinstance(pair, PackedPointwisePair) else (block.pwconv2.weight, block.pwconv2.bias)
        scale = block.gamma.detach().reshape(-1)
        with torch.no_grad():
            folded_weight = weight.detach() * scale.reshape(-1, *([1] * (weight.ndim - 1)))
            folded_bias = None if bias is None else bias.detach() * scale
        self.register_buffer("folded_weight", folded_weight)
        self.register_buffer("folded_bias", folded_bias)

    def forward(self, inputs, mask=None):
        block = self.block
        residual, hidden = _boundary(block, inputs, mask)
        if mask is not None:
            hidden = hidden * mask
        hidden = block.norm(hidden)
        if isinstance(block.pwconv1, PackedPointwisePair):
            hidden = _packed_pair(block.pwconv1, hidden, self.folded_weight, self.folded_bias)
        else:
            hidden = F.conv1d(block.act(block.pwconv1(hidden)), self.folded_weight, self.folded_bias)
        output = residual + hidden
        return output if mask is None else output * mask


class DropHiddenMaskBlock(GatherBoundaryBlock):
    """Omit the depthwise-output mask, retaining the final output mask.

    After dwconv all operations are pointwise in time: channel LayerNorm,
    pointwise MLP, elementwise gamma and residual. Therefore changing an invalid
    position cannot affect a valid one, even with nonzero normalization and
    projection biases. The final mask removes every invalid position. The
    initial residual mask remains necessary for gather boundary extension.
    """

    def __init__(self, block):
        _validate_block(block)
        norm = getattr(block.norm, "norm", None)
        if not isinstance(norm, torch.nn.LayerNorm) or tuple(norm.normalized_shape) != (block.gamma.shape[1],):
            raise ValueError("Mask simplification requires channel-only LayerNorm")
        pair = block.pwconv1
        activation = pair.activation if isinstance(pair, PackedPointwisePair) else block.act
        # The production baseline is exact GELU, possibly wrapped for BLC.
        activation = getattr(activation, "activation", activation)
        if not isinstance(activation, torch.nn.GELU):
            raise ValueError("Mask simplification requires the known pointwise GELU activation")
        if not isinstance(pair, PackedPointwisePair):
            if (not isinstance(pair, torch.nn.Conv1d) or pair.kernel_size != (1,)
                    or pair.stride != (1,) or pair.padding != (0,) or pair.dilation != (1,) or pair.groups != 1):
                raise ValueError("Mask simplification requires pointwise first projection")
        super().__init__(block)

    def forward(self, inputs, mask=None):
        block = self.block
        residual, hidden = _boundary(block, inputs, mask)
        hidden = block.pwconv2(block.act(block.pwconv1(block.norm(hidden))))
        output = residual + block.gamma * hidden
        return output if mask is None else output * mask


def _tensor_record(value):
    if value is None:
        return None
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def derived_parameter_arrays(model):
    """Exact original and derived arrays for the exporter's standalone NPZ."""
    arrays = {}
    for name, wrapped in named_convnext_blocks(model):
        if not isinstance(wrapped, FoldedGammaBlock):
            continue
        block, pair = wrapped.block, wrapped.block.pwconv1
        weight, bias = (pair.w2, pair.b2) if isinstance(pair, PackedPointwisePair) else (block.pwconv2.weight, block.pwconv2.bias)
        for key, value in (("source_weight", weight), ("source_bias", bias), ("source_gamma", block.gamma),
                           ("folded_weight", wrapped.folded_weight), ("folded_bias", wrapped.folded_bias)):
            if value is not None:
                arrays[f"{name}.{key}"] = value.detach().cpu().numpy()
    return arrays


def apply_fusion(model, fusion, *, metadata=None):
    """Apply one independent probe to an already prepared combined baseline.

    All wrappers subclass GatherBoundaryBlock, so existing block enumeration and
    shared CFG prefix composition remain valid. Reject a second fusion to avoid
    accidentally combining hypotheses. Constructor validation precedes mutation.
    """
    if fusion not in FUSIONS:
        raise ValueError(f"Unknown fusion: {fusion}")
    blocks = named_convnext_blocks(model)
    if not blocks or any(type(wrapped) is not GatherBoundaryBlock for _, wrapped in blocks):
        raise ValueError("Fusion requires unmodified GatherBoundaryBlock wrappers from the combined baseline")
    wrapper_type = {"packed-norm-tail": PackedNormTailBlock, "fold-gamma": FoldedGammaBlock,
                    "drop-hidden-mask": DropHiddenMaskBlock}[fusion]
    replacements, derived = [], []
    for index, (name, wrapped) in enumerate(blocks):
        if fusion == "packed-norm-tail" and not isinstance(wrapped.block.pwconv1, PackedPointwisePair):
            continue
        replacement = wrapper_type(wrapped.block)
        replacement.train(wrapped.training)
        replacements.append((index, name, replacement))
        if isinstance(replacement, FoldedGammaBlock):
            block, pair = wrapped.block, wrapped.block.pwconv1
            weight, bias = (pair.w2, pair.b2) if isinstance(pair, PackedPointwisePair) else (block.pwconv2.weight, block.pwconv2.bias)
            derived.append({"index": index, "block": name,
                            "formula": "folded_weight[o,...] = source_weight[o,...] * gamma[o]; folded_bias[o] = source_bias[o] * gamma[o]",
                            "source_weight": _tensor_record(weight), "source_bias": _tensor_record(bias),
                            "source_gamma": _tensor_record(block.gamma),
                            "folded_weight": _tensor_record(replacement.folded_weight),
                            "folded_bias": _tensor_record(replacement.folded_bias)})
    if not replacements:
        raise ValueError("Fusion selected no eligible ConvNeXt blocks")
    for _, name, replacement in replacements:
        if name:
            parent_name, _, child_name = name.rpartition(".")
            setattr(model.get_submodule(parent_name), child_name, replacement)
        else:
            model = replacement
    if metadata is not None:
        metadata.update(fusion=fusion, fusion_blocks=[index for index, _, _ in replacements],
                        fusion_block_names=[name for _, name, _ in replacements],
                        source_parameters_preserved=True, derived_parameters=derived,
                        derived_parameter_bytes=sum(item[key]["bytes"] for item in derived
                                                    for key in ("folded_weight", "folded_bias") if item[key] is not None),
                        mask_contract="finite inputs/intermediates; binary nonempty prefix masks; channel-only normalization and pointwise tail")
    return model
