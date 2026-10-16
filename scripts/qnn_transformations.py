"""QNN delegation rewrites, equivalent layout changes, and an opt-in GELU approximation."""

import torch
from torch.nn import functional as F


class MaskAwareConvNeXtBlock(torch.nn.Module):
    """Replicate the valid sequence boundary when a fixed bucket is padded."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs, mask=None):
        if mask is None:
            return self.block(inputs)
        residual = inputs * mask
        terminal = mask - torch.cat((mask[..., 1:], torch.zeros_like(mask[..., :1])), dim=-1)
        last_valid = (residual * terminal).sum(dim=-1, keepdim=True)
        extended = residual + last_valid * (1 - mask)
        block = self.block
        hidden = block.dwconv(block.pad(extended)) * mask
        hidden = block.pwconv2(block.act(block.pwconv1(block.norm(hidden))))
        return (residual + block.gamma * hidden) * mask


def replace_masked_padding(model):
    from examples.models.supertonic.model.layers import ConvNeXtBlock

    for name, child in list(model.named_children()):
        if isinstance(child, ConvNeXtBlock):
            setattr(model, name, MaskAwareConvNeXtBlock(child))
        else:
            replace_masked_padding(child)
    return model


def stable_softplus(value, beta=1, threshold=20):
    if beta != 1 or threshold != 20:
        raise ValueError("Only Supertonic's default softplus is supported")
    # Avoid exp overflow and the scalar comparison introduced by the default
    # softplus decomposition after QNN's scalar-lifting pass has already run.
    return torch.relu(value) + torch.log1p(torch.exp(-torch.abs(value)))


def prepare_vector_graph(model, inputs):
    program = torch.export.export(model, inputs, strict=True)
    for node in program.graph.nodes:
        if node.target == torch.ops.aten.bitwise_not.default:
            if node.args[0].meta["val"].dtype != torch.bool:
                raise TypeError("Only boolean mask inversion can become logical_not")
            node.target = torch.ops.aten.logical_not.default
    program.graph_module.recompile()
    program = program.run_decompositions({torch.ops.aten.softplus.default: stable_softplus})
    return program.module()


class TanhGELU(torch.nn.Module):
    """Explicit standard tanh approximation, avoiding QNN's exact GELU kernel."""

    def forward(self, value):
        return 0.5 * value * (1 + torch.tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)))


class ChannelLastConvNeXt(torch.nn.Module):
    """Keep normalization and pointwise MLP in BLC layout between convolutions."""

    def __init__(self, block, masked):
        super().__init__()
        self.block = block
        self.masked = masked

    def forward(self, inputs, mask=None):
        block = self.block
        residual = inputs if mask is None else inputs * mask
        extended = residual
        if self.masked and mask is not None:
            terminal = mask - torch.cat((mask[..., 1:], torch.zeros_like(mask[..., :1])), dim=-1)
            extended = residual + (residual * terminal).sum(dim=-1, keepdim=True) * (1 - mask)
        hidden = block.dwconv(block.pad(extended))
        if mask is not None:
            hidden = hidden * mask
        hidden = block.norm.norm(hidden.transpose(1, 2))
        hidden = block.act(F.linear(hidden, block.pwconv1.weight[..., 0], block.pwconv1.bias))
        hidden = F.linear(hidden, block.pwconv2.weight[..., 0], block.pwconv2.bias)
        output = residual + (hidden * block.gamma.transpose(1, 2)).transpose(1, 2)
        return output if mask is None else output * mask


def optimize_modules(model, gelu="exact", conv_layout="original"):
    from examples.models.supertonic.model.vocoder import DecoderConvNeXtBlock

    for name, child in list(model.named_children()):
        optimize_modules(child, gelu, conv_layout)
        if gelu == "tanh" and isinstance(child, torch.nn.GELU):
            setattr(model, name, TanhGELU())
        elif conv_layout == "channel-last" and isinstance(child, MaskAwareConvNeXtBlock):
            setattr(model, name, ChannelLastConvNeXt(child.block, masked=True))
        elif conv_layout == "channel-last" and isinstance(child, DecoderConvNeXtBlock):
            setattr(model, name, ChannelLastConvNeXt(child, masked=False))
    return model
