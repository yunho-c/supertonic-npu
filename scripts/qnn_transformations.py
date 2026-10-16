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


class PointwiseConv2d(torch.nn.Module):
    """Keep a ConvNeXt pointwise pair in rank four across its activation.

    QNN already lowers each Conv1d to Conv2d. Expanding before the first
    projection and squeezing only after the second lets its layout pass carry
    channels-last through GELU. The weights retain their original storage.
    """

    def __init__(self, conv, *, expand_input=False, squeeze_output=False):
        super().__init__()
        if (not isinstance(conv, torch.nn.Conv1d) or conv.kernel_size != (1,)
                or conv.stride != (1,) or conv.padding != (0,) or conv.dilation != (1,)):
            raise ValueError("Expected an unpadded, unit-stride pointwise Conv1d")
        # Register the 4D view as a static parameter. QNN's convolution visitor
        # requires a constant filter and cannot consume a runtime unsqueeze.
        self.weight = torch.nn.Parameter(conv.weight.detach().unsqueeze(2), requires_grad=conv.weight.requires_grad)
        self.bias = conv.bias
        self.groups = conv.groups
        self.expand_input = expand_input
        self.squeeze_output = squeeze_output

    def forward(self, inputs):
        if self.expand_input:
            inputs = inputs.unsqueeze(2)
        output = F.conv2d(inputs, self.weight, self.bias, groups=self.groups)
        return output.squeeze(2) if self.squeeze_output else output


class ChannelLastActivation(torch.nn.Module):
    """Evaluate an elementwise activation in BLC, preserving both Conv1ds."""

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, inputs):
        return self.activation(inputs.transpose(1, 2)).transpose(1, 2)


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
        elif conv_layout == "pointwise-2d" and isinstance(child, (MaskAwareConvNeXtBlock, DecoderConvNeXtBlock)):
            block = child.block if isinstance(child, MaskAwareConvNeXtBlock) else child
            if not isinstance(block.pwconv1, PointwiseConv2d):
                block.pwconv1 = PointwiseConv2d(block.pwconv1, expand_input=True)
                block.pwconv2 = PointwiseConv2d(block.pwconv2, squeeze_output=True)
        elif conv_layout == "gelu-last" and isinstance(child, (MaskAwareConvNeXtBlock, DecoderConvNeXtBlock)):
            block = child.block if isinstance(child, MaskAwareConvNeXtBlock) else child
            if not isinstance(block.act, ChannelLastActivation):
                block.act = ChannelLastActivation(block.act)
    return model
