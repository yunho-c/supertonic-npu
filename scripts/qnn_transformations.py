"""Equivalent graph rewrites needed for complete QNN delegation."""

import torch


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
