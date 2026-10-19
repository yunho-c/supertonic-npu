"""Experimental activation/MLP tiling and boundary rewrites; weights are shared."""

import torch

try:
    from .qnn_transformations import ChannelLastActivation, MaskAwareConvNeXtBlock
except ImportError:
    from qnn_transformations import ChannelLastActivation, MaskAwareConvNeXtBlock


PACKING_ROWS = (2, 4, 8, 16)
EXPERIMENTS = tuple(f"{kind}-pack{rows}" for kind in ("gelu", "mlp") for rows in PACKING_ROWS) + ("gather-boundary",)


class PackedActivation(torch.nn.Module):
    """Expose a different dense tile shape to HTP while retaining Conv1d arithmetic.

    The pointwise probe found a faster 8-row packing for the whole MLP, but that
    changed convolution rounding. Here only the elementwise activation is packed.
    Shape-only transformations preserve every sample's activation input/output.
    Physical-device equality must still be checked after QNN compilation.
    """

    def __init__(self, activation, rows=8):
        super().__init__()
        if rows not in PACKING_ROWS:
            raise ValueError("Packing rows must be 2, 4, 8 or 16")
        self.activation = activation
        self.rows = rows

    def forward(self, inputs):
        batch, channels, length = inputs.shape
        if batch * length % self.rows:
            raise ValueError("Batch times bucket length must be divisible by packing rows")
        packed = inputs.transpose(1, 2).reshape(1, self.rows, batch * length // self.rows, channels).contiguous()
        return self.activation(packed).reshape(batch, length, channels).transpose(1, 2)


class PackedPointwisePair(torch.nn.Module):
    """Research-only full MLP packing. Algebraically equivalent, not bit-exact.

    This is retained for full-flow investigation of the isolated-block result;
    a candidate with changed output is rejected by the exact-execution campaign.
    """

    def __init__(self, first, activation, second, rows=8):
        super().__init__()
        if rows not in PACKING_ROWS:
            raise ValueError("Packing rows must be 2, 4, 8 or 16")
        for conv in (first, second):
            if (not isinstance(conv, torch.nn.Conv1d) or conv.kernel_size != (1,)
                    or conv.stride != (1,) or conv.padding != (0,) or conv.dilation != (1,)
                    or conv.groups != 1):
                raise ValueError("MLP packing requires ungrouped, unpadded, unit-stride pointwise Conv1d projections")
        if first.out_channels != second.in_channels or first.in_channels != second.out_channels:
            raise ValueError("MLP projection channels must connect and restore the input channels")
        self.w1 = torch.nn.Parameter(first.weight.detach().unsqueeze(-1), requires_grad=first.weight.requires_grad)
        self.w2 = torch.nn.Parameter(second.weight.detach().unsqueeze(-1), requires_grad=second.weight.requires_grad)
        self.b1, self.b2 = first.bias, second.bias
        self.activation = activation
        self.rows = rows

    def forward(self, inputs):
        batch, channels, length = inputs.shape
        if batch * length % self.rows:
            raise ValueError("Batch times bucket length must be divisible by packing rows")
        packed = inputs.transpose(1, 2).reshape(1, self.rows, batch * length // self.rows, channels).permute(0, 3, 1, 2)
        hidden = torch.nn.functional.conv2d(packed, self.w1, self.b1)
        hidden = torch.nn.functional.conv2d(self.activation(hidden), self.w2, self.b2)
        return hidden.permute(0, 2, 3, 1).reshape(batch, length, channels).transpose(1, 2)


class GatherBoundaryBlock(torch.nn.Module):
    """Select the final valid position instead of a full activation reduction."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, inputs, mask=None):
        if mask is None:
            return self.block(inputs)
        residual = inputs * mask
        index = (mask.sum(dim=-1, keepdim=True).to(torch.int64) - 1).expand(-1, inputs.shape[1], -1)
        last_valid = torch.gather(residual, 2, index)
        extended = residual + last_valid * (1 - mask)
        block = self.block
        hidden = block.dwconv(block.pad(extended)) * mask
        hidden = block.pwconv2(block.act(block.pwconv1(block.norm(hidden))))
        return (residual + block.gamma * hidden) * mask


def named_convnext_blocks(model):
    """Return boundary wrappers in depth-first module registration order.

    In the standard estimator, indices 0..23 are the four main stages (six
    blocks each), followed by the four final blocks at 24..27. Do not sort
    module names lexically: main_blocks.10 must follow main_blocks.6.
    Gather replacement and MLP packing preserve these names and indices.
    """
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, (MaskAwareConvNeXtBlock, GatherBoundaryBlock))]


def _block_indices(value, name, count, *, allow_all=False):
    if allow_all and (value is None or isinstance(value, str) and value.strip() == "all"):
        return list(range(count))
    if isinstance(value, str):
        if not value.strip() and not allow_all:
            return []
        try:
            value = [int(item.strip()) for item in value.split(",")]
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of block indices") from exc
    try:
        indices = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a list of block indices") from exc
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError(f"{name} must contain integer block indices")
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(f"{name} indices must be in 0..{count - 1} ({count} ConvNeXt blocks)")
    return sorted(set(indices))


def apply_experiment(model, experiment, *, mlp_blocks="all", mlp_exclude=(), gather_boundary=False, metadata=None):
    """Rewrite a model in place, retaining the original two-argument API.

    MLP indices are zero-based and exclusions win over selection. ``all`` is
    the default; iterables and comma-separated lists are accepted. Selection
    order and duplicates do not change block numbering. Gather, when enabled,
    applies to every boundary block, including blocks excluded from packing.
    An optional metadata dict receives the resolved selection and name map.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    blocks = named_convnext_blocks(model)
    requested = _block_indices(mlp_blocks, "mlp_blocks", len(blocks), allow_all=True)
    excluded = _block_indices(mlp_exclude, "mlp_exclude", len(blocks))
    is_mlp = experiment.startswith("mlp-pack")
    if not is_mlp and (requested != list(range(len(blocks))) or excluded):
        raise ValueError("MLP block selection requires an mlp-pack experiment")
    selected = [index for index in requested if index not in excluded] if is_mlp else []
    rows = int(experiment.rsplit("pack", 1)[1]) if experiment != "gather-boundary" else None
    gather_boundary = gather_boundary or experiment == "gather-boundary"

    # Validate every selected projection before mutating any block. Parameter
    # views share the original storage; neither construction nor execution
    # changes source weights or biases.
    pairs = {}
    for index in selected:
        block = blocks[index][1].block
        # Packing defines its own rank-four activation layout. Keep gelu-last's
        # BCL/BLC wrapper only on blocks whose Conv1d MLP remains unpacked.
        activation = block.act.activation if isinstance(block.act, ChannelLastActivation) else block.act
        pairs[index] = PackedPointwisePair(block.pwconv1, activation, block.pwconv2, rows)
    for index, (name, wrapped) in enumerate(blocks):
        block = wrapped.block
        if index in pairs:
            block.pwconv1 = pairs[index]
            block.act = torch.nn.Identity()
            block.pwconv2 = torch.nn.Identity()
        elif experiment.startswith("gelu-pack"):
            block.act = PackedActivation(block.act, rows)
        if gather_boundary and not isinstance(wrapped, GatherBoundaryBlock):
            replacement = GatherBoundaryBlock(block)
            replacement.training = wrapped.training
            if name:
                parent_name, _, child_name = name.rpartition(".")
                setattr(model.get_submodule(parent_name), child_name, replacement)
            else:
                model = replacement
    if metadata is not None:
        metadata.update(
            packing_rows=rows,
            mlp_rows=rows if is_mlp else None,
            mlp_blocks_requested=("all" if mlp_blocks is None or isinstance(mlp_blocks, str)
                                  and mlp_blocks.strip() == "all" else requested),
            mlp_blocks=selected,
            mlp_exclude=excluded,
            gather_boundary=any(isinstance(wrapped, GatherBoundaryBlock) for _, wrapped in named_convnext_blocks(model)),
            convnext_block_names=[name for name, _ in blocks],
        )
    return model
