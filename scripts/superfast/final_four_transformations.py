"""Extend the frozen 24-block packed-norm-tail model to selected final blocks."""
try:
    from .fusion_transformations import PackedNormTailBlock
    from .research_transformations import GatherBoundaryBlock, apply_experiment, named_convnext_blocks
except ImportError:
    from fusion_transformations import PackedNormTailBlock
    from research_transformations import GatherBoundaryBlock, apply_experiment, named_convnext_blocks


FINAL = (24, 25, 26, 27)
MODES = {
    "full28": (FINAL, True),
    "full28-mlp-only": (FINAL, False),
    **{f"leave-{i}": (tuple(j for j in FINAL if j != i), True) for i in FINAL},
}


def extend_final_four(model, final_blocks=FINAL, *, norm_tail=True, metadata=None):
    """Keep the first 24 fused blocks intact; pack only the declared final subset.

    Packing affects channel-pointwise work only. The existing gather boundary
    and temporal depthwise convolution stay in their original BCL positions.
    Every candidate starts from the same unfurther-modified 24-block baseline.
    """
    selected = list(final_blocks)
    if (not selected or any(type(i) is not int or i not in FINAL for i in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("final_blocks must be a nonempty unique subset of 24..27")
    selected.sort()
    blocks = named_convnext_blocks(model)
    if (len(blocks) != 28 or any(type(b) is not PackedNormTailBlock for _, b in blocks[:24])
            or any(type(b) is not GatherBoundaryBlock for _, b in blocks[24:])):
        raise ValueError("Expected the frozen 24-block packed-norm-tail baseline")
    model = apply_experiment(model, "mlp-pack8", mlp_blocks=selected)
    if norm_tail:
        for index in selected:
            name, wrapped = blocks[index]
            replacement = PackedNormTailBlock(wrapped.block)
            replacement.train(wrapped.training)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, replacement)
    if metadata is not None:
        metadata.update(
            final_four_blocks=selected, final_four_norm_tail=norm_tail,
            mlp_blocks_requested=list(range(24)) + selected,
            mlp_blocks=list(range(24)) + selected,
            mlp_exclude=[i for i in FINAL if i not in selected],
            fusion_blocks=list(range(24)) + (selected if norm_tail else []),
            fusion_block_names=[name for i, (name, _) in enumerate(blocks)
                                if i < 24 or (norm_tail and i in selected)],
            baseline="Candidate8: 24/28 pack8 with packed normalization/tail",
            temporal_convolution_and_cfg_boundaries_unchanged=True,
        )
    return model
