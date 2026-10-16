import pytest
import torch
from pathlib import Path

from scripts.qnn_transformations import (ChannelLastActivation, ChannelLastConvNeXt, MaskAwareConvNeXtBlock,
    PointwiseConv2d, TanhGELU, optimize_modules, prepare_vector_graph, stable_softplus)


def test_softplus_stays_finite_and_matches_reference_at_extremes():
    values = torch.cat((torch.linspace(-100, 100, 10001), torch.tensor([19.999, 20.0, 20.001])))
    torch.testing.assert_close(stable_softplus(values), torch.nn.functional.softplus(values), rtol=1e-6, atol=2e-7)


class Invert(torch.nn.Module):
    def forward(self, value):
        return ~value


def test_mask_rewrite_preserves_boolean_semantics():
    values = torch.tensor([[True, False], [False, True]])
    model = prepare_vector_graph(Invert(), (values,))
    torch.testing.assert_close(model(values), Invert()(values))
    assert all(node.target != torch.ops.aten.bitwise_not.default for node in model.graph.nodes)


def test_integer_inversion_is_not_silently_changed():
    with pytest.raises(TypeError, match="boolean mask"):
        prepare_vector_graph(Invert(), (torch.tensor([0, 1, 7]),))


@pytest.mark.parametrize("length", [1, 5, 16])
@pytest.mark.parametrize("dilation", [1, 4])
def test_bucket_padding_matches_original_sequence_boundary(monkeypatch, length, dilation):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".deps/executorch"))
    from examples.models.supertonic.model.layers import ConvNeXtBlock

    torch.manual_seed(19)
    original = ConvNeXtBlock(8, kernel_size=5, dilation=dilation, layer_scale_init_value=1.0).eval()
    bucket = MaskAwareConvNeXtBlock(original)
    values = torch.randn(1, 8, 16)
    mask = torch.zeros(1, 1, 16)
    mask[..., :length] = 1
    with torch.no_grad():
        expected = original(values[..., :length], mask[..., :length])
        actual = bucket(values, mask)
    torch.testing.assert_close(actual[..., :length], expected, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(actual[..., length:]) == 0


def test_tanh_gelu_matches_standard_approximation_and_has_bounded_error():
    values = torch.linspace(-20, 20, 20001)
    approximation = TanhGELU()(values)
    torch.testing.assert_close(approximation, torch.nn.functional.gelu(values, approximate="tanh"), atol=2e-6, rtol=1e-5)
    assert (approximation - torch.nn.functional.gelu(values)).abs().max() < 0.00048


@pytest.mark.parametrize("length", [1, 7, 16])
def test_channel_last_preserves_masked_boundary_and_mlp(monkeypatch, length):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".deps/executorch"))
    from examples.models.supertonic.model.layers import ConvNeXtBlock

    torch.manual_seed(31)
    block = ConvNeXtBlock(8, 5, dilation=4, layer_scale_init_value=1.0).eval()
    values = torch.randn(1, 8, 16)
    mask = torch.zeros(1, 1, 16)
    mask[..., :length] = 1
    with torch.no_grad():
        actual = ChannelLastConvNeXt(block, True)(values, mask)
        expected = MaskAwareConvNeXtBlock(block)(values, mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_channel_last_preserves_causal_vocoder_block(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".deps/executorch"))
    from examples.models.supertonic.model.vocoder import DecoderConvNeXtBlock

    torch.manual_seed(71)
    block = DecoderConvNeXtBlock(8, dilation=4, expansion=4).eval()
    with torch.no_grad():
        block.gamma.fill_(1)
        values = torch.randn(1, 8, 27)
        expected = block(values)
        actual = ChannelLastConvNeXt(block, False)(values)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("length", [1, 7, 16])
@pytest.mark.parametrize("dilation", [1, 4])
@pytest.mark.parametrize("layout", ["pointwise-2d", "gelu-last"])
def test_conv_preserving_layout_matches_unpadded_sequence_and_shares_weight_storage(monkeypatch, length, dilation, layout):
    import copy
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".deps/executorch"))
    from examples.models.supertonic.model.layers import ConvNeXtBlock

    torch.manual_seed(83)
    block = ConvNeXtBlock(8, 5, dilation=dilation, layer_scale_init_value=1.0).eval()
    reference = copy.deepcopy(block)
    holder = torch.nn.Module()
    holder.add_module("block", MaskAwareConvNeXtBlock(block))
    parameter_storage = {(p.data_ptr(), p.numel()) for p in holder.parameters()}
    optimize_modules(holder, conv_layout=layout)
    assert {(p.data_ptr(), p.numel()) for p in holder.parameters()} == parameter_storage
    values = torch.randn(2, 8, 16)
    mask = torch.zeros(2, 1, 16)
    mask[..., :length] = 1
    with torch.no_grad():
        expected = reference(values[..., :length], mask[..., :length])
        actual = holder.block(values, mask)
    torch.testing.assert_close(actual[..., :length], expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(actual[..., length:]) == 0


def test_pointwise_2d_export_keeps_activation_between_two_convolutions():
    torch.manual_seed(89)
    first, second = torch.nn.Conv1d(8, 32, 1), torch.nn.Conv1d(32, 8, 1)
    pair = torch.nn.Sequential(PointwiseConv2d(first, expand_input=True), torch.nn.GELU(),
                               PointwiseConv2d(second, squeeze_output=True)).eval()
    values = torch.randn(2, 8, 11)
    exported = torch.export.export(pair, (values,))
    activation = next(n for n in exported.graph.nodes if n.target == torch.ops.aten.gelu.default)
    assert list(activation.meta["val"].shape) == [2, 32, 1, 11]
    assert activation.args[0].target == torch.ops.aten.conv2d.default
    assert activation.args[0].args[1].op == "placeholder"
    assert len(activation.users) == 1
    assert next(iter(activation.users)).target == torch.ops.aten.conv2d.default
    with torch.no_grad():
        torch.testing.assert_close(exported.module()(values), second(torch.nn.functional.gelu(first(values))),
                                   atol=1e-6, rtol=1e-5)


def test_pointwise_2d_rejects_spatial_convolution():
    with pytest.raises(ValueError, match="pointwise"):
        PointwiseConv2d(torch.nn.Conv1d(8, 8, 3), expand_input=True)


@pytest.mark.parametrize("layout", ["pointwise-2d", "gelu-last"])
def test_conv_preserving_layout_preserves_causal_vocoder_block(monkeypatch, layout):
    import copy
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".deps/executorch"))
    from examples.models.supertonic.model.vocoder import DecoderConvNeXtBlock

    torch.manual_seed(97)
    block = DecoderConvNeXtBlock(8, dilation=4, expansion=4).eval()
    with torch.no_grad():
        block.gamma.fill_(1)
    reference = copy.deepcopy(block)
    holder = torch.nn.Sequential(block)
    optimize_modules(holder, conv_layout=layout)
    values = torch.randn(1, 8, 27)
    with torch.no_grad():
        torch.testing.assert_close(holder(values), reference(values), atol=1e-6, rtol=1e-5)


def test_channel_last_activation_export_preserves_conv1d_and_exact_gelu():
    torch.manual_seed(101)
    first, second = torch.nn.Conv1d(8, 32, 1), torch.nn.Conv1d(32, 8, 1)
    pair = torch.nn.Sequential(first, ChannelLastActivation(torch.nn.GELU()), second).eval()
    values = torch.randn(2, 8, 11)
    exported = torch.export.export(pair, (values,))
    activation = next(n for n in exported.graph.nodes if n.target == torch.ops.aten.gelu.default)
    assert list(activation.meta["val"].shape) == [2, 11, 32]
    assert sum(n.target == torch.ops.aten.conv1d.default for n in exported.graph.nodes) == 2
    with torch.no_grad():
        torch.testing.assert_close(exported.module()(values), second(torch.nn.functional.gelu(first(values))),
                                   atol=1e-6, rtol=1e-5)
