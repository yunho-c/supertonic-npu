from pathlib import Path
from types import SimpleNamespace
import copy

import pytest
import torch

from scripts.flow_unroll import UnrolledFlow
from scripts.qnn_transformations import optimize_modules, prepare_vector_graph, replace_masked_padding


@pytest.fixture
def flow(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / ".deps/executorch"))
    from examples.models.supertonic.model.vector_estimator import VectorEstimator
    from examples.models.supertonic.source_transformations.mlx import exportable_vector_estimator, replace_same_padding
    torch.manual_seed(23)
    config = SimpleNamespace(ttl=SimpleNamespace(latent_dim=4, chunk_compress_factor=1))
    model = VectorEstimator(config, hidden_channels=8, time_dim=4, time_hidden_channels=8,
        num_main_blocks=2, main_convnext_dilations=(1,), post_time_dilations=(1,),
        post_text_dilations=(1,), final_dilations=(1,), text_channels=8, style_tokens=3,
        style_channels=8, attention_heads=2, style_attention_heads=2, max_positions=16).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("gamma"):
                parameter.fill_(1)
    return exportable_vector_estimator(replace_masked_padding(replace_same_padding(model)))


@pytest.mark.parametrize("unroll", [1, 2, 4, 8])
@pytest.mark.parametrize("reuse,specialize", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("layout", ["original", "pointwise-2d", "gelu-last"])
def test_grouped_flow_matches_each_original_step_and_shares_parameters(flow, unroll, reuse, specialize, layout):
    reference = copy.deepcopy(flow)
    flow = optimize_modules(flow, conv_layout=layout)
    grouped = UnrolledFlow(flow, unroll, 8, reuse, specialize)
    assert {id(p) for p in grouped.parameters()} == {id(p) for p in flow.parameters()}
    for valid_text, valid_latent in ((1, 1), (5, 7)):
        torch.manual_seed(41)
        inputs = [torch.randn(1, 4, 9), torch.randn(1, 8, 6), torch.randn(1, 3, 8),
                  torch.zeros(1, 1, 9), torch.zeros(1, 1, 6)]
        inputs[3][..., :valid_latent] = 1
        inputs[4][..., :valid_text] = 1
        actual, expected = inputs[0], inputs[0]
        with torch.no_grad():
            for start in range(0, 8, unroll):
                actual = grouped(actual, *inputs[1:], torch.tensor([float(start)]), torch.tensor([8.]))
                for step in range(start, start + unroll):
                    expected = reference(expected, *inputs[1:], torch.tensor([float(step)]), torch.tensor([8.]))
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                assert torch.count_nonzero(actual[..., valid_latent:]) == 0


def test_specialized_partial_group_exports_with_runtime_schedule_selection(flow):
    grouped = UnrolledFlow(flow, 2, 8, True, True)
    inputs = (torch.randn(1, 4, 7), torch.randn(1, 8, 5), torch.randn(1, 3, 8),
              torch.ones(1, 1, 7), torch.ones(1, 1, 5), torch.tensor([0.]), torch.tensor([8.]))
    exported = prepare_vector_graph(grouped, inputs)
    later = (*inputs[:5], torch.tensor([6.]), inputs[-1])
    with torch.no_grad():
        torch.testing.assert_close(exported(*later), grouped(*later), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("unroll,schedule,reuse,specialize", [(3, 8, False, False), (4, 5, False, False), (2, 8, False, True)])
def test_invalid_group_configuration_rejected(flow, unroll, schedule, reuse, specialize):
    with pytest.raises(ValueError):
        UnrolledFlow(flow, unroll, schedule, reuse, specialize)
