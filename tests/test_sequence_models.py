"""Sequence models must ignore the padding after an event's last real frame."""

import pytest

torch = pytest.importorskip("torch")

from sequence_classifier import BiLSTM, MeanPoolMLP, TemporalCNN, TransformerClassifier  # noqa: E402


@pytest.mark.parametrize("cls", [BiLSTM, MeanPoolMLP, TransformerClassifier])
def test_padding_contents_do_not_change_the_output(cls):
    torch.manual_seed(0)
    model = cls(8, 16, 3, 10).eval()
    x = torch.randn(2, 10, 8)
    m = torch.ones(2, 10)
    m[0, 6:] = 0  # the first event has only 6 real frames
    garbage = x.clone()
    garbage[0, 6:] = 100.0
    with torch.no_grad():
        assert torch.allclose(model(x, m), model(garbage, m), atol=1e-5)
        # ... and equals the same event given alone, without padding
        alone = model(x[:1, :6], m[:1, :6])
    assert torch.allclose(model(x, m)[0], alone[0], atol=1e-5)


def test_temporal_cnn_runs_on_padded_batches():
    model = TemporalCNN(8, 16, 3, 10).eval()
    m = torch.ones(2, 10)
    m[0, 6:] = 0
    with torch.no_grad():
        assert model(torch.randn(2, 10, 8), m).shape == (2, 3)
