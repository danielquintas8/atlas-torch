import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.diagnostics.state_norm_probe import bucket_stats
from titans_pytorch.neural_memory import NeuralMemory


def test_bucket_stats_averages_within_each_bucket_and_keeps_a_short_tail():
    values = [1.0] * 4 + [3.0] * 4 + [10.0] * 2
    stats = bucket_stats(values = values, positions = None, bucket = 4)
    assert stats == {"0-4": 1.0, "4-8": 3.0, "8-10": 10.0}, stats


def _memory(spectral_norm):
    torch.manual_seed(0)
    kwargs = NeuralMemory.atlas_config(dim_head = 8, heads = 2, use_sequential_scan = True,
                                       spectral_norm_surprises = spectral_norm, short_conv_size = 0)
    return NeuralMemory(dim = 16, chunk_size = 8, batch_size = 32, **kwargs).double().eval()


def test_flipping_spectral_norm_on_a_built_module_changes_the_update():
    """The probe's intervention is `mem.spectral_norm_surprises = ...` on a
    LOADED model, so the flag must actually reach the update path: a module
    built with Newton-Schulz off and switched on must produce the output of one
    built with it on, and vice versa. If the attribute were read only at
    construction the intervention would silently do nothing — the probe would
    then 'show' that Newton-Schulz changes nothing, which is the exact wrong
    conclusion."""
    torch.manual_seed(1)
    seq = torch.randn(1, 64, 16, dtype = torch.float64)

    with torch.no_grad():
        on_out, _ = _memory(spectral_norm = True)(seq)
        off_out, _ = _memory(spectral_norm = False)(seq)
        flipped_on = _memory(spectral_norm = False)
        flipped_on.spectral_norm_surprises = True
        flipped_on_out, _ = flipped_on(seq)
        flipped_off = _memory(spectral_norm = True)
        flipped_off.spectral_norm_surprises = False
        flipped_off_out, _ = flipped_off(seq)

    assert not torch.allclose(on_out, off_out, atol = 1e-8), 'instrument: the two settings must differ at all'
    assert torch.equal(flipped_on_out, on_out), 'flipping the attribute on must match a module built with it on'
    assert torch.equal(flipped_off_out, off_out), 'flipping it off must match a module built with it off'


def test_newton_schulz_bounds_the_update_norm_which_is_the_hypothesis_under_test():
    """The reason Newton-Schulz is the prime suspect for the chunk-wise
    divergence: it orthogonalizes the update, so the update's norm is set by
    its shape rather than by the surprise magnitude. Scaling the incoming
    surprise 100x must leave the Newton-Schulz output's norm essentially
    unchanged while the raw path's grows with it."""
    from titans_pytorch.neural_memory import newtonschulz5

    torch.manual_seed(2)
    update = torch.randn(2, 4, 8, 24, dtype = torch.float64)
    small = newtonschulz5(update, recompute = False)
    large = newtonschulz5(update * 100, recompute = False)

    assert abs(large.norm() / small.norm() - 1.0) < 1e-6, 'Newton-Schulz must be scale-invariant'
    assert (update * 100).norm() / update.norm() == pytest.approx(100.0), 'instrument: the raw update does scale'
