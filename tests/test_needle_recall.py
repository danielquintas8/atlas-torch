import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.diagnostics.needle_recall import build_trial, score_tail


def _generator(seed = 0):
    return torch.Generator().manual_seed(seed)


def test_trial_layout_places_the_needle_then_filler_then_cue():
    """[needle][filler][cue]: the cue must be the needle's own prefix, the
    scored tail must never appear before it, and the filler must not contain
    the needle by accident (a disjoint filler range makes that impossible)."""
    ids, needle, length = build_trial(
        generator = _generator(), needle_lo = 0, needle_hi = 50, needle_len = 12, cue_len = 4,
        distance = 100, filler_lo = 50, filler_hi = 80,
    )
    assert length == ids.numel() == 12 + 100 + 4
    assert torch.equal(ids[:12], needle)
    assert torch.equal(ids[-4:], needle[:4])
    filler = ids[12:112]
    assert filler.min() >= 50 and filler.max() < 80, 'filler drawn outside its own range'
    assert needle.max() < 50, 'the needle must come from a range disjoint from the filler'
    assert not any(int(t) in set(filler.tolist()) for t in needle), 'no needle token may appear in the filler'
    tail = needle[4:]
    # the scored tail appears exactly once in the input, inside the first occurrence
    windows = ids.unfold(0, tail.numel(), 1)
    assert int((windows == tail).all(dim = 1).sum()) == 1


def test_distance_and_seed_control_the_trial():
    kwargs = dict(needle_lo = 0, needle_hi = 50, needle_len = 8, cue_len = 3, filler_lo = 50, filler_hi = 100)
    short, _, _ = build_trial(generator = _generator(1), distance = 10, **kwargs)
    long, _, _ = build_trial(generator = _generator(1), distance = 1000, **kwargs)
    assert long.numel() - short.numel() == 990
    again, _, _ = build_trial(generator = _generator(1), distance = 10, **kwargs)
    assert torch.equal(short, again), 'same seed must reproduce the trial'
    other, _, _ = build_trial(generator = _generator(2), distance = 10, **kwargs)
    assert not torch.equal(short, other)


class _CopyModel(torch.nn.Module):
    """A model that can copy: it returns a huge logit for the token that
    followed the last occurrence of the current token earlier in the input,
    and a uniform distribution when it has never seen that token before. It
    stands in for a model with perfect recall at any distance."""

    def __init__(self, vocab_size, window = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.window = window          # None = unlimited memory; an int = only this far back
        self.to_logits = torch.nn.Identity()

    def forward(self, ids, return_hidden = False):
        batch, length = ids.shape
        logits = torch.zeros(batch, length, self.vocab_size)
        for b in range(batch):
            for t in range(length):
                token = int(ids[b, t])
                lowest = 0 if self.window is None else max(0, t - self.window)
                earlier = [i for i in range(lowest, t) if int(ids[b, i]) == token and i + 1 < length]
                if earlier:
                    logits[b, t, int(ids[b, earlier[-1] + 1])] = 20.0
        return logits


def test_score_tail_reads_zero_nll_for_a_perfect_copier_and_the_prior_without_context():
    """The instrument itself: a model that copies from anywhere scores ~0 nats
    on the recalled tail, and the same model scores the uniform prior when the
    needle is absent — so a margin is real recall, not a quirk of the scoring."""
    vocab = 64
    ids, needle, _ = build_trial(generator = _generator(3), needle_lo = 0, needle_hi = vocab // 2, needle_len = 10,
                                 cue_len = 3, distance = 40, filler_lo = vocab // 2, filler_hi = vocab)
    tail = needle[3:]
    copier = _CopyModel(vocab_size = vocab)

    recalled = score_tail(model = copier, ids = ids, tail = tail, device = 'cpu')
    baseline = score_tail(model = copier, ids = ids[10:], tail = tail, device = 'cpu')
    uniform = float(torch.log(torch.tensor(float(vocab))))   # the copier is uniform over the FULL vocab when it has no match

    assert recalled < 0.05, f'a perfect copier must recall the tail (got {recalled:.3f} nats)'
    assert abs(baseline - uniform) < 0.2, f'without the needle the score must be the prior (got {baseline:.3f} vs {uniform:.3f})'
    assert baseline - recalled > 3.0, 'the margin must separate recall from the prior'


def test_a_windowed_model_shows_recall_only_inside_its_window():
    """The reading the probe rests on: a model that can only look back `window`
    tokens recalls at short distances and sits at the prior beyond them. This
    is the shape that distinguishes 'the memory carries the needle' from 'the
    attention window happened to cover it'."""
    vocab = 64
    windowed = _CopyModel(vocab_size = vocab, window = 30)
    margins = {}
    for distance in (8, 200):
        ids, needle, _ = build_trial(generator = _generator(4), needle_lo = 0, needle_hi = vocab // 2, needle_len = 10,
                                     cue_len = 3, distance = distance, filler_lo = vocab // 2, filler_hi = vocab)
        tail = needle[3:]
        recalled = score_tail(model = windowed, ids = ids, tail = tail, device = 'cpu')
        baseline = score_tail(model = windowed, ids = ids[10:], tail = tail, device = 'cpu')
        margins[distance] = baseline - recalled

    assert margins[8] > 3.0, f'inside the window the copier must recall (margin {margins[8]:.2f})'
    assert abs(margins[200]) < 0.2, f'beyond the window it must sit at the prior (margin {margins[200]:.2f})'


def test_shuffle_control_removes_the_margin():
    """The probe's own negative control: replacing the first occurrence with a
    DIFFERENT needle must collapse the margin, for a model that genuinely
    copies. A probe that still shows a margin here is leaking the answer."""
    vocab = 64
    ids, needle, _ = build_trial(generator = _generator(5), needle_lo = 0, needle_hi = vocab // 2, needle_len = 10,
                                 cue_len = 3, distance = 50, filler_lo = vocab // 2, filler_hi = vocab)
    tail = needle[3:]
    copier = _CopyModel(vocab_size = vocab)
    other = torch.randint(0, vocab // 2, (10,), generator = _generator(6))
    mixed = torch.cat((other, ids[10:]))

    baseline = score_tail(model = copier, ids = ids[10:], tail = tail, device = 'cpu')
    recalled = score_tail(model = copier, ids = ids, tail = tail, device = 'cpu')
    control = score_tail(model = copier, ids = mixed, tail = tail, device = 'cpu')

    assert baseline - recalled > 3.0
    # the control must show no RECALL. Scoring worse than the baseline is fine and
    # expected of a confident model that found the cue token inside the wrong needle
    # and predicted its continuation; only a positive margin would mean leakage.
    assert baseline - control < 0.2, f'the shuffle control must not look like recall (margin {baseline - control:+.2f})'


def test_cue_must_be_shorter_than_the_needle():
    """cue_len >= needle_len leaves nothing to score; the CLI refuses it."""
    from eval.diagnostics import needle_recall

    argv = ["--checkpoint", "x", "--model", "170m", "--variant", "atlas-mac", "--needle-len", "4", "--cue-len", "4"]
    saved = sys.argv
    sys.argv = ["needle_recall.py"] + argv
    try:
        with pytest.raises(ValueError, match = "cue_len"):
            needle_recall.main()
    finally:
        sys.argv = saved
