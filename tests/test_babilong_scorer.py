"""Regression tests for the BABILong log-likelihood scorer.

The scorer (eval/babilong/evaluate.py::score_example) must score exactly the
candidate's own tokens. The pre-fix implementation derived the prompt length
from `tokenizer.encode(prompt)`, which for T5 includes an appended `</s>`, so
the scored positions skipped the first candidate token and included the
trailing `</s>`; every qa1-qa4 answer is a single T5 token, so the scored set
was `['</s>']` for every candidate and every BABILong number was an
end-of-text prior (found 2026-09-02). These tests use the real T5 tokenizer
(skipped cleanly if it is not cached locally) and a stub LM.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.babilong.evaluate import score_example

QA1_CANDIDATES = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
PROMPT = "Mary moved to the bathroom. John went to the hallway. Where is Mary? Answer:"
VOCAB = 32128


@pytest.fixture(scope = "module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("google-t5/t5-base", local_files_only = True)
    except Exception as err:  # noqa: BLE001 — any load failure means "not cached here"
        pytest.skip(f"T5 tokenizer not cached locally: {err}")


class FavouringStubModel:
    """Stub LM: flat logits everywhere except a boost for `favoured_token` at
    the position that predicts the first candidate token. Records every ids
    tensor it is asked to score so tests can inspect exactly what was fed."""

    def __init__(self, prompt_len, favoured_token):
        self.prompt_len = prompt_len
        self.favoured_token = favoured_token
        self.calls = []

    def forward(self, ids, disable_flex_attn = False):
        self.calls.append(ids[0].tolist())
        logits = torch.zeros(1, ids.shape[1], VOCAB)
        logits[0, self.prompt_len - 1, self.favoured_token] = 10.0
        return logits


def _run(tokenizer, favoured):
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens = False)
    favoured_token = tokenizer.encode(" " + favoured, add_special_tokens = False)[0]
    stub = FavouringStubModel(prompt_len = len(prompt_ids), favoured_token = favoured_token)
    chosen, log_probs, num_tokens, details = score_example(
        model = stub,
        tokenizer = tokenizer,
        prompt_text = PROMPT,
        candidates = QA1_CANDIDATES,
        device = "cpu",
    )
    return prompt_ids, stub, chosen, log_probs, num_tokens, details


def test_scored_ids_are_exactly_the_candidate_tokens(tokenizer):
    """What reaches the model is prompt_ids + cand_ids with no special tokens,
    and the scored token count equals the candidate's own tokenization."""
    eos = tokenizer.eos_token_id
    prompt_ids, stub, _, log_probs, num_tokens, details = _run(tokenizer, favoured = "garden")

    assert num_tokens == len(prompt_ids)
    assert len(stub.calls) == len(QA1_CANDIDATES)

    for cand, ids in zip(QA1_CANDIDATES, stub.calls):
        cand_ids = tokenizer.encode(" " + cand, add_special_tokens = False)
        assert ids[:len(prompt_ids)] == prompt_ids
        assert ids[len(prompt_ids):] == cand_ids, f"scored ids for {cand!r} are not its own tokens"
        assert eos not in ids, f"eos token reached the model for {cand!r}"
        assert details[cand]["num_tokens"] == len(cand_ids)
        # liveness: the qa1 answers really are single tokens — the geometry
        # under which the old bug scored only </s>
        assert len(cand_ids) == 1
        # normalized statistic is sum / count
        assert abs(details[cand]["sum"] / details[cand]["num_tokens"] - log_probs[cand]) < 1e-9


def test_argmax_returns_the_favoured_candidate(tokenizer):
    for favoured in ("garden", "office"):
        _, _, chosen, log_probs, _, _ = _run(tokenizer, favoured = favoured)
        assert chosen == favoured, f"expected {favoured!r}, got {chosen!r} ({log_probs})"
        assert log_probs[favoured] > max(v for k, v in log_probs.items() if k != favoured)


def test_old_indexing_scored_only_eos(tokenizer):
    """The pre-fix indexing, reconstructed: encode with specials, score
    positions [len(prompt_ids), len(full_ids)). For every single-token qa1
    answer this scored exactly [</s>] — the bug the fix removes."""
    eos = tokenizer.eos_token_id
    prompt_old = tokenizer.encode(PROMPT)
    assert prompt_old[-1] == eos, "T5 appends </s>; the old prompt_len counted it"
    for cand in QA1_CANDIDATES:
        full_old = tokenizer.encode(PROMPT + " " + cand)
        scored_old = full_old[len(prompt_old):]
        assert scored_old == [eos], f"old indexing for {cand!r} scored {scored_old}, expected [eos]"


def test_joint_equals_separate_tokenization(tokenizer):
    """Separate tokenization (the protocol) should coincide with joint
    tokenization at this boundary for the qa1 candidates. If a future
    tokenizer breaks this, the separate protocol still stands — this test
    documents the equivalence, it does not define correctness."""
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens = False)
    for cand in QA1_CANDIDATES:
        cand_ids = tokenizer.encode(" " + cand, add_special_tokens = False)
        joint = tokenizer.encode(PROMPT + " " + cand, add_special_tokens = False)
        assert joint == prompt_ids + cand_ids, f"joint/separate tokenization differ for {cand!r}"
