"""Regression tests for the BABILong log-likelihood scorer.

The scorer (eval/babilong/evaluate.py::score_example) must score exactly the
candidate's own tokens. The pre-fix implementation derived the prompt length
from `tokenizer.encode(prompt)`, which for T5 includes an appended `</s>`, so
the scored positions skipped the first candidate token and included the
trailing `</s>`; every qa1-qa4 answer is a single T5 token, so the scored set
was `['</s>']` for every candidate and every BABILong number was an
end-of-text prior (found 2026-09-02).

Two suites: a FAKE tokenizer suite that runs everywhere (the regression guard
must not vanish on machines without the tokenizer cache — BSC compute nodes
have no internet), and a real-T5 suite that is skipped when the tokenizer is
not cached locally.
"""

import os
import sys
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.babilong.evaluate import (
    STORE_PATH_PEAK_FACTOR,
    _candidate_rows_log_softmax,
    encode_prompt_and_candidates,
    estimate_peak_memory_state_bytes,
    estimate_retrieve_state_bytes,
    score_example,
)
from eval.babilong.prompts import TASK_LABELS, build_prompt

QA1_CANDIDATES = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
PROMPT = "Mary moved to the bathroom. John went to the hallway. Where is Mary? Answer:"
VOCAB = 32128


# ---------------------------------------------------------------------------
# fake tokenizer + stub model (run unconditionally)
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Deterministic whitespace tokenizer with T5's two relevant behaviours:
    `encode(..., add_special_tokens=True)` appends an EOS id, and a candidate
    encoded as " cand" gets the same ids as it does inside the joint string
    (whitespace splitting is context-free, so joint == separate holds). A "-"
    inside a word marks sub-word pieces, giving multi-token candidates on
    demand ("hall-way" -> 2 ids, "of-fi-ce" -> 3 ids)."""

    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = 2

    def __init__(self):
        self.vocab = {}

    def _id(self, piece):
        return self.vocab.setdefault(piece, 3 + len(self.vocab))

    def encode(self, text, add_special_tokens = True):
        ids = [self._id(piece) for word in text.split() for piece in word.split("-")]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids


class StubModel:
    """Stub LM whose 'hidden states' ARE the logits (to_logits is identity) —
    the scorer projects only the rows it needs via return_hidden. `boosts`
    maps absolute position -> token id that gets a +10 logit there. Records
    every ids list it is asked to score."""

    def __init__(self, boosts, vocab = VOCAB):
        self.boosts = boosts
        self.vocab = vocab
        self.calls = []

    def forward(self, ids, disable_flex_attn = False, return_hidden = False):
        assert return_hidden, 'the scorer must request hidden states, not full logits'
        self.calls.append(ids[0].tolist())
        logits = torch.zeros(1, ids.shape[1], self.vocab)
        for pos, token in self.boosts.items():
            if pos < ids.shape[1]:
                logits[0, pos, token] = 10.0
        return logits

    def to_logits(self, hidden):
        return hidden


def _fake_ids(tok, text):
    return tok.encode(text, add_special_tokens = False)


def test_fake_single_token_candidates_scored_with_one_forward():
    """All-single-token candidate sets: ONE forward over the prompt, no
    special token reaches the model, every candidate scored from the last
    row, argmax = the boosted candidate."""
    tok = FakeTokenizer()
    prompt_ids = _fake_ids(tok, PROMPT)
    favoured = "garden"
    boosts = {len(prompt_ids) - 1: _fake_ids(tok, " " + favoured)[0]}
    stub = StubModel(boosts = boosts)

    chosen, log_probs, num_tokens, details = score_example(
        model = stub, tokenizer = tok, prompt_text = PROMPT,
        candidates = QA1_CANDIDATES, device = "cpu",
    )

    assert num_tokens == len(prompt_ids)
    assert len(stub.calls) == 1, 'single-token candidate sets must use one forward'
    assert stub.calls[0] == prompt_ids
    assert tok.eos_token_id not in stub.calls[0]
    assert chosen == favoured
    assert log_probs[favoured] > max(v for k, v in log_probs.items() if k != favoured)
    for cand in QA1_CANDIDATES:
        assert details[cand]["num_tokens"] == 1
        assert abs(details[cand]["sum"] - log_probs[cand]) < 1e-9


def test_fake_multi_token_candidates_scored_over_exactly_their_positions():
    """Mixed token counts: one forward per candidate over prompt + candidate,
    each candidate's rows are exactly its own positions, and the decision
    statistic is sum / true token count."""
    tok = FakeTokenizer()
    prompt_ids = _fake_ids(tok, PROMPT)
    P = len(prompt_ids)
    candidates = ["red", "hall-way", "of-fi-ce"]
    cand_ids = {c: _fake_ids(tok, " " + c) for c in candidates}
    assert [len(cand_ids[c]) for c in candidates] == [1, 2, 3]

    # boost every piece of the 3-token candidate at its own row
    boosts = {P - 1 + j: t for j, t in enumerate(cand_ids["of-fi-ce"])}
    stub = StubModel(boosts = boosts)

    chosen, log_probs, num_tokens, details = score_example(
        model = stub, tokenizer = tok, prompt_text = PROMPT,
        candidates = candidates, device = "cpu",
    )

    assert num_tokens == P
    assert len(stub.calls) == len(candidates), 'multi-token sets fall back to one forward per candidate'
    longest = max(len(ids) for ids in cand_ids.values())
    assert len({len(ids) for ids in stub.calls}) == 1, 'every candidate forward must have the same length'
    for cand, ids in zip(candidates, stub.calls):
        n_cand = len(cand_ids[cand])
        assert ids[:P + n_cand] == prompt_ids + cand_ids[cand], f'model input for {cand!r} is not prompt + candidate'
        # padding sits AFTER the candidate, never inside its scored rows
        assert ids[P + n_cand:] == [tok.pad_token_id] * (longest - n_cand)
        assert tok.eos_token_id not in ids
        assert details[cand]["num_tokens"] == len(cand_ids[cand])
        assert abs(details[cand]["sum"] / details[cand]["num_tokens"] - log_probs[cand]) < 1e-9
    assert chosen == "of-fi-ce"
    # the boosted 3-token candidate's per-token mean is the boosted row value
    # for every one of its tokens; the unboosted ones sit at the flat value
    assert log_probs["of-fi-ce"] > log_probs["red"] and log_probs["of-fi-ce"] > log_probs["hall-way"]


def test_fake_old_indexing_scored_only_eos():
    """The pre-fix indexing, reconstructed on the fake tokenizer: encode WITH
    specials, score positions [len(prompt_ids), len(full_ids)). For a
    single-token candidate this scores exactly [eos] — the bug the fix
    removes. Runs everywhere (no tokenizer cache needed)."""
    tok = FakeTokenizer()
    prompt_old = tok.encode(PROMPT)                 # ends with eos
    assert prompt_old[-1] == tok.eos_token_id
    for cand in QA1_CANDIDATES:
        full_old = tok.encode(PROMPT + " " + cand)  # [prompt..., cand, eos]
        scored_old = full_old[len(prompt_old):]
        assert scored_old == [tok.eos_token_id], f'old indexing scored {scored_old}, expected [eos]'


def test_fake_encode_rejects_specials_and_boundary_mismatch():
    """The runtime guard is a real check, not a tautology: a tokenizer whose
    joint tokenization re-segments the boundary, or one that leaks a special
    token, must be refused."""
    class JointDiffers(FakeTokenizer):
        def encode(self, text, add_special_tokens = True):
            ids = super().encode(text, add_special_tokens = add_special_tokens)
            if text.endswith(" garden") and len(text) > len(" garden"):
                ids = ids[:-1] + [999]          # joint form re-segments the candidate
            return ids

    with pytest.raises(ValueError, match = "joint tokenization"):
        encode_prompt_and_candidates(tokenizer = JointDiffers(), prompt_text = PROMPT, candidates = ["garden"])

    class LeaksEos(FakeTokenizer):
        def encode(self, text, add_special_tokens = True):
            return super().encode(text, add_special_tokens = True)   # ignores the flag

    with pytest.raises(ValueError, match = "special token"):
        encode_prompt_and_candidates(tokenizer = LeaksEos(), prompt_text = PROMPT, candidates = ["garden"])

    # liveness: the plain fake tokenizer passes
    prompt_ids, cand_ids = encode_prompt_and_candidates(
        tokenizer = FakeTokenizer(), prompt_text = PROMPT, candidates = QA1_CANDIDATES
    )
    assert len(prompt_ids) > 0 and set(cand_ids) == set(QA1_CANDIDATES)


def test_prompt_template_is_plain_text():
    """The template must not rely on characters T5 cannot encode (`<`) or on
    newlines its normalizer maps to whitespace; the few-shot scaffold the
    model sees must be the one designed."""
    prompt = build_prompt(task = "qa1", context = "Mary moved to the bathroom.", question = "Where is Mary?")
    assert "<" not in prompt and ">" not in prompt
    assert "\n" not in prompt and "\t" not in prompt
    assert prompt.endswith("Answer:")
    assert "Context: Mary moved to the bathroom. Question: Where is Mary? Answer:" in prompt


def test_qa5_labels_use_in_context_surface_form():
    """qa5's names are capitalized in the dataset; the labels must match that
    form (lowercase names split into 2-3 T5 pieces — a class-correlated
    normalization bias)."""
    assert {"Bill", "Fred", "Jeff", "Mary"} <= TASK_LABELS["qa5"]
    assert not any(label in TASK_LABELS["qa5"] for label in ("bill", "fred", "jeff", "mary"))


def _tiny_atlas_mac(mem_layers, per_head = True, dim = 32, heads = 4, dim_head = 8, seed = 0):
    """A tiny atlas-config MAC in the geometry the tail-quirk test uses
    (store chunk 8, one neural-memory batch segment), memory-free when
    mem_layers is empty."""
    from titans_pytorch import MemoryAsContextTransformer
    from titans_pytorch.neural_memory import NeuralMemory

    torch.manual_seed(seed)
    mem_kwargs = NeuralMemory.atlas_config()
    mem_kwargs.update(
        dim_head = dim_head, heads = heads, use_sequential_scan = True,
        per_head_learned_parameters = per_head,
    )
    return MemoryAsContextTransformer(
        num_tokens = 256, dim = dim, depth = 4, segment_len = 64,
        num_persist_mem_tokens = 4, num_longterm_mem_tokens = 4,
        neural_memory_layers = mem_layers, neural_memory_segment_len = 8,
        neural_memory_batch_size = 1024, use_flex_attn = False,
        sliding_window_attn = True, neural_memory_kwargs = mem_kwargs,
        use_axial_pos_emb = False,
    ).eval()


@pytest.mark.parametrize("per_head", [True, False])
def test_retrieve_state_estimate_matches_actual_retrieve_bytes(monkeypatch, per_head):
    """The retrieve-state estimate is checked against the bytes the retrieve
    actually receives on a real forward (not against a re-statement of the
    formula). Both head-parameter modes: with per_head_learned_parameters
    off, memory_model_parameters lacks the head dim and init_weights repeats
    it `heads` times — an estimate reading the parameter list alone is 4x
    low there (review, 2026-09-02)."""
    from titans_pytorch import neural_memory as nm

    model = _tiny_atlas_mac(mem_layers = (1, 4), per_head = per_head)
    seen = []
    orig = nm.NeuralMemory.retrieve_memories

    def spy(self, seq, weights):
        seen.append(sum(t.numel() * t.element_size() for t in weights.values()))
        return orig(self, seq, weights)

    monkeypatch.setattr(nm.NeuralMemory, "retrieve_memories", spy)
    num_tokens = 100
    with torch.no_grad():
        model(torch.randint(0, 256, (1, num_tokens)), return_hidden = True)

    assert len(seen) == 2, 'one retrieve per memory layer'
    actual = sum(seen)
    estimate = estimate_retrieve_state_bytes(model = model, num_tokens = num_tokens)
    # the retrieve receives positions + 1 states (the initial state M_0 too)
    positions = model.seq_len_with_longterm_mem(num_tokens)
    expected_ratio = (positions + 1) / positions
    assert abs(actual / estimate - expected_ratio) < 0.02, (actual, estimate, per_head)


class LiveStorageTracker(TorchDispatchMode):
    """Allocator-independent high-water mark: the bytes of tensor storages
    created inside the mode that are still referenced, tracked per storage
    (views share one entry) with a refcount released by weakref finalizers
    on the Python tensor wrappers. Storages that already existed (parameters,
    buffers, inputs) are excluded. Same instrument that produced
    STORE_PATH_PEAK_FACTOR."""

    def __init__(self, exclude_tensors):
        super().__init__()
        self.exclude = {t.untyped_storage().data_ptr() for t in exclude_tensors}
        self.live = {}
        self.peak = 0

    def _release(self, ptr):
        entry = self.live.get(ptr)
        if entry is None:
            return
        nbytes, count = entry
        if count <= 1:
            del self.live[ptr]
        else:
            self.live[ptr] = (nbytes, count - 1)

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if not isinstance(t, torch.Tensor):
                continue
            storage = t.untyped_storage()
            ptr = storage.data_ptr()
            if ptr in self.exclude or storage.nbytes() == 0:
                continue
            nbytes, count = self.live.get(ptr, (storage.nbytes(), 0))
            self.live[ptr] = (nbytes, count + 1)
            weakref.finalize(t, self._release, ptr)
        total = sum(nbytes for nbytes, _ in self.live.values())
        self.peak = max(self.peak, total)
        return out


def _measured_peak(model, num_tokens):
    ids = torch.randint(0, 256, (1, num_tokens))
    exclude = list(model.parameters()) + list(model.buffers()) + [ids]
    tracker = LiveStorageTracker(exclude_tensors = exclude)
    with torch.no_grad(), tracker:
        model(ids, return_hidden = True)
    return tracker.peak


def test_peak_memory_estimate_matches_live_storage():
    """The ceiling guard must estimate what the forward PEAKS at, not the
    retrieve tensor alone (a review measured the retrieve-only estimate ~6x
    low — a guard that says 'fits' and then OOMs hours into a cluster eval).
    Measured with the live-storage tracker: one memory layer peaks within
    [0.7, 1.3]x the calibrated estimate and the peak grows linearly with n;
    for two layers the flat factor is an upper bound (measured 4.6-4.7x the
    combined retrieve state vs 6.4x assumed)."""
    one_layer = _tiny_atlas_mac(mem_layers = (1,), dim = 64, heads = 4, dim_head = 16)
    peaks = {}
    for num_tokens in (256, 512):
        peak = _measured_peak(model = one_layer, num_tokens = num_tokens)
        estimate = estimate_peak_memory_state_bytes(model = one_layer, num_tokens = num_tokens)
        ratio = peak / estimate
        assert 0.7 <= ratio <= 1.3, f'n={num_tokens}: measured peak / estimate = {ratio:.2f}'
        peaks[num_tokens] = peak
    growth = peaks[512] / peaks[256]
    assert 1.8 <= growth <= 2.2, f'peak must grow linearly with n, got x{growth:.2f} for 2x tokens'

    # liveness of the instrument: the retrieve-only figure is far below the peak
    retrieve_only = estimate_retrieve_state_bytes(model = one_layer, num_tokens = 512)
    assert peaks[512] / retrieve_only > 4.0

    two_layers = _tiny_atlas_mac(mem_layers = (1, 4), dim = 64, heads = 4, dim_head = 16)
    peak2 = _measured_peak(model = two_layers, num_tokens = 256)
    ratio2 = peak2 / estimate_peak_memory_state_bytes(model = two_layers, num_tokens = 256)
    assert 0.6 <= ratio2 <= 1.0, f'two layers: flat factor {STORE_PATH_PEAK_FACTOR} must be an upper bound, got {ratio2:.2f}'


def test_mixed_length_candidates_padded_to_common_length():
    """Mixed-length candidate sets: every candidate's forward must have the
    same total length. This model is not length-invariant (per-token
    retrieve reads a stale state in an incomplete final store chunk), so
    forwards of different lengths put the shared prompt rows at different
    chunk phases — the first candidate token was scored on different memory
    states depending on how many tokens the OTHER candidate has (review
    measured 1.03 nats). With the pad appended after the candidate the
    prompt-row distribution is identical across candidates at every prompt
    length; without it the instrument shows the divergence."""
    model = _tiny_atlas_mac(mem_layers = (1,))
    tok = FakeTokenizer()
    pad = tok.pad_token_id
    a_ids = _fake_ids(tok, " a")
    bc_ids = _fake_ids(tok, " b-c")
    assert (len(a_ids), len(bc_ids)) == (1, 2)

    padded_devs, raw_devs = [], []
    for length in range(40, 56):
        prompt = " ".join(f"w{i}" for i in range(length))
        prompt_ids = _fake_ids(tok, prompt)
        assert len(prompt_ids) == length
        rows = slice(length - 1, length)          # the row predicting the first candidate token

        def row(ids):
            return _candidate_rows_log_softmax(
                model = model, input_ids = torch.tensor([ids]), rows = rows, disable_flex_attn = True,
            )

        with torch.no_grad():
            row_a_padded = row(prompt_ids + a_ids + [pad])   # the scorer's protocol
            row_bc = row(prompt_ids + bc_ids)
            row_a_raw = row(prompt_ids + a_ids)              # unpadded: one token shorter
        padded_devs.append((row_a_padded - row_bc).abs().max().item())
        raw_devs.append((row_a_raw - row_bc).abs().max().item())

    assert max(padded_devs) < 1e-6, padded_devs
    assert max(raw_devs) > 1e-3, 'instrument liveness: unpadded forwards of different length must diverge somewhere'

    # end to end through the scorer on the real model
    prompt = " ".join(f"w{i}" for i in range(47))
    chosen, log_probs, num_tokens, details = score_example(
        model = model, tokenizer = tok, prompt_text = prompt, candidates = ["a", "b-c"], device = "cpu",
    )
    assert num_tokens == 47 and set(log_probs) == {"a", "b-c"}
    assert details["a"]["num_tokens"] == 1 and details["b-c"]["num_tokens"] == 2


def test_mixed_length_candidates_require_a_pad_token():
    class NoPad(FakeTokenizer):
        pad_token_id = None

    stub = StubModel(boosts = {})
    with pytest.raises(ValueError, match = "pad token"):
        score_example(model = stub, tokenizer = NoPad(), prompt_text = PROMPT, candidates = ["red", "hall-way"], device = "cpu")


# ---------------------------------------------------------------------------
# real T5 tokenizer (skipped when not cached locally)
# ---------------------------------------------------------------------------

@pytest.fixture(scope = "module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("google-t5/t5-base", local_files_only = True)
    except Exception as err:  # noqa: BLE001 — any load failure means "not cached here"
        pytest.skip(f"T5 tokenizer not cached locally: {err}")


def test_t5_scored_ids_are_exactly_the_candidate_tokens(tokenizer):
    """What reaches the model carries no special tokens, every qa1 answer is a
    single T5 token (the geometry under which the old bug scored only </s>),
    and the argmax follows the boosted candidate."""
    eos = tokenizer.eos_token_id
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens = False)
    for favoured in ("garden", "office"):
        favoured_token = tokenizer.encode(" " + favoured, add_special_tokens = False)[0]
        stub = StubModel(boosts = {len(prompt_ids) - 1: favoured_token})
        chosen, log_probs, num_tokens, details = score_example(
            model = stub, tokenizer = tokenizer, prompt_text = PROMPT,
            candidates = QA1_CANDIDATES, device = "cpu",
        )
        assert num_tokens == len(prompt_ids)
        assert stub.calls == [prompt_ids]
        assert eos not in stub.calls[0]
        assert chosen == favoured, f"expected {favoured!r}, got {chosen!r} ({log_probs})"
        for cand in QA1_CANDIDATES:
            cand_ids = tokenizer.encode(" " + cand, add_special_tokens = False)
            assert len(cand_ids) == 1
            assert details[cand]["num_tokens"] == 1


def test_t5_old_indexing_scored_only_eos(tokenizer):
    """The pre-fix indexing on the real tokenizer: for every single-token qa1
    answer it scored exactly [</s>]."""
    eos = tokenizer.eos_token_id
    prompt_old = tokenizer.encode(PROMPT)
    assert prompt_old[-1] == eos, "T5 appends </s>; the old prompt_len counted it"
    for cand in QA1_CANDIDATES:
        full_old = tokenizer.encode(PROMPT + " " + cand)
        scored_old = full_old[len(prompt_old):]
        assert scored_old == [eos], f"old indexing for {cand!r} scored {scored_old}, expected [eos]"


def test_t5_joint_equals_separate_for_every_task_label(tokenizer):
    """The property the runtime guard enforces holds for every label of every
    task at the real template boundary."""
    for task, labels in TASK_LABELS.items():
        prompt = build_prompt(task = task, context = "Mary moved to the bathroom.", question = "Where is Mary?")
        prompt_ids, cand_ids = encode_prompt_and_candidates(
            tokenizer = tokenizer, prompt_text = prompt, candidates = sorted(labels)
        )
        for cand, ids in cand_ids.items():
            assert tokenizer.encode(prompt + " " + cand, add_special_tokens = False) == prompt_ids + ids


def test_t5_prompt_template_has_no_unk(tokenizer):
    """The old `<context>` tags produced 6 <unk> per prompt; the plain
    template must produce none."""
    for task in TASK_LABELS:
        prompt = build_prompt(task = task, context = "Fred gave the apple to Jeff.", question = "Who has the apple?")
        ids = tokenizer.encode(prompt, add_special_tokens = False)
        assert tokenizer.unk_token_id not in ids, f"{task}: template produces <unk>"


def test_t5_qa5_names_are_single_tokens(tokenizer):
    """Capitalized in-context forms are single tokens; the old lowercase
    labels were not (the normalization-bias finding)."""
    for name in ("Bill", "Fred", "Jeff", "Mary"):
        assert len(tokenizer.encode(" " + name, add_special_tokens = False)) == 1
    assert any(len(tokenizer.encode(" " + n, add_special_tokens = False)) > 1 for n in ("fred", "jeff", "mary"))
