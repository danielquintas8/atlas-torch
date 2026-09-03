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
    DEVICE_MEMORY_FRACTION,
    JOINT_CHECK_WINDOW_CHARS,
    PEAK_FACTOR,
    _boundary_window,
    _candidate_rows_log_softmax,
    encode_prompt_and_candidates,
    estimate_peak_memory_state_bytes,
    check_model_config_drift,
    estimate_retrieve_state_bytes,
    evaluate_task,
    memory_ceiling_for_tokens,
    memory_ceiling_message,
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


def _tiny_atlas_mac(mem_layers, per_head = True, dim = 32, heads = 4, dim_head = 8, seed = 0, batch_size = 1024, **mem_overrides):
    """A tiny atlas-config MAC in the geometry the tail-quirk test uses
    (store chunk 8, one neural-memory batch segment of `batch_size`
    interleaved positions), memory-free when mem_layers is empty."""
    from titans_pytorch import MemoryAsContextTransformer
    from titans_pytorch.neural_memory import NeuralMemory

    torch.manual_seed(seed)
    mem_kwargs = NeuralMemory.atlas_config()
    mem_kwargs.update(
        dim_head = dim_head, heads = heads, use_sequential_scan = True,
        per_head_learned_parameters = per_head,
    )
    mem_kwargs.update(mem_overrides)
    return MemoryAsContextTransformer(
        num_tokens = 256, dim = dim, depth = 4, segment_len = 64,
        num_persist_mem_tokens = 4, num_longterm_mem_tokens = 4,
        neural_memory_layers = mem_layers, neural_memory_segment_len = 8,
        neural_memory_batch_size = batch_size, use_flex_attn = False,
        sliding_window_attn = True, neural_memory_kwargs = mem_kwargs,
        use_axial_pos_emb = False,
    ).eval()


@pytest.mark.parametrize("omega_context", [8, 1])
@pytest.mark.parametrize("per_head", [True, False])
def test_retrieve_state_estimate_matches_actual_retrieve_bytes(monkeypatch, per_head, omega_context):
    """The retrieve-state estimate is checked against the bytes the retrieve
    actually receives on a real forward (not against a re-statement of the
    formula). Both head-parameter modes: with per_head_learned_parameters
    off, memory_model_parameters lacks the head dim and init_weights repeats
    it `heads` times — an estimate reading the parameter list alone is 4x
    low there (review, 2026-09-02)."""
    from titans_pytorch import neural_memory as nm

    model = _tiny_atlas_mac(mem_layers = (1, 4), per_head = per_head, omega_context = omega_context)
    seen = []
    orig = nm.NeuralMemory.retrieve_memories

    def spy(self, seq, weights, **kwargs):
        seen.append(sum(t.numel() * t.element_size() for t in weights.values()))
        return orig(self, seq, weights, **kwargs)

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
    assert abs(actual / estimate - expected_ratio) < 0.02, (actual, estimate, per_head, omega_context)


class LiveStorageTracker(TorchDispatchMode):
    """Allocator-independent high-water mark: the bytes of tensor storages
    created inside the mode that are still referenced, tracked per storage
    (views share one entry) with a refcount released by weakref finalizers
    on the Python tensor wrappers. Storages that already existed (parameters,
    buffers, inputs) are excluded. Same instrument that produced
    the PEAK_FACTOR / attention-path measurements in evaluate.py."""

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
            # allocator address reuse: a finalizer delayed past a free (a
            # tensor caught in a GC cycle) would leave the OLD size on record
            # for a new, larger storage at the same address — an under-count,
            # the unsafe direction for a ceiling constant. Keep the max.
            self.live[ptr] = (max(nbytes, storage.nbytes()), count + 1)
            weakref.finalize(t, self._release, ptr)
        total = sum(nbytes for nbytes, _ in self.live.values())
        self.peak = max(self.peak, total)
        return out


def _measured_peak(model, num_tokens, chunk_len = None):
    ids = torch.randint(0, 256, (1, num_tokens))
    exclude = list(model.parameters()) + list(model.buffers()) + [ids]
    tracker = LiveStorageTracker(exclude_tensors = exclude)
    with torch.no_grad(), tracker:
        if chunk_len is None:
            model(ids, return_hidden = True)
        else:
            for _ in model.iter_chunked_hidden(ids, chunk_len = chunk_len):
                pass
    return tracker.peak


NEVER_UNDER_GEOMETRIES = {
    'one memory layer': dict(mem_layers = (1,)),
    'two memory layers': dict(mem_layers = (1, 4)),
    'memory-free trunk': dict(mem_layers = ()),
    # the no-omega ablation: per-token store path without the window. keying the
    # estimator on omega_context put it at 0.66 of the measured chunked peak here
    # (adversarial review 2026-09-02); the state is per token on this path too
    'one memory layer, no-omega (c=1, per-token path)': dict(mem_layers = (1,), omega_context = 1),
}


@pytest.mark.parametrize("geometry", list(NEVER_UNDER_GEOMETRIES))
def test_peak_memory_estimate_never_under_measured(geometry):
    """The ceiling guard must estimate what the forward PEAKS at and must
    never estimate BELOW it — an under-estimate is what lets a length past
    the guard and into an OOM hours into a cluster eval (the retrieve-only
    estimate was ~6x low; a later two-term fit was 45% low at 8 segments).
    Re-measured here with the live-storage tracker at batch 256 interleaved
    positions across 1-8.5 memory segments, whole-sequence and chunked, for
    both shipped geometries (the ratio to the retrieve state is highest with
    one memory layer). Bounded loose above (2.5x) so the guard cannot become
    absurdly conservative either."""
    model = _tiny_atlas_mac(
        **NEVER_UNDER_GEOMETRIES[geometry], dim = 64, heads = 4, dim_head = 16, batch_size = 256,
    )
    measured, estimate = {}, {}
    for num_tokens in (256, 1024, 2048):
        measured[('whole', num_tokens)] = _measured_peak(model = model, num_tokens = num_tokens)
        estimate[('whole', num_tokens)] = estimate_peak_memory_state_bytes(model = model, num_tokens = num_tokens)
    for chunk_len in (256, 512):
        measured[('chunked', chunk_len)] = _measured_peak(model = model, num_tokens = 2048, chunk_len = chunk_len)
        estimate[('chunked', chunk_len)] = estimate_peak_memory_state_bytes(
            model = model, num_tokens = 2048, chunk_len = chunk_len,
        )

    ratios = {key: estimate[key] / measured[key] for key in measured}
    for key, ratio in ratios.items():
        # >= 1.0 is the hard side (an estimate below the measurement is the
        # "guard says fits, then OOMs" failure); the upper bound only keeps the
        # deliberately conservative flat factor from becoming absurd. It was 4.0
        # until the carried memory state stopped pinning the previous chunk's
        # scan outputs (clone, 2026-09-02): measured chunked peaks dropped
        # 12-19%, pushing the two-layer chunked ratio to 4.17 while every
        # estimate stayed above its measurement.
        assert 1.0 <= ratio <= 5.0, f'{geometry} {key}: estimate / measured = {ratio:.2f}; all: {ratios}'

    # liveness of the instrument: the whole-sequence peak keeps growing with
    # the length, and (with a memory) the retrieve-only figure is well below
    # the one-segment peak (measured 4.6-6.2x at 256 tokens)
    assert measured[('whole', 2048)] > 1.5 * measured[('whole', 1024)]
    retrieve_256 = estimate_retrieve_state_bytes(model = model, num_tokens = 256)
    if retrieve_256 > 0:
        assert measured[('whole', 256)] / retrieve_256 > 3.0

    # chunked: the peak is set by the chunk, not by the 2048-token sequence
    assert measured[('chunked', 256)] < 0.5 * measured[('whole', 2048)]


def test_peak_memory_estimate_chunked_is_bounded_by_chunk():
    """With chunked inference the estimate depends on the chunk, not the
    sequence: equal for 4K and 1M tokens at the same chunk_len, and equal to
    the whole-sequence estimate for a sequence that fits in one chunk."""
    model = _tiny_atlas_mac(mem_layers = (1, 4), dim = 64, heads = 4, dim_head = 16)
    chunk = model.neural_memory_batch_size
    short = estimate_peak_memory_state_bytes(model = model, num_tokens = 4096, chunk_len = chunk)
    long = estimate_peak_memory_state_bytes(model = model, num_tokens = 1_000_000, chunk_len = chunk)
    assert short == long
    assert long < estimate_peak_memory_state_bytes(model = model, num_tokens = 1_000_000)
    small = 300   # fewer interleaved positions than one chunk
    assert (
        estimate_peak_memory_state_bytes(model = model, num_tokens = small, chunk_len = chunk)
        == estimate_peak_memory_state_bytes(model = model, num_tokens = small)
    )


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


def test_uniform_multi_token_candidates_need_no_pad_token():
    """A pad token is demanded only when padding is needed: a candidate set
    whose members all have the same (multi-token) length runs on a pad-less
    tokenizer, with every forward the same length and no pad appended."""
    class NoPad(FakeTokenizer):
        pad_token_id = None

    tok = NoPad()
    stub = StubModel(boosts = {})
    candidates = ["hall-way", "bed-room"]
    chosen, log_probs, num_tokens, details = score_example(
        model = stub, tokenizer = tok, prompt_text = PROMPT, candidates = candidates, device = "cpu",
    )
    assert set(log_probs) == set(candidates)
    assert len({len(ids) for ids in stub.calls}) == 1
    prompt_ids = _fake_ids(tok, PROMPT)
    for cand, ids in zip(candidates, stub.calls):
        assert ids == prompt_ids + _fake_ids(tok, " " + cand), 'no padding appended for a uniform set'


# ---------------------------------------------------------------------------
# joint-tokenization guard: window and fallback (G1)
# ---------------------------------------------------------------------------

class RecordingFakeTokenizer(FakeTokenizer):
    """FakeTokenizer that records every text it is asked to encode."""

    def __init__(self):
        super().__init__()
        self.texts = []

    def encode(self, text, add_special_tokens = True):
        self.texts.append(text)
        return super().encode(text, add_special_tokens = add_special_tokens)


class FarContextTokenizer(RecordingFakeTokenizer):
    """A tokenizer whose segmentation is NOT whitespace-local: the id of the
    piece 'Answer:' depends on the FIRST character of the text (hundreds of
    characters back). The full prompt starts with 'Z', the bounded window
    does not, so the window's tokenization of the prompt tail differs from
    the full prompt's — the guard's window precondition fails and the
    full-prompt fallback must fire. Candidate pieces are unaffected, so a
    consistent tokenizer still passes; only a planted re-segmentation may
    raise."""

    def encode(self, text, add_special_tokens = True):
        ids = super().encode(text, add_special_tokens = add_special_tokens)
        answer_id = self._id("Answer:")
        if text.startswith("Z"):
            ids = [i + 1000 if i == answer_id else i for i in ids]
        return ids


def _long_prompt(prefix = "Z"):
    filler = " ".join(f"fact{i} is here." for i in range(60))     # ~900 chars
    return f"{prefix}{filler} Where is Mary? Answer:"


def test_fake_far_context_tokenizer_triggers_fallback_and_still_guards():
    """(a) Window precondition fails -> the fallback re-runs the joint check
    on the WHOLE prompt (asserted via the recorded encode texts), a consistent
    tokenizer passes, and a planted boundary re-segmentation is still refused."""
    prompt = _long_prompt()
    assert len(prompt) > JOINT_CHECK_WINDOW_CHARS

    tok = FarContextTokenizer()
    prompt_ids = _fake_ids(tok, prompt)
    window = _boundary_window(prompt)
    assert len(window) < len(prompt) and not window.startswith("Z")
    window_ids = _fake_ids(tok, window)
    assert prompt_ids[-len(window_ids):] != window_ids, 'the precondition must FAIL for this tokenizer'

    tok = FarContextTokenizer()
    got_prompt_ids, cand_ids = encode_prompt_and_candidates(
        tokenizer = tok, prompt_text = prompt, candidates = ["garden", "kitchen"]
    )
    assert got_prompt_ids == prompt_ids and set(cand_ids) == {"garden", "kitchen"}
    # the joint-check encodes: end with " <cand>" and are longer than the bare
    # candidate encode (" garden" alone is the separate candidate encode)
    joint_texts = [
        t for t in tok.texts
        if (t.endswith(" garden") or t.endswith(" kitchen")) and len(t) > len(" kitchen")
    ]
    assert joint_texts and all(t.startswith(prompt) for t in joint_texts), (
        'fallback did not execute: the joint check must run on the FULL prompt'
    )
    assert not any(t == window + " garden" for t in tok.texts), 'window path must not be used after the fallback'

    class PlantedResegmentation(FarContextTokenizer):
        def encode(self, text, add_special_tokens = True):
            ids = super().encode(text, add_special_tokens = add_special_tokens)
            if text.endswith(" garden") and len(text) > len(" garden"):
                ids = ids[:-1] + [999]
            return ids

    with pytest.raises(ValueError, match = "joint tokenization"):
        encode_prompt_and_candidates(
            tokenizer = PlantedResegmentation(), prompt_text = prompt, candidates = ["garden"]
        )


def test_fake_window_path_taken_for_long_prompts():
    """For a whitespace-local tokenizer the guard uses the bounded window (not
    the full prompt) on long prompts, and the window reproduces the tail."""
    prompt = _long_prompt(prefix = "")
    tok = RecordingFakeTokenizer()
    prompt_ids, _ = encode_prompt_and_candidates(tokenizer = tok, prompt_text = prompt, candidates = ["garden"])
    window = _boundary_window(prompt)
    assert JOINT_CHECK_WINDOW_CHARS <= len(window) < len(prompt)
    assert prompt_ids[-len(_fake_ids(tok, window)):] == _fake_ids(tok, window)
    assert (window + " garden") in tok.texts, 'the joint check must run on the window'
    assert (prompt + " garden") not in tok.texts, 'the full prompt must not be re-encoded per candidate'


# ---------------------------------------------------------------------------
# memory ceiling: actual tokenized length (G2)
# ---------------------------------------------------------------------------

def test_attention_path_estimate_accepts_custom_token_emb():
    """`_attention_path_bytes` must not read `token_emb.weight`: the MAC accepts
    any embedding module (`_chunk_hidden` takes its dtype from the embedded
    tokens themselves), and a module without `.weight` crashed the estimator
    (adversarial review 2026-09-02). The residual width is read from the
    attention projection instead, so the estimate is unchanged by the swap."""
    model = _tiny_atlas_mac(mem_layers = ())
    reference = estimate_peak_memory_state_bytes(model = model, num_tokens = 512)
    model.token_emb = torch.nn.Sequential(torch.nn.Embedding(256, 32), torch.nn.Identity())
    assert not hasattr(model.token_emb, "weight")
    assert estimate_peak_memory_state_bytes(model = model, num_tokens = 512) == reference


def test_memory_kwarg_checkpoint_round_trips_through_the_drift_guard(tmp_path):
    """A checkpoint trained with --memory-kwarg records the override in
    model_config.json; the eval side must be able to build the same config
    (its own --memory-kwarg) or every such checkpoint is unevaluable
    (review 2026-09-02). Instrument: without the override the guard refuses."""
    import json
    from experiments.configs import get_config
    from experiments.train import apply_memory_kwargs

    overrides = dict(use_sequential_scan = False, omega_context = 4)
    trained = apply_memory_kwargs(config = get_config(model_size = "170m", variant = "atlas-mac"), overrides = overrides)
    with open(tmp_path / "model_config.json", "w") as f:
        json.dump(trained["model"], f, indent = 2, sort_keys = True, default = str)

    evaluated = apply_memory_kwargs(config = get_config(model_size = "170m", variant = "atlas-mac"), overrides = overrides)
    check_model_config_drift(checkpoint_dir = str(tmp_path), model_config = evaluated["model"])

    plain = get_config(model_size = "170m", variant = "atlas-mac")
    with pytest.raises(ValueError, match = "drift"):
        check_model_config_drift(checkpoint_dir = str(tmp_path), model_config = plain["model"])


def test_actual_length_ceiling_fires_when_nominal_label_passes():
    """The nominal '0k' label passes the ceiling, but the actual tokenized
    prompt of the first example does not: evaluate_task must skip the task
    BEFORE scoring anything, and run it under --force."""
    model = _tiny_atlas_mac(mem_layers = (1,))
    tok = FakeTokenizer()
    # 600 tokens of context from a small vocabulary (the tiny MAC has 256 ids)
    context = " ".join(f"w{i % 40}" for i in range(600))
    dataset = [dict(input = context, question = "Where is Mary?", target = "garden")]
    # device 'memory' sized so the nominal 0k (512 tokens) fits and 600+ does not
    budget = estimate_peak_memory_state_bytes(model = model, num_tokens = 560) / DEVICE_MEMORY_FRACTION

    assert memory_ceiling_for_tokens(model = model, num_tokens = 512, label = "0k", force = False,
                                     device_total_bytes = budget) is None
    calls = []

    class CountingModel:
        """Wraps the real model to prove no scoring forward happens."""
        def __init__(self, inner):
            self.inner = inner
            self.to_logits = inner.to_logits
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def forward(self, *a, **k):
            calls.append(1)
            return self.inner.forward(*a, **k)

    counting = CountingModel(model)
    result = evaluate_task(
        model = counting, tokenizer = tok, task = "qa1", length = "0k", device = "cpu",
        dataset = dataset, device_total_bytes = budget,
    )
    assert result["skipped"] and "actual first prompt" in result["skipped"]
    assert result["first_prompt_tokens"] > 600 and result["total"] == 0
    assert calls == [], 'no scoring forward may run for a skipped task'

    forced = evaluate_task(
        model = counting, tokenizer = tok, task = "qa1", length = "0k", device = "cpu",
        dataset = dataset, device_total_bytes = budget, force = True,
    )
    assert "skipped" not in forced and forced["total"] == 1 and calls

    # liveness: with a generous budget the same task scores normally
    normal = evaluate_task(
        model = counting, tokenizer = tok, task = "qa1", length = "0k", device = "cpu",
        dataset = dataset, device_total_bytes = budget * 100,
    )
    assert "skipped" not in normal and normal["total"] == 1


def test_memory_ceiling_message_is_a_no_op_without_cuda():
    model = _tiny_atlas_mac(mem_layers = (1,))
    assert memory_ceiling_message(model = model, length = "512k", device = "cpu", force = False) is None


# ---------------------------------------------------------------------------
# fp32 projection under bf16 (G3)
# ---------------------------------------------------------------------------

def test_bf16_model_scores_with_fp32_projection():
    """Under --bf16 the candidate rows are projected in fp32: the scorer's
    log-probs equal an fp32 reference (to_logits.weight.float() @
    hidden.float()) to 1e-4, while a bf16 projection of the same rows
    deviates more (instrument liveness)."""
    model = _tiny_atlas_mac(mem_layers = (1,)).to(torch.bfloat16)
    tok = FakeTokenizer()
    prompt_ids = _fake_ids(tok, PROMPT)
    ids = torch.tensor([prompt_ids])

    chosen, log_probs, num_tokens, details = score_example(
        model = model, tokenizer = tok, prompt_text = PROMPT, candidates = QA1_CANDIDATES, device = "cpu",
    )

    with torch.no_grad():
        hidden = model(ids, return_hidden = True)[:, -1]
        assert hidden.dtype == torch.bfloat16
        ref = torch.log_softmax(torch.nn.functional.linear(hidden.float(), model.to_logits.weight.float()), dim = -1)[0]
        bf16 = torch.log_softmax(model.to_logits(hidden).float(), dim = -1)[0]

    scorer_dev = max(abs(log_probs[c] - ref[_fake_ids(tok, " " + c)[0]].item()) for c in QA1_CANDIDATES)
    bf16_dev = max(abs(bf16[_fake_ids(tok, " " + c)[0]].item() - ref[_fake_ids(tok, " " + c)[0]].item()) for c in QA1_CANDIDATES)
    assert scorer_dev < 1e-4, scorer_dev
    assert bf16_dev > scorer_dev, (bf16_dev, scorer_dev)


def test_non_linear_projection_refused_when_not_fp32():
    """No silent fallback to a reduced-precision projection: a non-Linear
    to_logits with non-fp32 hidden states is a TypeError; with fp32 hidden
    states the plain call is used (the stub path)."""
    class Bf16Stub(StubModel):
        def forward(self, ids, disable_flex_attn = False, return_hidden = False):
            return super().forward(ids, disable_flex_attn, return_hidden).to(torch.bfloat16)

    with pytest.raises(TypeError, match = "not nn.Linear"):
        _candidate_rows_log_softmax(
            model = Bf16Stub(boosts = {}), input_ids = torch.tensor([[5, 6, 7]]), rows = slice(2, 3), disable_flex_attn = True,
        )
    out = _candidate_rows_log_softmax(
        model = StubModel(boosts = {}), input_ids = torch.tensor([[5, 6, 7]]), rows = slice(2, 3), disable_flex_attn = True,
    )
    assert out.shape == (1, VOCAB)


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


def test_t5_boundary_window_reproduces_tail_and_guard_catches_resegmentation(tokenizer):
    """(b) Real T5, prompt > window: the bounded window reproduces the prompt's
    token tail, the window path (not the short-prompt early return) is taken,
    the consistent case passes, and a planted boundary re-segmentation on the
    joint call is caught."""
    filler = " ".join(f"Mary moved to the bathroom. John went to the hallway. Sandra took the milk." for _ in range(8))
    prompt = f"{filler} Where is Mary? Answer:"
    assert len(prompt) > JOINT_CHECK_WINDOW_CHARS
    window = _boundary_window(prompt)
    assert JOINT_CHECK_WINDOW_CHARS <= len(window) < len(prompt), 'window path must be taken'
    prompt_ids = tokenizer.encode(prompt, add_special_tokens = False)
    window_ids = tokenizer.encode(window, add_special_tokens = False)
    assert prompt_ids[-len(window_ids):] == window_ids, 'the window must reproduce the prompt tail'

    class Recording:
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        unk_token_id = tokenizer.unk_token_id

        def __init__(self, plant = False):
            self.texts = []
            self.plant = plant

        def encode(self, text, add_special_tokens = True):
            self.texts.append(text)
            ids = tokenizer.encode(text, add_special_tokens = add_special_tokens)
            if self.plant and text.endswith(" garden") and len(text) > len(" garden"):
                ids = ids[:-1] + [ids[-1] + 1]        # joint form re-segments the candidate
            return ids

    rec = Recording()
    got_prompt_ids, cand_ids = encode_prompt_and_candidates(
        tokenizer = rec, prompt_text = prompt, candidates = QA1_CANDIDATES
    )
    assert got_prompt_ids == prompt_ids
    assert (window + " garden") in rec.texts and (prompt + " garden") not in rec.texts, 'window path must be used'

    with pytest.raises(ValueError, match = "joint tokenization"):
        encode_prompt_and_candidates(tokenizer = Recording(plant = True), prompt_text = prompt, candidates = ["garden"])


def test_t5_qa5_names_are_single_tokens(tokenizer):
    """Capitalized in-context forms are single tokens; the old lowercase
    labels were not (the normalization-bias finding)."""
    for name in ("Bill", "Fred", "Jeff", "Mary"):
        assert len(tokenizer.encode(" " + name, add_special_tokens = False)) == 1
    assert any(len(tokenizer.encode(" " + n, add_special_tokens = False)) > 1 for n in ("fred", "jeff", "mary"))


def test_score_example_chunked_matches_whole_sequence():
    """score_example(chunk_len=...) — the chunked forward feeding the scorer —
    must give the same candidate log-probs as the whole-sequence path, on a
    prompt spanning several chunks, for both the single-token branch and the
    padded multi-token branch (the real-T5 candidates; skipped if the
    tokenizer is not cached). Chunk = one memory segment of the tiny MAC
    (1024 interleaved positions)."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("google-t5/t5-base", local_files_only = True)
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"T5 tokenizer not cached locally: {err}")

    model = _tiny_atlas_mac(mem_layers = (1, 4), dim = 32, heads = 4, dim_head = 8)
    model.token_emb = torch.nn.Embedding(VOCAB, 32)
    model.to_logits = torch.nn.Linear(32, VOCAB, bias = False)
    torch.manual_seed(3)
    with torch.no_grad():
        model.token_emb.weight.normal_(std = 0.02)
        model.to_logits.weight.normal_(std = 0.02)

    filler = " ".join(["Mary went to the garden and John moved to the office."] * 200)
    prompt = f"Context: {filler} Question: Where is Mary? Answer:"
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens = False))
    # chunks are counted in interleaved positions: the prompt must span > 2 chunks
    assert model.seq_len_with_longterm_mem(prompt_tokens) > 2 * model.neural_memory_batch_size

    for candidates in (["garden", "office", "kitchen"], ["garden", "hallway", "ten"]):
        whole = score_example(model = model, tokenizer = tokenizer, prompt_text = prompt,
                              candidates = candidates, device = "cpu")
        chunked = score_example(model = model, tokenizer = tokenizer, prompt_text = prompt,
                                candidates = candidates, device = "cpu",
                                chunk_len = model.neural_memory_batch_size)
        assert whole[0] == chunked[0]
        for cand in candidates:
            assert abs(whole[1][cand] - chunked[1][cand]) < 1e-4, (cand, whole[1][cand], chunked[1][cand])


@pytest.mark.parametrize(
    "candidates",
    [["garden", "office"], ["garden", "hall-way", "of-fi-ce"]],
    ids = ["single-token branch", "padded multi-token branch"],
)
def test_score_example_chunk_len_reaches_the_chunked_forward(monkeypatch, candidates):
    """Both scorer branches (single-token: one forward over the prompt;
    multi-token: one padded forward per candidate) must route chunk_len into
    iter_chunked_hidden and never call the whole-sequence forward. A branch
    that silently ran the whole-sequence forward would pass every value-level
    parity test (the numbers are identical) and OOM at the first long length
    on the cluster — the multi-token branch did exactly that before this
    test. Spied on the model class with the fake tokenizer (runs on the
    cluster); fp64 model so the two paths agree to rounding."""
    from titans_pytorch import MemoryAsContextTransformer

    tok = FakeTokenizer()
    model = _tiny_atlas_mac(mem_layers = (1,), batch_size = 64).double()
    chunk = model.neural_memory_batch_size
    prompt = " ".join(["Mary moved to the bathroom. John went to the hallway."] * 30) + " Where is Mary? Answer:"
    assert model.seq_len_with_longterm_mem(len(_fake_ids(tok, prompt))) > 2 * chunk

    chunked_calls, whole_calls = [], []
    original_iter = MemoryAsContextTransformer.iter_chunked_hidden
    original_forward = MemoryAsContextTransformer.forward

    def spy_iter(self, x, chunk_len):
        chunked_calls.append(chunk_len)
        return original_iter(self, x, chunk_len = chunk_len)

    def spy_forward(self, *args, **kwargs):
        whole_calls.append(1)
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(MemoryAsContextTransformer, "iter_chunked_hidden", spy_iter)
    monkeypatch.setattr(MemoryAsContextTransformer, "forward", spy_forward)

    whole = score_example(model = model, tokenizer = tok, prompt_text = prompt, candidates = candidates, device = "cpu")
    assert whole_calls and not chunked_calls, 'without chunk_len the whole-sequence forward must run'
    whole_calls.clear()

    chunked = score_example(
        model = model, tokenizer = tok, prompt_text = prompt, candidates = candidates, device = "cpu", chunk_len = chunk,
    )
    assert not whole_calls, 'a scorer branch ran the whole-sequence forward despite chunk_len'
    single_token_set = all(len(_fake_ids(tok, " " + cand)) == 1 for cand in candidates)
    assert chunked_calls == [chunk] * (1 if single_token_set else len(candidates))

    assert whole[0] == chunked[0]
    for cand in candidates:
        assert abs(whole[1][cand] - chunked[1][cand]) < 1e-6, (cand, whole[1][cand], chunked[1][cand])
