"""
BABILong evaluation for Atlas/Titans models.

Evaluates trained MAC transformer checkpoints on the BABILong benchmark,
testing long-context recall at increasing context lengths.

Usage:
    # Evaluate on specific tasks and lengths:
    python eval/babilong/evaluate.py \
        --checkpoint runs/170m-atlas-mac/step-1000 \
        --model 170m --variant atlas-mac \
        --lengths 0k 4k 16k 64k \
        --tasks qa1 qa2 qa3 qa4 qa5 \
        --output results/atlas-mac-babilong.json

    # Quick smoke test:
    python eval/babilong/evaluate.py \
        --checkpoint runs/170m-atlas-mac/step-1000 \
        --model 170m --variant atlas-mac \
        --lengths 0k 4k --tasks qa1 --max-examples 10

    # BSC (via SLURM):
    sbatch eval/babilong/evaluate_slurm.sh

Scoring protocol (fixed 2026-09-02): context and candidate are tokenized
SEPARATELY with no special tokens and concatenated, and exactly the
candidate's token positions are scored. The previous implementation derived
the prompt length from `tokenizer.encode(prompt)`, which for T5 includes an
appended `</s>`, so the scored positions skipped the first candidate token
and included the trailing `</s>`; every qa1-qa4 answer is a single T5 token,
so the scored set was `['</s>']` for every candidate — every BABILong number
produced before this fix was an end-of-text prior, not an answer likelihood.

Memory ceiling (measured 2026-09-02): with per-token retrieve the model's
whole-sequence forward materializes every token's memory-weight state for the
final retrieve — 0.59 MB/token PER memory layer in bf16 at 170M (1.18 MB/token
for the two shipped layers; double in fp32): 4.8 GB @4K, 19 GB @16K, 39 GB
@32K, 78 GB @64K, 1.2 TB @1M for the retrieve state alone. The forward peaks
well above that (live-tensor high-water tracker, tiny atlas MACs, CPU): the
store path holds several per-token-sized copies on top of the retrieve state
(grads, negated surprises, momentum, post-Newton-Schulz update, scan outputs,
the per-segment `updates` list, the final concatenation, the retrieve's
rearranged copy) plus the attention / residual path's own copies — measured
2.8-10x the retrieve state across toy geometries and lengths (highest within
one memory segment and where the attention path is a large share; with that
path subtracted the memory-only ratio is 3.7-5.6x within one segment and
1-1.6x at 4-8 segments). The estimate is PEAK_FACTOR (8, above every raw
ratio) x R(positions processed at once) + the attention / residual path
(_attention_path_bytes), never under any measured peak; a length is refused
past 80% of device memory unless --force is given. On a 64 GB H100 at
170M/2 layers that is ~9.6 MB per interleaved position: roughly 5K tokens
whole-sequence in bf16 under the estimate (~2.5K fp32; the measured
two-layer ratios would allow 2-3x more — chunked inference makes the
whole-sequence cap moot).

Chunked inference (--chunk-len, MemoryAsContextTransformer.iter_chunked_hidden)
removes the ceiling: the sequence is processed in chunks of `chunk_len`
INTERLEAVED positions (a multiple of neural_memory_batch_size — 1024 in the
shipped config — so chunk boundaries coincide with the memory's segment
boundaries), carrying only the memory state and each attention layer's
un-rotated keys/values for the last two attention segments across chunks
(each chunk is folded into attention segments exactly as the whole-sequence
forward folds the full sequence, so attention memory is linear in the chunk).
Output equals the whole-sequence forward (parity-tested, fp64-exact); peak
memory is O(chunk): ~10 GB by the estimate (~6 GB at the measured ratios) at
170M in bf16 with 1024-position chunks, at any context length. Cheap store-path win left for
future work: `grads` stays alive after `surprises = grads.mul(-1)` in
NeuralMemory.store_memories, so one per-token copy is avoidable under no_grad
— deliberately not changed (the training path shares that code and autograd
needs the original tensor).
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from titans_pytorch import MemoryAsContextTransformer
from experiments.configs import get_config
from eval.babilong.prompts import TASK_LABELS, build_prompt


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def normalize_model_config(model_config):
    """JSON round-trip so tuples become lists and non-JSON values (dtypes,
    modules) become strings — the form written next to a checkpoint by
    train.save_checkpoint and compared on load."""
    return json.loads(json.dumps(model_config, sort_keys=True, default=str))


def check_model_config_drift(checkpoint_dir, model_config):
    """Compare the config being built against the checkpoint's recorded
    `model_config.json`. Strict state-dict loading only catches PARAMETER-set
    drift; a non-parameter field (neural_memory_batch_size, omega_context,
    per_token_retrieve, ...) changes eval semantics without changing any
    shape and would load clean. Raise on any differing key; warn when the
    checkpoint predates the recording."""
    path = os.path.join(checkpoint_dir, "model_config.json")
    if not os.path.exists(path):
        print(
            f"WARNING: {path} missing (checkpoint predates model_config.json "
            f"recording) — cannot verify the checkpoint was trained with this "
            f"model config; only parameter-set drift is checked (strict load)."
        )
        return
    with open(path) as f:
        saved = f.read()
    saved = json.loads(saved)
    current = normalize_model_config(model_config)
    differing = sorted(
        key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
    )
    if differing:
        details = "\n".join(
            f"  {key}: checkpoint={saved.get(key)!r} current={current.get(key)!r}"
            for key in differing
        )
        raise ValueError(
            f"model config drift between {path} and the config being built — "
            f"evaluating would run different semantics than training:\n{details}"
        )


def load_model(checkpoint_dir, model_size, variant, ablation=None, device="cuda", vanilla=False, dtype=None):
    """Load model from accelerate checkpoint.

    `vanilla` mirrors train.py --vanilla (neural_memory_layers=()): a
    memory-free baseline checkpoint has no NeuralMemory weights, so it can
    only load into a memory-free model.
    """
    config = get_config(model_size=model_size, variant=variant, ablation=ablation)
    if vanilla:
        config["model"]["neural_memory_layers"] = ()
    check_model_config_drift(checkpoint_dir=checkpoint_dir, model_config=config["model"])
    model = MemoryAsContextTransformer(**config["model"])

    # accelerate saves model weights in a subdirectory
    weights_file = None
    for name in ["model.safetensors", "pytorch_model.bin"]:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            weights_file = path
            break

    if weights_file is None:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )

    if weights_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_file, device="cpu")
    else:
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

    # Strip 'module.' prefix if saved with DDP
    cleaned = {}
    for k, v in state_dict.items():
        key = k.removeprefix("module.")
        cleaned[key] = v

    # Strict load, deliberately: a checkpoint whose parameter set does not
    # match the current config is not evaluable — its store semantics differ
    # (omega window fix), it lacks value_conv, or it carries the axial
    # positional embedding that MAC_DEFAULTS now disables. Loading it with
    # strict=False would silently run a randomly-initialized value conv or
    # drop the positional embedding the model was trained with.
    try:
        model.load_state_dict(cleaned)
    except RuntimeError as err:
        raise RuntimeError(
            f"Checkpoint {weights_file} does not match the current model config "
            f"({model_size} / {variant}" + (f" / {ablation}" if ablation else "") + "). "
            f"Likely cause: the checkpoint predates one of — the omega window fix "
            f"(different store semantics), the value_conv addition, or "
            f"use_axial_pos_emb=False in MAC_DEFAULTS (checkpoint carries "
            f"axial_pos_emb.* weights). Such checkpoints are not evaluable on this "
            f"code; retrain. Original error:\n{err}"
        ) from err
    model = model.to(device).eval()
    if dtype is not None:
        model = model.to(dtype)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {param_count / 1e6:.1f}M params from {weights_file}")

    return model, config


def load_tokenizer(tokenizer_dir):
    """Load tokenizer from directory."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_dir)


# ---------------------------------------------------------------------------
# Memory ceiling
# ---------------------------------------------------------------------------

def memory_layers(model):
    """The NeuralMemory modules of a MemoryAsContextTransformer (layer slot 4)."""
    return [layer[4] for layer in model.layers if layer[4] is not None]


# Peak live storage of a no_grad forward, relative to the retrieve state R(N)
# (N = interleaved positions processed at once). Measured 2026-09-02 with an
# allocator-independent live-tensor high-water tracker (a TorchDispatchMode
# summing the bytes of storages created inside the forward that are still
# referenced; parameters and inputs excluded — see
# tests/test_babilong_scorer.py::LiveStorageTracker) on CPU, tiny atlas MACs
# with per-token retrieve, neural_memory_batch_size 256 interleaved
# positions, one and two memory layers, 256-2048 tokens (1-8.5 memory
# segments), whole-sequence and chunked (256 / 512-position chunks).
#
# The store path holds several per-token-sized copies on top of the retrieve
# state (grads, negated surprises, momentum, post-Newton-Schulz update, scan
# outputs, the per-segment `updates` list, the final concatenation, the
# retrieve's rearranged copy), and the attention / residual path holds its
# own per-position copies, which at toy scale are a large share of the peak:
#
#     measured peak / R(N)          whole 256 / 1024 / 2048 tokens   chunked 256 / 512
#     dim 32, one memory layer       9.96 / 9.25 / 9.13               12.4 / 9.8
#     dim 32, two memory layers      6.02 / 5.33 / 5.21                7.2 / 5.4
#     dim 64, one memory layer       6.21 / 3.54 / 3.42                8.8 / 5.5
#     dim 64, two memory layers      4.60 / 2.90 / 2.78                5.8 / 3.6
#
# (memory dim_head 8 / 16 with 4 heads; attention 8 heads x 64 at both, so
# the attention path is a larger share at dim 32 — the raw ratio is not a
# property of the memory alone, which is why an earlier two-term fit of it
# (3.4x asymptote + a one-segment transient, no attention term) was up to
# 45% under on another geometry.) With the attention / residual path
# estimated separately (_attention_path_bytes, never under the memory-free
# trunk by 9-17%) and subtracted, the memory-only ratio is 3.7-5.6 within
# one segment and 1-1.6 at 4-8 whole-sequence segments. The estimate keeps
# one flat factor above every raw ratio on the retrieve state of the
# positions processed at once, plus the attention path — deliberately
# conservative (estimate / measured 1.27-3.32 with a memory layer, largest
# for long whole-sequence runs, which chunked inference replaces anyway)
# until a two-term memory model is validated on the target device.
#
# tests/test_babilong_scorer.py::test_peak_memory_estimate_never_under_measured
# re-measures on every run and fails if the estimate drops under a measured
# peak or exceeds 4x it. CUDA-calibrate on the first BSC run with
# torch.cuda.max_memory_allocated() and update: this is a CPU measurement of
# the mechanism, not of the target device.
PEAK_FACTOR = 8.0

# attention / residual path (see _attention_path_bytes), copy counts per
# folded row or per position, from the same tracker on the memory-free
# trunk (never under at either toy geometry, whole and chunked):
ATTENTION_HEAD_COPIES = 20        # q, k, v (3), rotated q, k (2), k, v with the previous window (4) and with the persistent tokens (4), out + merged (2), margin
ATTENTION_SCORE_COPIES = 2        # scores + probabilities
RESIDUAL_COPIES_PER_LAYER = 4     # attn_in, attn_out, ff_in, ff_out kept for the memory's qkv selector
RESIDUAL_TRANSIENT_COPIES = 48    # hyper-connection streams, feed-forward inner activations (~5 dim), norms, residual adds


def estimate_retrieve_state_bytes(model, num_tokens):
    """Bytes of memory-weight state the whole-sequence forward holds for the
    final retrieve at `num_tokens` input tokens — the retrieve tensor ALONE.
    The forward peaks well above it; see estimate_peak_memory_state_bytes.

    Every memory layer's decay scan emits one weight state per token when the
    omega rule is on (per-token store granularity; the per-token retrieve
    reads them all at once), or one per store chunk otherwise. The states for
    ALL tokens of the sequence are concatenated before the retrieve, so the
    footprint is linear in the sequence length. The interleaved longterm-mem
    tokens count too: the memory sees seq_len_with_longterm_mem(num_tokens)
    positions. `memory_model_parameters` carries the head dimension only when
    per_head_learned_parameters is on; otherwise init_weights repeats the
    shared parameters `heads` times, so the state is `heads` times larger
    than the parameter list suggests.
    """
    positions = model.seq_len_with_longterm_mem(num_tokens) if num_tokens > 0 else 0
    return int(sum(_per_position_state_bytes(mem) for mem in memory_layers(model)) * positions)


def _per_position_state_bytes(mem):
    """Bytes of memory-weight state one memory layer emits per interleaved
    position (per store chunk when the omega rule is off)."""
    state_values = sum(p.numel() for p in mem.memory_model_parameters)
    if not mem.per_head_learned_parameters:
        state_values *= mem.heads
    elem_bytes = next(iter(mem.memory_model_parameters)).element_size()
    per_position = state_values * elem_bytes
    if mem.omega_context <= 1:
        per_position /= mem.store_chunk_size
    return per_position


def _attention_path_bytes(model, positions, chunked):
    """Bytes held by the attention / residual path at the peak of a forward
    over `positions` interleaved positions (one chunk of them when
    `chunked`): one layer's head-sized transients over the folded rows
    (q / k / v, rotated q / k, k / v with the previous window and with the
    persistent tokens, the attention output and its merge), its scores and
    probabilities (rows x heads x (2 segments + persistent tokens)), every
    layer's un-rotated keys / values (the whole-sequence forward keeps all
    of them until it returns; the chunked one holds two segments per layer),
    and the residual-stream copies: the four per-layer branch tensors the
    memory's qkv selector keeps (`mem_input_layers`, twice when chunked —
    the previous chunk's die as the next is built) plus transient copies
    (hyper-connection streams, feed-forward inner activations, norms). Small
    next to the memory state for any model with a memory layer; it is what
    bounds a memory-free (vanilla) model."""
    attn = model.layers[0][5]
    _, heads, num_persist, dim_head = attn.persistent_memory.shape
    window = attn.total_segment_len
    elem = attn.to_qkv.weight.element_size()
    depth = len(model.layers)
    dim = model.token_emb.weight.shape[1]

    # the folded rows: the chunk plus up to two cached segments in front of it
    rows = positions + (2 * window if chunked else 0)
    head_transients = ATTENTION_HEAD_COPIES * heads * dim_head * elem * rows
    scores = ATTENTION_SCORE_COPIES * heads * (2 * window + num_persist) * elem * rows
    kv_caches = 2 * heads * dim_head * elem * depth * (min(positions, 2 * window) if chunked else positions)
    residual_copies = RESIDUAL_COPIES_PER_LAYER * depth * (2 if chunked else 1) + RESIDUAL_TRANSIENT_COPIES
    residual = residual_copies * dim * elem * positions
    return head_transients + scores + kv_caches + residual


def estimate_peak_memory_state_bytes(model, num_tokens, chunk_len=None):
    """Estimated peak live storage of a no_grad forward at `num_tokens` input
    tokens: PEAK_FACTOR x the retrieve state of the positions processed at
    once, plus the attention / residual path's per-position bytes (see the
    constants' note for the measurement and the never-under test).

    `chunk_len` (interleaved positions) = chunked inference: at most
    `chunk_len` positions are processed at once and only the compact memory
    state and two attention segments of keys / values cross chunk boundaries,
    so the estimate is O(chunk_len) and independent of the sequence length."""
    positions = model.seq_len_with_longterm_mem(num_tokens) if num_tokens > 0 else 0
    chunked = chunk_len is not None and positions > chunk_len
    if chunked:
        positions = chunk_len

    per_position = [_per_position_state_bytes(mem) for mem in memory_layers(model)]

    return int(
        PEAK_FACTOR * sum(per_position) * positions
        + _attention_path_bytes(model=model, positions=positions, chunked=chunked)
    )


def length_label_to_tokens(length):
    """BABILong split label ('0k', '4k', '64k', ...) -> approximate context
    tokens. '0k' is the bare bAbI facts (a few hundred tokens)."""
    k = int(length.rstrip("k"))
    return k * 1024 if k > 0 else 512


DEVICE_MEMORY_FRACTION = 0.8


def device_total_memory_bytes(device):
    """Total memory of a CUDA device, or None when the device is not CUDA /
    CUDA is unavailable (the ceiling check is then skipped)."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(torch.device(device)).total_memory


def memory_ceiling_for_tokens(model, num_tokens, label, force, device_total_bytes, chunk_len=None):
    """Return None when the estimated PEAK memory-state footprint at
    `num_tokens` input tokens (see estimate_peak_memory_state_bytes) fits
    within DEVICE_MEMORY_FRACTION of `device_total_bytes`, else a message
    explaining the refusal (unless `force`, which downgrades it to a warning
    printed by the caller). `device_total_bytes=None` disables the check (no
    CUDA). With `chunk_len` (chunked inference) the estimate is O(chunk)."""
    if device_total_bytes is None:
        return None
    retrieve = estimate_retrieve_state_bytes(model=model, num_tokens=num_tokens)
    peak = estimate_peak_memory_state_bytes(model=model, num_tokens=num_tokens, chunk_len=chunk_len)
    if peak <= DEVICE_MEMORY_FRACTION * device_total_bytes:
        return None
    mode = (
        f"chunked inference with {chunk_len:,}-position chunks would still peak"
        if chunk_len is not None else
        f"the whole-sequence forward would hold ~{retrieve / 1e9:.1f} GB of per-token "
        f"memory-weight state for the final retrieve and peak"
    )
    return (
        f"{label} ({num_tokens:,} tokens): {mode} at ~{peak / 1e9:.1f} GB of live storage "
        f"(x{PEAK_FACTOR:g} the retrieve state of the positions processed at once, plus the "
        f"attention path; device total {device_total_bytes / 1e9:.1f} GB, "
        f"limit {DEVICE_MEMORY_FRACTION:.0%}). At 170M with two memory layers a 64 GB GPU fits "
        f"roughly 5K tokens whole-sequence in bf16 under this estimate (~2.5K fp32); "
        f"--chunk-len 1024 makes the footprint O(chunk) (a few GB) at any length; --bf16 halves it. "
        + ("Running anyway (--force)." if force else "Skipping (pass --force to try).")
    )


def memory_ceiling_message(model, length, device, force, chunk_len=None):
    """Early exit on the NOMINAL split label ('4k' -> 4096 tokens). BABILong's
    labels are not T5 token counts (English prose runs ~1.15-1.3x longer under
    T5's 32K vocab, and the few-shot scaffold adds more), so evaluate_task
    re-checks against the ACTUAL tokenized prompt before scoring."""
    return memory_ceiling_for_tokens(
        model=model,
        num_tokens=length_label_to_tokens(length),
        label=f"context {length} (nominal)",
        force=force,
        device_total_bytes=device_total_memory_bytes(device),
        chunk_len=chunk_len,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

JOINT_CHECK_WINDOW_CHARS = 256


def _boundary_window(prompt_text):
    """The suffix of the prompt used for the joint-tokenization check: cut at
    a whitespace boundary at least JOINT_CHECK_WINDOW_CHARS from the end, or
    the whole prompt when it is shorter than the window (or has no earlier
    whitespace). SentencePiece/BPE segmentation is whitespace-local — a piece
    never spans whitespace (T5's normalizer turns every whitespace run into
    the '▁' word boundary) — so how the prompt's last words and the appended
    candidate tokenize depends only on the text from the last boundary on.
    Re-encoding a bounded suffix keeps the check O(window) per candidate
    instead of O(prompt): at 64K tokens x 11 candidates the full re-encode
    cost seconds per example."""
    if len(prompt_text) <= JOINT_CHECK_WINDOW_CHARS:
        return prompt_text
    cut = prompt_text.rfind(" ", 0, len(prompt_text) - JOINT_CHECK_WINDOW_CHARS)
    if cut < 0:
        return prompt_text
    return prompt_text[cut + 1:]


def encode_prompt_and_candidates(tokenizer, prompt_text, candidates):
    """Tokenize the prompt and each candidate separately (no special tokens)
    and verify the two properties the scorer depends on:

      1. joint == separate: `encode(prompt + " " + cand)` must equal
         `prompt_ids + cand_ids`, otherwise the candidate's tokens would not be
         the ones the model sees in context (boundary re-segmentation). The
         check runs on a bounded suffix of the prompt (see _boundary_window);
         if that suffix does not itself tokenize as the tail of the full
         prompt (a tokenizer whose segmentation is not whitespace-local), the
         full-prompt check is used instead;
      2. no special token (T5's `</s>`) may appear at or before a scored row —
         that is exactly how the pre-fix scorer ended up scoring `</s>`
         instead of the answer. (score_example may append the PAD token AFTER
         the candidate to equalize input lengths; those positions are never
         scored and never in a scored row's causal past.)

    Raises ValueError (not assert — asserts are stripped under -O) on either.
    Returns (prompt_ids, {cand: cand_ids}).
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    specials = {t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None}
    if any(t in specials for t in prompt_ids):
        raise ValueError("special token found in the prompt ids — encode with add_special_tokens=False")

    window_text = _boundary_window(prompt_text)
    window_ids = tokenizer.encode(window_text, add_special_tokens=False)
    if window_text != prompt_text and (
        len(window_ids) == 0 or prompt_ids[-len(window_ids):] != window_ids
    ):
        # the suffix does not reproduce the tail of the full tokenization —
        # fall back to checking against the whole prompt
        window_text, window_ids = prompt_text, prompt_ids

    cand_ids = {}
    for cand in candidates:
        ids = tokenizer.encode(" " + cand, add_special_tokens=False)
        if len(ids) == 0:
            raise ValueError(f"candidate {cand!r} tokenizes to nothing")
        if any(t in specials for t in ids):
            raise ValueError(f"special token found in candidate ids for {cand!r}")
        joint = tokenizer.encode(window_text + " " + cand, add_special_tokens=False)
        if joint != window_ids + ids:
            raise ValueError(
                f"joint tokenization of prompt + {cand!r} differs from prompt_ids + cand_ids "
                f"at the boundary — the scored tokens would not be the in-context tokens"
            )
        cand_ids[cand] = ids
    return prompt_ids, cand_ids


def _rows_from_chunked_hidden(model, input_ids, rows, chunk_len):
    """Hidden states for the token positions in `rows` (a slice) gathered from
    the chunked forward: each chunk's hidden states are consumed as they are
    produced and only the needed rows are kept, so nothing O(L) survives."""
    start, stop = rows.start, rows.stop
    pieces = []
    for chunk_start, hidden in model.iter_chunked_hidden(input_ids, chunk_len=chunk_len):
        chunk_stop = chunk_start + hidden.shape[1]
        lo, hi = max(start, chunk_start), min(stop, chunk_stop)
        if lo < hi:
            pieces.append(hidden[:, lo - chunk_start:hi - chunk_start])
        if chunk_stop >= stop:
            break   # rows complete; later chunks are not needed
    if not pieces:
        raise RuntimeError(
            f"chunked forward yielded no rows for {rows} (sequence length {input_ids.shape[1]})"
        )
    out = torch.cat(pieces, dim=1)
    if out.shape[1] != stop - start:
        raise RuntimeError(f"chunked forward yielded {out.shape[1]} rows for {rows}, expected {stop - start}")
    return out


def _candidate_rows_log_softmax(model, input_ids, rows, disable_flex_attn, chunk_len=None):
    """Log-softmax over the vocab for the given positions only. The forward
    returns pre-logit hidden states; only the needed rows are projected, so
    the [L, vocab] logits tensor is never materialized (it was 8.4 GB fp32 at
    64K on its own). With `chunk_len` the hidden states come from the chunked
    forward (memory O(chunk_len) instead of O(L))."""
    if chunk_len is None:
        hidden = model.forward(input_ids, disable_flex_attn=disable_flex_attn, return_hidden=True)
        rows_hidden = hidden[:, rows]
    else:
        rows_hidden = _rows_from_chunked_hidden(model=model, input_ids=input_ids, rows=rows, chunk_len=chunk_len)
    to_logits = model.to_logits
    if isinstance(to_logits, torch.nn.Linear):
        # project in fp32 even under --bf16: the candidate log-probs are the
        # decision statistic, and a bf16 projection can flip close argmaxes
        bias = None if to_logits.bias is None else to_logits.bias.float()
        logits = F.linear(rows_hidden.float(), to_logits.weight.float(), bias)
    elif rows_hidden.dtype == torch.float32:
        logits = to_logits(rows_hidden)
    else:
        # never fall back silently to a reduced-precision projection
        raise TypeError(
            f"to_logits is a {type(to_logits).__name__}, not nn.Linear, and the hidden states are "
            f"{rows_hidden.dtype}: the scorer cannot project in fp32. Run in float32 or give the "
            f"scorer an nn.Linear output projection."
        )
    return F.log_softmax(logits[0].float(), dim=-1)


@torch.no_grad()
def score_example(model, tokenizer, prompt_text, candidates, device="cuda", disable_flex_attn=True, chunk_len=None):
    """Pick the candidate with the highest log-likelihood under the model.

    Closed-answer-set protocol for a pretrained-only (non-instruction-tuned)
    base LM: score log P(candidate tokens | prompt) for every candidate and
    pick the argmax of the length-normalized log-probability.

    Tokenization: the prompt and the candidate are tokenized SEPARATELY with
    `add_special_tokens=False` and concatenated (candidate encoded as
    " <cand>" so it carries its leading-space piece). This is the standard
    guard against the joint-tokenization trap — tokenizing prompt+candidate
    together lets the boundary re-segment differently per candidate, and a
    tokenizer that appends specials (T5 appends `</s>`) shifts the scored
    positions. `encode_prompt_and_candidates` verifies joint == separate and
    that no special token appears at or before a scored row; exactly the
    candidate's token positions are scored.

    Forward passes: when every candidate is a single token, ONE forward over
    the prompt scores all of them from the last row. Multi-token candidate
    sets fall back to one forward per candidate over prompt + candidate, with
    every input PADDED to the same total length (pad token appended AFTER the
    candidate — never inside a scored row's causal past). Only the candidate
    rows are projected to logits (return_hidden).

    Why one forward is valid — and why "the last row depends only on the
    prompt" is NOT the reason: this model is not length-invariant. With the
    omega rule + per-token retrieve, positions inside an incomplete final
    store chunk read the state after the last COMPLETE chunk (the remainder
    is cached, never stored in a whole-sequence forward), so the logits at a
    fixed position change when an appended token completes the chunk
    (measured 0.8-2.1 nats at random init, only at length % chunk == chunk-1;
    zero for the memory-free trunk — see
    test_per_token_retrieve_tail_reads_last_complete_chunk_state). Scoring
    all candidates from ONE forward keeps them consistent with each other.
    In the multi-token branch, candidates of different token counts would get
    forwards of different lengths and hence different chunk phases — the
    shared prompt rows then read different memory states (measured 1.03 nats
    on the first candidate token at 2/16 prompt lengths, test
    test_mixed_length_candidates_padded_to_common_length), so every input is
    padded to `prompt_len + max candidate length`: the padding sits after the
    scored rows, which never see it (causal attention; a position's memory
    state contains only tokens at or before it), while the chunk-completion
    phase becomes identical for all candidates. Roughly 1/8 of prompts fall
    in the fresh-state regime under either branch — the same tail quirk
    training saw on the last 4 of every 1084 positions.

    Normalization: mean log-prob per candidate token. Residual asymmetry:
    where a candidate set mixes token counts (qa7's " ten" is 2 T5 tokens
    while the other numerals are 1), the per-token mean is not perfectly
    comparable; the raw sums and token counts are persisted in `details` so
    post-hoc analyses can re-normalize. qa5's names are single tokens only in
    their capitalized in-context form — see prompts.TASK_LABELS.

    Known, unquantified conditioning shift: training documents end with
    `</s>`; scoring inputs contain none. Shared by every candidate, so it
    cannot create a per-candidate asymmetry.

    History (2026-09-02): the previous implementation computed the prompt
    length from `tokenizer.encode(prompt_text)`, which for T5 includes the
    appended `</s>`, so `range(prompt_len, full_len)` skipped the first
    candidate token and scored the trailing `</s>`. Every qa1-qa4 answer is a
    single T5 token, so the scored set was `['</s>']` for every candidate:
    every BABILong number produced before this fix measured
    P(end-of-text | prompt + word) — an end-of-text prior with no dependence
    on the context — and sat at chance by construction.

    Returns: (chosen_candidate, per_candidate_norm_log_probs, prompt_num_tokens,
              per_candidate_details) where details[cand] = {"sum": raw summed
              log-prob, "num_tokens": candidate token count}.
    """
    prompt_ids, cand_ids = encode_prompt_and_candidates(
        tokenizer=tokenizer, prompt_text=prompt_text, candidates=candidates
    )
    prompt_len = len(prompt_ids)

    log_probs = {}
    details = {}

    if all(len(ids) == 1 for ids in cand_ids.values()):
        # one forward over the prompt; the last row predicts the first
        # (and only) candidate token for every candidate
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        rows = slice(prompt_len - 1, prompt_len)
        log_softmax = _candidate_rows_log_softmax(
            model=model, input_ids=input_ids, rows=rows, disable_flex_attn=disable_flex_attn,
            chunk_len=chunk_len,
        )
        for cand, ids in cand_ids.items():
            lp = log_softmax[0, ids[0]].item()
            log_probs[cand] = lp
            details[cand] = {"sum": lp, "num_tokens": 1}
    else:
        lengths = {len(ids) for ids in cand_ids.values()}
        common_len = prompt_len + max(lengths)
        pad_id = tokenizer.pad_token_id
        if len(lengths) > 1 and pad_id is None:
            # padding is only needed when the candidates differ in length
            raise ValueError(
                "mixed-length candidate set needs a pad token to equalize the input "
                "lengths (chunk-phase consistency, see score_example) — the tokenizer has none"
            )
        for cand, ids in cand_ids.items():
            # pad AFTER the candidate so every candidate's forward has the
            # same length (same chunk phase); the scored rows never see it
            full_ids = prompt_ids + ids + [pad_id] * (common_len - prompt_len - len(ids))
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            # logits[:, i] predicts the token at position i+1: candidate token
            # at position prompt_len + j is scored by row prompt_len - 1 + j
            rows = slice(prompt_len - 1, prompt_len - 1 + len(ids))
            log_softmax = _candidate_rows_log_softmax(
                model=model, input_ids=input_ids, rows=rows, disable_flex_attn=disable_flex_attn,
                chunk_len=chunk_len,
            )
            total_lp = sum(log_softmax[j, target].item() for j, target in enumerate(ids))
            log_probs[cand] = total_lp / len(ids)
            details[cand] = {"sum": total_lp, "num_tokens": len(ids)}

    chosen = max(log_probs, key=log_probs.get)
    return chosen, log_probs, prompt_len, details


def _target_matches(target, chosen):
    """qa8 target can be a comma-separated list (e.g. 'apple,milk'). Everything
    else is a single label. Compare case-insensitively."""
    target_parts = {t.strip().lower() for t in target.split(",")}
    chosen_parts = {c.strip().lower() for c in chosen.split(",")}
    return target_parts == chosen_parts


def target_in_candidates(target, candidates):
    """Whether a target is reachable by argmax over the candidate set (case-
    insensitive). qa8's two-object targets ('apple,football', ~6% of examples)
    are not — no single label can match them — so they are excluded and
    counted rather than scored as guaranteed failures."""
    lowered = {c.lower() for c in candidates}
    return target.strip().lower() in lowered


def actual_length_ceiling(model, tokenizer, task, length, dataset, candidates, force, device_total_bytes,
                          chunk_len=None):
    """The memory-ceiling check against the ACTUAL tokenized prompt of the
    first scorable example (plus the longest candidate), not the nominal
    split label: '4k' prose tokenizes to more than 4096 T5 tokens and the
    few-shot scaffold adds more — a label that passes can still OOM. Returns
    (message_or_None, num_tokens_or_None)."""
    for example in dataset:
        if not target_in_candidates(target=example["target"], candidates=candidates):
            continue
        prompt_text = build_prompt(task=task, context=example["input"], question=example["question"])
        prompt_ids, cand_ids = encode_prompt_and_candidates(
            tokenizer=tokenizer, prompt_text=prompt_text, candidates=candidates
        )
        num_tokens = len(prompt_ids) + max(len(ids) for ids in cand_ids.values())
        message = memory_ceiling_for_tokens(
            model=model,
            num_tokens=num_tokens,
            label=f"{task}@{length} (actual first prompt)",
            force=force,
            device_total_bytes=device_total_bytes,
            chunk_len=chunk_len,
        )
        return message, num_tokens
    return None, None


def evaluate_task(model, tokenizer, task, length, max_examples=None, device="cuda", disable_flex_attn=True,
                  force=False, device_total_bytes=None, dataset=None, chunk_len=None):
    """Evaluate model on a single task at a single context length via
    log-likelihood scoring over the closed candidate set (TASK_LABELS[task]).

    Before the example loop the memory ceiling is re-checked against the
    ACTUAL tokenized length of the first scorable prompt (see
    actual_length_ceiling); when it does not fit and `force` is off the task
    is skipped and the returned dict carries `skipped` with the reason.
    `dataset` (an iterable of {'input','question','target'} dicts) and
    `device_total_bytes` are injection points for tests; by default the
    BABILong split is loaded from the hub and the device's total memory is
    used."""
    if dataset is None:
        from datasets import load_dataset

        dataset = load_dataset("RMT-team/babilong", length, split=task)

        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
    elif max_examples:
        dataset = list(dataset)[:max_examples]

    candidates = sorted(TASK_LABELS[task])

    ceiling, first_num_tokens = actual_length_ceiling(
        model=model, tokenizer=tokenizer, task=task, length=length, dataset=dataset,
        candidates=candidates, force=force, device_total_bytes=device_total_bytes,
        chunk_len=chunk_len,
    )
    if ceiling is not None:
        print(f"  {ceiling}")
        if not force:
            return {
                "accuracy": 0.0, "accuracy_all": 0.0, "correct": 0, "total": 0,
                "n_excluded": 0, "results": [], "skipped": ceiling,
                "first_prompt_tokens": first_num_tokens,
            }

    correct = 0
    total = 0
    n_excluded = 0
    results = []

    for example in tqdm(dataset, desc=f"{task}@{length}", leave=False):
        context = example["input"]
        question = example["question"]
        target = example["target"]

        if not target_in_candidates(target=target, candidates=candidates):
            n_excluded += 1
            continue

        prompt_text = build_prompt(task=task, context=context, question=question)

        t0 = time.time()
        chosen, log_probs, num_tokens, details = score_example(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            candidates=candidates,
            device=device,
            disable_flex_attn=disable_flex_attn,
            chunk_len=chunk_len,
        )
        elapsed = time.time() - t0

        is_correct = _target_matches(target=target, chosen=chosen)
        correct += int(is_correct)
        total += 1

        results.append({
            "question": question.strip(),
            "target": target,
            "chosen": chosen,
            "correct": is_correct,
            "log_probs": {k: round(v, 3) for k, v in log_probs.items()},
            "log_prob_sums": {k: round(d["sum"], 3) for k, d in details.items()},
            "cand_num_tokens": {k: d["num_tokens"] for k, d in details.items()},
            "num_tokens": num_tokens,
            "time_s": round(elapsed, 2),
        })

    if n_excluded:
        print(f"  {task}@{length}: {n_excluded} example(s) excluded — target not in the candidate set")

    accuracy = correct / total if total > 0 else 0.0
    # official-protocol denominator: excluded examples (target unreachable by
    # any candidate) count as failures
    accuracy_all = correct / (total + n_excluded) if (total + n_excluded) > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "accuracy_all": round(accuracy_all, 4),
        "correct": correct,
        "total": total,
        "n_excluded": n_excluded,
        "results": results,
        "first_prompt_tokens": first_num_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"]
ALL_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k"]


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="BABILong evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to accelerate checkpoint dir")
    p.add_argument("--model", required=True, choices=["170m", "340m", "760m", "1.3b"])
    p.add_argument("--variant", required=True,
                    choices=["titans-mac", "titans-mag", "atlas-mac", "atlas-mag"])
    p.add_argument("--ablation", default=None)
    p.add_argument("--vanilla", action="store_true",
                   help="Memory-free baseline: build with neural_memory_layers=() to match train.py --vanilla.")
    p.add_argument("--tokenizer-dir", default=None,
                    help="Path to tokenizer (default: checkpoint/../tokenizer or google-t5/t5-base)")
    p.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa3", "qa4", "qa5"], choices=ALL_TASKS,
                    help="Tasks to evaluate (default: qa1-qa5)")
    p.add_argument("--lengths", nargs="+", default=["0k", "4k", "16k", "64k"], choices=ALL_LENGTHS,
                    help="Context lengths to evaluate")
    p.add_argument("--max-examples", type=int, default=None,
                    help="Max examples per task (for quick testing)")
    p.add_argument("--output", default=None, help="Output JSON path (written after every task)")
    p.add_argument("--device", default=default_device())
    p.add_argument("--bf16", action="store_true",
                   help="Run the model in bfloat16 (training dtype); halves the per-token retrieve-state footprint")
    p.add_argument("--use-flex-attn", action="store_true",
                   help="Use flex attention (CUDA only); default disables it")
    p.add_argument("--force", action="store_true",
                   help="Run a context length even when the estimated PEAK live storage "
                        "(PEAK_FACTOR x retrieve state + the attention path) "
                        "exceeds 80%% of device memory")
    p.add_argument("--chunk-len", type=int, default=None,
                   help="Chunked inference: process the sequence in chunks of this many INTERLEAVED "
                        "positions (tokens + longterm-mem tokens), carrying only the memory state and the "
                        "attention window across chunks — memory O(chunk) instead of O(L). Must be a "
                        "multiple of neural_memory_batch_size (1024 in the shipped config; use 1024). "
                        "Output equals the whole-sequence forward.")
    return p.parse_args()


def write_results(all_results, output_path):
    """Atomic write: dump to a temp file, then os.replace — a kill mid-dump
    leaves the previous complete file in place."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(all_results, f, indent=2)
    os.replace(tmp_path, output_path)


def main():
    args = parse_args()

    # Load model
    model, config = load_model(
        checkpoint_dir=args.checkpoint,
        model_size=args.model,
        variant=args.variant,
        ablation=args.ablation,
        device=args.device,
        vanilla=args.vanilla,
        dtype=torch.bfloat16 if args.bf16 else None,
    )

    if args.chunk_len is not None:
        # a misaligned chunk length fails here, before any data is loaded,
        # not inside the first scored example
        model.chunked_inference_alignment(args.chunk_len)

    # Load tokenizer
    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        # Try to find tokenizer relative to checkpoint
        for candidate in [
            os.path.join(os.path.dirname(args.checkpoint), "tokenizer"),
            os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "tokenizer"),
        ]:
            if os.path.exists(candidate):
                tokenizer_dir = candidate
                break
    tokenizer = load_tokenizer(tokenizer_dir or "google-t5/t5-base")
    # T5 reports 32100 (no sentinel ids); the model's embedding has 32128 rows
    # (padded to a multiple of 128) — both are correct, they are different
    # quantities.
    print(f"Tokenizer: vocab_size={tokenizer.vocab_size} | model num_tokens={config['model']['num_tokens']}")

    output_path = args.output or f"results/{args.model}-{args.variant}-babilong.json"

    # Run evaluation — results are written after every (length, task) so a
    # mid-run failure at a long context keeps everything finished before it
    all_results = {
        "model": args.model,
        "variant": args.variant,
        "ablation": args.ablation,
        "checkpoint": args.checkpoint,
        "dtype": "bfloat16" if args.bf16 else "float32",
        "chunk_len": args.chunk_len,
        "tasks": {},
        "skipped_lengths": {},
    }

    for length in args.lengths:
        print(f"\n=== Context length: {length} ===")
        ceiling = memory_ceiling_message(
            model=model, length=length, device=args.device, force=args.force, chunk_len=args.chunk_len
        )
        if ceiling is not None:
            print(f"  {ceiling}")
            if not args.force:
                all_results["skipped_lengths"][length] = ceiling
                write_results(all_results=all_results, output_path=output_path)
                continue

        all_results["tasks"][length] = {}

        for task in args.tasks:
            result = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                length=length,
                max_examples=args.max_examples,
                device=args.device,
                disable_flex_attn=not args.use_flex_attn,
                force=args.force,
                device_total_bytes=device_total_memory_bytes(args.device),
                chunk_len=args.chunk_len,
            )

            if result.get("skipped"):
                # the ACTUAL tokenized prompt exceeded the ceiling the nominal
                # label passed; recorded, not scored
                all_results["tasks"][length][task] = {
                    "skipped": result["skipped"],
                    "first_prompt_tokens": result["first_prompt_tokens"],
                }
                write_results(all_results=all_results, output_path=output_path)
                continue

            all_results["tasks"][length][task] = {
                "accuracy": result["accuracy"],
                "accuracy_all": result["accuracy_all"],
                "correct": result["correct"],
                "total": result["total"],
                "n_excluded": result["n_excluded"],
                "examples": result["results"],
            }
            write_results(all_results=all_results, output_path=output_path)

            line = f"  {task}: {result['accuracy']:.1%} ({result['correct']}/{result['total']} scored)"
            if result["n_excluded"]:
                line += (f"; {result['accuracy_all']:.1%} over all "
                         f"{result['total'] + result['n_excluded']} ({result['n_excluded']} excluded)")
            print(line)

    # Summary table
    print("\n=== Summary ===")
    header = f"{'Task':<6}" + "".join(f"{l:>8}" for l in args.lengths)
    print(header)
    print("-" * len(header))
    for task in args.tasks:
        row = f"{task:<6}"
        for length in args.lengths:
            acc = all_results["tasks"].get(length, {}).get(task, {}).get("accuracy", None)
            row += f"{acc:>7.1%} " if acc is not None else f"{'—':>8}"
        print(row)

    write_results(all_results=all_results, output_path=output_path)
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
