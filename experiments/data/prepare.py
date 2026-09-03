"""
Pre-tokenize FineWeb for offline training.

Tokenizes with T5, saves as memory-mapped binary files (uint16).
Can run locally (downloads from HuggingFace) or on BSC (from local parquet).

Tokenizer property worth knowing (same tokenizer as the Atlas paper, Appendix E):
T5's SentencePiece model has NO byte fallback — characters outside its vocab
(`{ } < \\`, CJK, emoji, ...) become `<unk>` (id 2) — and its normalizer maps
newlines and tabs to plain whitespace (`encode("a\\nb") == encode("a b")`), so
paragraph/line structure is gone after tokenization. Measured on FineWeb
sample-10BT (200 docs / 141K tokens, 2026-09-02): `<unk>` rate 0.10%, 18% of
documents carry at least one `<unk>`, and about one newline per 45 tokens is
normalized away. The audit counters below record the cost on the corpus
actually tokenized (meta.json: unk_tokens / unk_rate / docs_with_unk /
newlines_normalized).

Usage:
    # Local (downloads from HF):
    python experiments/data/prepare.py --output /tmp/fineweb-t5

    # On BSC (from pre-downloaded parquet files):
    python experiments/data/prepare.py --output /gpfs/.../fineweb-t5 \
        --data-dir /gpfs/.../fineweb-parquet \
        --tokenizer-dir /gpfs/.../t5-tokenizer

    # Smaller subset for testing:
    python experiments/data/prepare.py --output /tmp/fineweb-t5 --max-tokens 100_000_000

Step 1 — Download raw data locally (needs internet):
    huggingface-cli download HuggingFaceFW/fineweb sample/10BT --repo-type dataset \
        --local-dir /tmp/fineweb-parquet

Step 2 — Download tokenizer locally:
    python -c "from transformers import AutoTokenizer; \\
        t = AutoTokenizer.from_pretrained('google-t5/t5-base'); \\
        t.save_pretrained('/tmp/t5-tokenizer')"

Step 3 — Transfer to BSC:
    rsync -avz /tmp/fineweb-parquet/ transfer1.bsc.es:/gpfs/projects/YOUR_BSC_ACCOUNT/data/fineweb-parquet/
    rsync -avz /tmp/t5-tokenizer/ transfer1.bsc.es:/gpfs/projects/YOUR_BSC_ACCOUNT/data/t5-tokenizer/

Step 4 — Tokenize on BSC (copy the example launcher, fill in your account, submit):
    cp experiments/slurm/prepare_data.example.sh experiments/slurm/prepare_data.sh
    sbatch experiments/slurm/prepare_data.sh
"""

import argparse
import json
import os

import numpy as np
from tqdm import tqdm


def load_dataset_stream(data_dir=None):
    from datasets import load_dataset

    if data_dir:
        return load_dataset(data_dir, split="train", streaming=True)
    return load_dataset(
        "HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True
    )


def load_tokenizer(tokenizer_dir=None):
    from transformers import AutoTokenizer

    name_or_path = tokenizer_dir or "google-t5/t5-base"
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    # Save tokenizer alongside output for portability
    return tokenizer


def write_train_val_split(shard_paths, train_tokens, train_path, val_path):
    """Concatenate shards into train.bin / val.bin, splitting at train_tokens.

    The boundary may fall anywhere, including across multiple shards: per
    shard, the first `clamp(train_tokens - train_written, 0, len(shard))`
    tokens go to train and the remainder to val. (The previous inline loop
    sliced `shard[split:]` with a NEGATIVE split once the boundary had been
    passed, silently dropping all but the last tokens of any later shard —
    token loss whenever the val overhang spanned 2+ shards; found
    2026-09-01.) Shards are deleted as they are consumed.

    Returns (train_written, val_written).
    """
    train_written = 0
    val_written = 0
    with open(train_path, "wb") as ft, open(val_path, "wb") as fv:
        for sp in shard_paths:
            shard = np.fromfile(sp, dtype=np.uint16)
            split = min(max(train_tokens - train_written, 0), len(shard))
            if split > 0:
                shard[:split].tofile(ft)
                train_written += split
            if split < len(shard):
                shard[split:].tofile(fv)
                val_written += len(shard) - split
            os.remove(sp)
    return train_written, val_written


def main():
    p = argparse.ArgumentParser(description="Pre-tokenize FineWeb for training")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--data-dir", default=None, help="Local path to FineWeb parquet (offline)")
    p.add_argument("--tokenizer-dir", default=None, help="Local path to T5 tokenizer (offline)")
    p.add_argument("--max-tokens", type=int, default=None, help="Stop after N tokens")
    p.add_argument("--val-tokens", type=int, default=10_000_000, help="Held-out validation tokens")
    p.add_argument("--shard-size", type=int, default=100_000_000, help="Tokens per temp shard")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_dir)
    tokenizer_out = os.path.join(args.output, "tokenizer")
    tokenizer.save_pretrained(tokenizer_out)
    print(f"  vocab_size={tokenizer.vocab_size}, saved → {tokenizer_out}")
    assert tokenizer.vocab_size <= 65535, "Vocab too large for uint16"

    # Stream dataset
    print(f"Loading dataset{' from ' + args.data_dir if args.data_dir else ' (streaming from HF)'}...")
    dataset = load_dataset_stream(args.data_dir)

    # Tokenize → shards
    total_tokens = 0
    shard_idx = 0
    buffer = []
    shard_paths = []

    # inherited-tokenizer audit counters (see module docstring). Counted over
    # every document tokenized, including the final buffer whether or not it
    # is flushed, so they describe the tokenizer's behaviour on the corpus.
    unk_id = tokenizer.unk_token_id
    n_docs = 0
    unk_tokens = 0
    docs_with_unk = 0
    newlines_normalized = 0

    for example in tqdm(dataset, desc="Tokenizing", unit=" docs"):
        tokens = tokenizer.encode(example["text"])
        buffer.extend(tokens)

        n_docs += 1
        n_unk = tokens.count(unk_id)
        unk_tokens += n_unk
        docs_with_unk += int(n_unk > 0)
        newlines_normalized += example["text"].count("\n")

        while len(buffer) >= args.shard_size:
            chunk = buffer[: args.shard_size]
            buffer = buffer[args.shard_size :]

            path = os.path.join(args.output, f"shard_{shard_idx:04d}.bin")
            np.array(chunk, dtype=np.uint16).tofile(path)
            shard_paths.append(path)

            total_tokens += len(chunk)
            shard_idx += 1
            tqdm.write(f"  shard {shard_idx}: {total_tokens / 1e9:.3f}B tokens")

            if args.max_tokens and total_tokens >= args.max_tokens:
                break

        if args.max_tokens and total_tokens >= args.max_tokens:
            break

    # Flush remainder
    if buffer and (not args.max_tokens or total_tokens < args.max_tokens):
        path = os.path.join(args.output, f"shard_{shard_idx:04d}.bin")
        np.array(buffer, dtype=np.uint16).tofile(path)
        shard_paths.append(path)
        total_tokens += len(buffer)

    # Concatenate into train.bin + val.bin
    val_tokens = min(args.val_tokens, total_tokens // 100)
    train_tokens = total_tokens - val_tokens

    print(f"\nTotal: {total_tokens:,} tokens ({total_tokens / 1e9:.2f}B)")
    print(f"Splitting: train={train_tokens:,}  val={val_tokens:,}")

    train_path = os.path.join(args.output, "train.bin")
    val_path = os.path.join(args.output, "val.bin")

    train_written, val_written = write_train_val_split(
        shard_paths=shard_paths,
        train_tokens=train_tokens,
        train_path=train_path,
        val_path=val_path,
    )

    # every token lands in exactly one of the two files (uint16 = 2 bytes)
    assert train_written == train_tokens, (
        f"train split short: wrote {train_written:,}, wanted {train_tokens:,}"
    )
    assert os.path.getsize(train_path) + os.path.getsize(val_path) == total_tokens * 2, (
        f"token loss in split: train.bin {os.path.getsize(train_path)}B + "
        f"val.bin {os.path.getsize(val_path)}B != {total_tokens:,} tokens * 2B"
    )

    # a 0-byte val.bin would crash np.memmap at train startup; remove it and
    # let train.py print "validation disabled" instead
    if val_written == 0:
        os.remove(val_path)
        print("val split empty — removed val.bin (validation will be disabled)")

    tokens_seen = total_tokens + (len(buffer) if args.max_tokens and total_tokens >= args.max_tokens else 0)
    unk_rate = unk_tokens / max(1, tokens_seen)

    meta = dict(
        vocab_size=tokenizer.vocab_size,
        total_tokens=total_tokens,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        dtype="uint16",
        # tokenizer audit (T5: no byte fallback; newlines/tabs normalized to whitespace)
        docs=n_docs,
        unk_tokens=unk_tokens,
        unk_rate=unk_rate,
        docs_with_unk=docs_with_unk,
        newlines_normalized=newlines_normalized,
    )
    with open(os.path.join(args.output, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone → {args.output}/")
    print(f"  train.bin  {os.path.getsize(train_path) / 1e9:.2f} GB")
    if os.path.exists(val_path):
        print(f"  val.bin    {os.path.getsize(val_path) / 1e6:.1f} MB")
    print(f"  tokenizer/")
    print(f"  meta.json")
    print(
        f"\nTokenizer audit (T5, no byte fallback): {n_docs:,} docs, "
        f"{unk_tokens:,} <unk> tokens ({unk_rate * 100:.3f}%), "
        f"{docs_with_unk:,} docs with <unk>, "
        f"{newlines_normalized:,} newlines in source text normalized to whitespace"
    )


if __name__ == "__main__":
    main()
