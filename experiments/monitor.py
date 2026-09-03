"""Training-run monitor: parse a train.py log and judge whether the run needs
attention. Pure stdlib (no torch), so it runs on the cluster login node
against a live log; train.py imports `format_train_line` / `format_val_line`
from here, so the producer and the parser cannot drift apart.

    python experiments/monitor.py runs/170m-atlas-mac_123.log --warmup-steps 2000 --json

Statuses, in priority order:
  failed  - a traceback, CUDA / NCCL error, segfault or SLURM step error
            (the numeric findings are still reported alongside)
  alarm   - non-finite loss or grad norm; loss above ln(vocab) after
            sanity_steps; the trailing loss window above the best earlier
            block by loss_spike_ratio (after warmup); grad-norm window
            maximum above grad_norm_spike x the rolling median for
            grad_norm_consecutive consecutive log lines; interval throughput
            below throughput_alarm x the run's median; validation loss above
            its best by val_rise_ratio after val_rises consecutive rises; a
            log that has not grown for stale_minutes while the job runs
  timeout - the SLURM wall-limit kill (the normal end of a chained job)
  done    - the final "Done." line
  watch   - throughput below throughput_watch x median
  ok      - nothing to report; `summary` carries the numbers either way

The monitor never changes a run. Exit code: 0 ok/watch/done/timeout,
2 alarm/failed, 3 the monitor itself could not judge (escalate).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

# ---- the log format: one producer, one parser ------------------------------

def format_train_line(step: int, loss: float, lr: float, tok_per_sec: float, tokens_total: float, grad_norm_max: float) -> str:
    """The training progress line. `tok_per_sec` is the INTERVAL rate since the
    previous log line (a cumulative average hides a mid-run slowdown);
    `grad_norm_max` is the largest pre-clip norm over the interval (the last
    step alone misses 9 of 10 spikes at log_every=10)."""
    return (
        f"step {step:>7d} | loss {loss:.4f} | ppl {math.exp(min(loss, 20)):.1f} | lr {lr:.2e} | "
        f"{tok_per_sec / 1e3:.1f}k tok/s | {tokens_total / 1e9:.3f}B | gnorm {grad_norm_max:.3f}"
    )


def format_val_line(step: int, val_loss: float, frac_near_certain: float) -> str:
    return (
        f"step {step:>7d} | val_loss {val_loss:.4f} | val_ppl {math.exp(min(val_loss, 20)):.1f} | "
        f"val_frac_near_certain {frac_near_certain:.4f}"
    )


_NUM = r"[-+\d.einfa]+"
TRAIN_RE = re.compile(
    rf"^step\s+(?P<step>\d+) \| loss (?P<loss>{_NUM}) \| ppl {_NUM} \| lr (?P<lr>{_NUM}) \| "
    rf"(?P<tok>{_NUM})k tok/s \| (?P<tokens>{_NUM})B(?: \| gnorm (?P<gnorm>{_NUM}))?\s*$"
)
VAL_RE = re.compile(rf"^step\s+(?P<step>\d+) \| val_loss (?P<val>{_NUM}) \|(?:.*val_frac_near_certain (?P<frac>{_NUM}))?")
FAILURE_MARKERS = (
    "Traceback (most recent call last)", "CUDA out of memory", "CUDA error", "NCCL error", "NCCL WARN",
    "Segmentation fault", "slurmstepd: error",
)
TIMEOUT_MARKER = "DUE TO TIME LIMIT"
DONE_MARKER = "Done."
LN_VOCAB_T5 = math.log(32100)   # uniform prediction over the T5 vocab; anything above it after sanity_steps is not learning


@dataclass
class Rules:
    warmup_steps: int = 2000
    window: int = 20                  # log lines per window (x log_every steps)
    loss_spike_ratio: float = 0.10    # trailing window mean > (1 + ratio) x best earlier block mean
    sanity_steps: int = 200           # loss must be below ln(vocab) after this many steps
    grad_norm_spike: float = 5.0      # x rolling median of the window maxima
    grad_norm_consecutive: int = 2
    throughput_watch: float = 0.8     # x median interval tok/s
    throughput_alarm: float = 0.5
    val_rises: int = 2                # consecutive validations with rising loss ...
    val_rise_ratio: float = 0.01      # ... AND the latest above the best validation by this ratio
    stale_minutes: float = 20.0


@dataclass
class Verdict:
    status: str
    reasons: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _num(text: Optional[str]) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def parse_lines(lines: Iterable[str]) -> dict:
    """-> dict(train=[{step, loss, lr, tok, tokens, gnorm}], val=[{step, val, frac}], failures=[...], timeout, done)."""
    train, val, failures, timeout, done = [], [], [], False, False
    for line in lines:
        m = TRAIN_RE.match(line)
        if m:
            g = m.groupdict()
            train.append(dict(step=int(g["step"]), loss=_num(g["loss"]), lr=_num(g["lr"]), tok=_num(g["tok"]) * 1e3,
                              tokens=_num(g["tokens"]) * 1e9, gnorm=_num(g["gnorm"]) if g["gnorm"] is not None else None))
            continue
        m = VAL_RE.match(line)
        if m:
            val.append(dict(step=int(m["step"]), val=_num(m["val"]), frac=_num(m["frac"])))
            continue
        if TIMEOUT_MARKER in line:
            timeout = True
        elif any(marker in line for marker in FAILURE_MARKERS):
            failures.append(line.strip()[:200])
        if line.startswith(DONE_MARKER):
            done = True
    return dict(train=train, val=val, failures=failures, timeout=timeout, done=done)


def parse_log(text: str) -> dict:
    return parse_lines(lines=text.splitlines())


def _finite(values):
    return [v for v in values if v is not None and math.isfinite(v)]


def assess(parsed: dict, rules: Optional[Rules] = None, log_age_minutes: Optional[float] = None, job_running: Optional[bool] = None) -> Verdict:
    """Apply every rule and rank the result. `log_age_minutes` (time since the
    log last grew) with `job_running=True` feeds the stall rule."""
    rules = rules or Rules()
    train, val = parsed["train"], parsed["val"]
    alarms, watch = [], []
    summary: dict = dict(steps=len(train), last_step=train[-1]["step"] if train else 0)

    if train:
        last = train[-1]
        summary.update(last_loss=last["loss"], last_lr=last["lr"], last_gnorm=last["gnorm"], tokens_seen=last["tokens"])

        # non-finite, and the absolute sanity bound (worse than uniform after sanity_steps)
        if any(not math.isfinite(t["loss"]) for t in train[-3:]):
            alarms.append(f"non-finite loss at step {last['step']}")
        recent_gnorms = [t["gnorm"] for t in train[-3:] if t["gnorm"] is not None]
        if any(not math.isfinite(g) for g in recent_gnorms):
            alarms.append(f"non-finite grad norm at step {last['step']}")
        if last["step"] >= rules.sanity_steps and math.isfinite(last["loss"]) and last["loss"] > LN_VOCAB_T5:
            alarms.append(f"loss {last['loss']:.3f} above ln(vocab) {LN_VOCAB_T5:.3f} at step {last['step']} (not learning)")

        # loss: TRAILING window vs the best earlier non-overlapping block, after warmup
        post = [t["loss"] for t in train if t["step"] > rules.warmup_steps and math.isfinite(t["loss"])]
        if len(post) >= 2 * rules.window:
            trailing = statistics.fmean(post[-rules.window:])
            earlier = post[:-rules.window]
            blocks = [earlier[i:i + rules.window] for i in range(0, len(earlier) - rules.window + 1, rules.window)] or [earlier]
            best_earlier = min(statistics.fmean(b) for b in blocks)
            summary.update(trailing_window_mean=trailing, best_earlier_block_mean=best_earlier)
            if trailing > (1 + rules.loss_spike_ratio) * best_earlier:
                alarms.append(f"trailing loss window {trailing:.4f} is {trailing / best_earlier - 1:+.1%} vs the best earlier block {best_earlier:.4f}")

        # grad norm: window maxima vs their rolling median
        gnorms = _finite(t["gnorm"] for t in train)
        if len(gnorms) >= 2 * rules.grad_norm_consecutive + 4:
            recent = gnorms[-rules.grad_norm_consecutive:]
            median = statistics.median(gnorms[:-rules.grad_norm_consecutive])
            summary["gnorm_median"] = median
            if median > 0 and all(g > rules.grad_norm_spike * median for g in recent):
                alarms.append(f"grad norm {recent} > {rules.grad_norm_spike}x median {median:.3f} for {rules.grad_norm_consecutive} consecutive logs")

        # throughput: interval rate vs the run's median (skip the first line: cold start)
        toks = [t["tok"] for t in train[1:] if math.isfinite(t["tok"]) and t["tok"] > 0]
        if len(toks) >= 5:
            median_tok = statistics.median(toks)
            summary.update(tok_per_s=last["tok"], tok_per_s_median=median_tok)
            if last["tok"] < rules.throughput_alarm * median_tok:
                alarms.append(f"throughput {last['tok'] / 1e3:.1f}k tok/s < {rules.throughput_alarm}x median {median_tok / 1e3:.1f}k")
            elif last["tok"] < rules.throughput_watch * median_tok:
                watch.append(f"throughput {last['tok'] / 1e3:.1f}k tok/s < {rules.throughput_watch}x median {median_tok / 1e3:.1f}k")

    # validation: consecutive rises AND a real magnitude above the best
    vals = [v["val"] for v in val if math.isfinite(v["val"])]
    if vals:
        summary["val"] = [(v["step"], round(v["val"], 4)) for v in val[-4:]]
        rises = 0
        for prev, cur in zip(vals[:-1], vals[1:]):
            rises = rises + 1 if cur > prev else 0
        if rises >= rules.val_rises and vals[-1] > min(vals) * (1 + rules.val_rise_ratio):
            alarms.append(f"validation loss {vals[-1]:.4f} rose {rises} times in a row and is {vals[-1] / min(vals) - 1:+.1%} above its best {min(vals):.4f}")

    # stall (the caller measures the log age and the job state)
    if job_running and log_age_minutes is not None and log_age_minutes > rules.stale_minutes:
        alarms.append(f"log has not grown for {log_age_minutes:.0f} min while the job is running")

    if parsed["failures"]:
        return Verdict(status="failed", reasons=parsed["failures"][:5] + alarms, summary=summary)
    if alarms:
        return Verdict(status="alarm", reasons=alarms, summary=summary)
    if parsed["timeout"]:
        return Verdict(status="timeout", reasons=["job killed at the wall limit (normal end of a chained job)"], summary=summary)
    if parsed["done"]:
        return Verdict(status="done", reasons=["training finished"], summary=summary)
    return Verdict(status="watch" if watch else "ok", reasons=watch, summary=summary)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


EXIT_CODES = dict(ok=0, watch=0, done=0, timeout=0, alarm=2, failed=2)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--warmup-steps", type=int, default=Rules.warmup_steps)
    p.add_argument("--window", type=int, default=Rules.window)
    p.add_argument("--loss-spike-ratio", type=float, default=Rules.loss_spike_ratio)
    p.add_argument("--grad-norm-spike", type=float, default=Rules.grad_norm_spike)
    p.add_argument("--throughput-alarm", type=float, default=Rules.throughput_alarm)
    p.add_argument("--val-rise-ratio", type=float, default=Rules.val_rise_ratio)
    p.add_argument("--stale-minutes", type=float, default=Rules.stale_minutes)
    p.add_argument("--log-age-minutes", type=float, default=None, help="minutes since the log last grew (the caller measures it)")
    p.add_argument("--job-running", action="store_true", help="the SLURM job is RUNNING (enables the stall rule)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    try:
        if args.window < 1:
            raise ValueError("--window must be >= 1")
        with open(args.log, encoding="utf-8", errors="replace") as f:
            parsed = parse_lines(lines=f)
        rules = Rules(warmup_steps=args.warmup_steps, window=args.window, loss_spike_ratio=args.loss_spike_ratio,
                      grad_norm_spike=args.grad_norm_spike, throughput_alarm=args.throughput_alarm,
                      val_rise_ratio=args.val_rise_ratio, stale_minutes=args.stale_minutes)
        verdict = assess(parsed=parsed, rules=rules, log_age_minutes=args.log_age_minutes, job_running=args.job_running)
    except Exception as err:  # noqa: BLE001 — a broken monitor must escalate, never read as "no alarm"
        verdict = Verdict(status="monitor_error", reasons=[f"{type(err).__name__}: {err}"], summary={})
    if args.json:
        print(json.dumps(_json_safe(asdict(verdict)), allow_nan=False))
    else:
        print(f"{verdict.status.upper()}: " + ("; ".join(verdict.reasons) or "no findings"))
        for key, value in verdict.summary.items():
            print(f"  {key}: {value}")
    return EXIT_CODES.get(verdict.status, 3)


if __name__ == "__main__":
    sys.exit(main())
