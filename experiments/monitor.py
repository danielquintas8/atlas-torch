"""Training-run monitor: parse a train.py log and judge whether the run needs
attention. Pure stdlib, so it runs on the cluster login node against a live
log:

    python experiments/monitor.py runs/170m-atlas-mac_123.log --warmup-steps 2000 --json

Verdicts: ok / watch / alarm / failed / done. The rules are deliberately
mechanical and pre-declared — the monitor never changes a run; it wakes a
human (or the watchdog loop that calls it) with the evidence:

  alarm  - non-finite loss or grad norm; loss over the latest window above
           (1 + loss_spike_ratio) x the best earlier window, after warmup;
           grad norm above grad_norm_spike x its rolling median for
           grad_norm_consecutive consecutive log lines; throughput below
           throughput_alarm x the run's median; validation loss up at
           val_rises consecutive validations; a log that has not grown for
           stale_minutes while the job is running
  failed - Traceback / CUDA error / SLURM error lines
  done   - the final "Done." line
  watch  - throughput below throughput_watch x median; loss flat (no
           improvement of the window mean over flat_windows windows)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field

TRAIN_RE = re.compile(
    r"^step\s+(?P<step>\d+) \| loss (?P<loss>[-\d.einfa]+) \| ppl (?P<ppl>[\d.einfa]+) \| "
    r"lr (?P<lr>[\d.e+-]+) \| (?P<tok>[\d.]+)k tok/s \| (?P<tokens>[\d.]+)B"
    r"(?: \| gnorm (?P<gnorm>[-\d.einfa]+))?"
)
VAL_RE = re.compile(r"^step\s+(?P<step>\d+) \| val_loss (?P<val>[\d.einfa]+) \|.*val_frac_near_certain (?P<frac>[\d.]+)")
FAILURE_MARKERS = ("Traceback (most recent call last)", "CUDA out of memory", "CUDA error", "NCCL", "slurmstepd: error", "DUE TO TIME LIMIT", "Segmentation fault")
DONE_MARKER = "Done."


@dataclass
class Rules:
    warmup_steps: int = 2000
    window: int = 20                 # log lines per window (x log_every steps)
    loss_spike_ratio: float = 0.10   # latest window mean > (1 + ratio) x best earlier window mean
    flat_windows: int = 5            # no improvement over this many windows -> watch
    grad_norm_spike: float = 5.0     # x rolling median
    grad_norm_consecutive: int = 3
    throughput_watch: float = 0.8    # x median tok/s
    throughput_alarm: float = 0.5
    val_rises: int = 2               # consecutive validations with rising loss
    stale_minutes: float = 20.0


@dataclass
class Verdict:
    status: str                      # ok | watch | alarm | failed | done
    reasons: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _num(text):
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_log(text):
    """-> dict(train=[{step, loss, lr, tok, tokens, gnorm}], val=[{step, val, frac}], failures=[lines], done=bool)."""
    train, val, failures, done = [], [], [], False
    for line in text.splitlines():
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
        if any(marker in line for marker in FAILURE_MARKERS):
            failures.append(line.strip()[:200])
        if line.startswith(DONE_MARKER):
            done = True
    return dict(train=train, val=val, failures=failures, done=done)


def _windows(values, size):
    return [values[i:i + size] for i in range(0, len(values) - size + 1, size)]


def assess(parsed, rules=None, log_age_minutes=None, job_running=None):
    """Apply the rules to a parsed log. `log_age_minutes` (time since the log
    last grew) with `job_running=True` feeds the stall rule."""
    rules = rules or Rules()
    train, val = parsed["train"], parsed["val"]
    reasons, watch = [], []
    summary = dict(steps=len(train), last_step=train[-1]["step"] if train else 0)

    if parsed["failures"]:
        return Verdict("failed", parsed["failures"][:5], summary)
    if parsed["done"]:
        summary["last_loss"] = train[-1]["loss"] if train else None
        return Verdict("done", ["training finished"], summary)
    if not train:
        if job_running and log_age_minutes is not None and log_age_minutes > rules.stale_minutes:
            reasons.append(f"no training lines and the log has not grown for {log_age_minutes:.0f} min")
        return Verdict("alarm" if reasons else "ok", reasons, summary)

    last = train[-1]
    summary.update(last_loss=last["loss"], last_lr=last["lr"], last_gnorm=last["gnorm"], tokens_seen=last["tokens"])

    # non-finite
    if not math.isfinite(last["loss"]) or any(not math.isfinite(t["loss"]) for t in train[-3:]):
        reasons.append(f"non-finite loss at step {last['step']}")
    gnorms = [t["gnorm"] for t in train if t["gnorm"] is not None]
    if gnorms and any(not math.isfinite(g) for g in gnorms[-3:]):
        reasons.append(f"non-finite grad norm at step {last['step']}")

    # loss spike / flat, after warmup, on window means
    post = [t for t in train if t["step"] > rules.warmup_steps and math.isfinite(t["loss"])]
    wins = _windows([t["loss"] for t in post], rules.window)
    if len(wins) >= 2:
        means = [statistics.fmean(w) for w in wins]
        best_earlier = min(means[:-1])
        summary.update(window_mean=means[-1], best_window_mean=best_earlier)
        if means[-1] > (1 + rules.loss_spike_ratio) * best_earlier:
            reasons.append(f"loss window mean {means[-1]:.4f} is {means[-1] / best_earlier - 1:+.1%} vs best earlier window {best_earlier:.4f}")
        elif len(means) > rules.flat_windows and min(means[-rules.flat_windows:]) >= min(means[:-rules.flat_windows]) * 0.995:
            watch.append(f"loss flat: no window improved on {min(means[:-rules.flat_windows]):.4f} over the last {rules.flat_windows} windows")

    # grad norm spikes vs rolling median
    if len(gnorms) >= 2 * rules.grad_norm_consecutive + 4:
        finite = [g for g in gnorms if math.isfinite(g)]
        median = statistics.median(finite[:-rules.grad_norm_consecutive]) if len(finite) > rules.grad_norm_consecutive else math.nan
        summary["gnorm_median"] = median
        recent = finite[-rules.grad_norm_consecutive:]
        if math.isfinite(median) and median > 0 and all(g > rules.grad_norm_spike * median for g in recent):
            reasons.append(f"grad norm {recent} > {rules.grad_norm_spike}x median {median:.3f} for {rules.grad_norm_consecutive} consecutive logs")

    # throughput vs the run's median (skip the first line: warm start)
    toks = [t["tok"] for t in train[1:] if t["tok"] > 0]
    if len(toks) >= 5:
        median_tok = statistics.median(toks)
        summary.update(tok_per_s=last["tok"], tok_per_s_median=median_tok)
        if last["tok"] < rules.throughput_alarm * median_tok:
            reasons.append(f"throughput {last['tok'] / 1e3:.1f}k tok/s < {rules.throughput_alarm}x median {median_tok / 1e3:.1f}k")
        elif last["tok"] < rules.throughput_watch * median_tok:
            watch.append(f"throughput {last['tok'] / 1e3:.1f}k tok/s < {rules.throughput_watch}x median {median_tok / 1e3:.1f}k")

    # validation trend
    if val:
        summary["val"] = [(v["step"], round(v["val"], 4)) for v in val[-4:]]
        rises = 0
        for prev, cur in zip(val[:-1], val[1:]):
            rises = rises + 1 if cur["val"] > prev["val"] else 0
        if rises >= rules.val_rises:
            reasons.append(f"validation loss rose at {rises} consecutive validations: {summary['val']}")

    # stall
    if job_running and log_age_minutes is not None and log_age_minutes > rules.stale_minutes:
        reasons.append(f"log has not grown for {log_age_minutes:.0f} min while the job is running")

    if reasons:
        return Verdict("alarm", reasons, summary)
    return Verdict("watch" if watch else "ok", watch, summary)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--warmup-steps", type=int, default=Rules.warmup_steps)
    p.add_argument("--window", type=int, default=Rules.window)
    p.add_argument("--log-age-minutes", type=float, default=None, help="minutes since the log last grew (the caller measures it)")
    p.add_argument("--job-running", action="store_true", help="the SLURM job is RUNNING (enables the stall rule)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    with open(args.log) as f:
        parsed = parse_log(f.read())
    verdict = assess(parsed, rules=Rules(warmup_steps=args.warmup_steps, window=args.window),
                     log_age_minutes=args.log_age_minutes, job_running=args.job_running)
    if args.json:
        print(json.dumps(asdict(verdict)))
    else:
        print(f"{verdict.status.upper()}: " + ("; ".join(verdict.reasons) or "no findings"))
        for key, value in verdict.summary.items():
            print(f"  {key}: {value}")
    raise SystemExit(2 if verdict.status in ("alarm", "failed") else 0)


if __name__ == "__main__":
    main()
