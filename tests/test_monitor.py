import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.monitor import Rules, assess, parse_log


def _train_line(step, loss, lr = 3e-3, tok = 20.0, gnorm = 0.9):
    ppl = math.exp(min(loss, 20)) if math.isfinite(loss) else float("inf")
    return f"step {step:>7d} | loss {loss:.4f} | ppl {ppl:.1f} | lr {lr:.2e} | {tok:.1f}k tok/s | {step * 0.0005:.3f}B | gnorm {gnorm:.3f}"


def _healthy(steps = 400, log_every = 10, start = 6.0, floor = 3.2, tok = 20.0):
    """A smoothly decreasing loss with mild noise, warmup at 50 steps."""
    lines = []
    for i in range(1, steps // log_every + 1):
        step = i * log_every
        loss = floor + (start - floor) * math.exp(-step / 120) + 0.01 * math.sin(i)
        lines.append(_train_line(step, loss, tok = tok))
        if step % 100 == 0:
            lines.append(f"step {step:>7d} | val_loss {loss + 0.05:.4f} | val_ppl {math.exp(loss + 0.05):.1f} | val_frac_near_certain 0.0100")
    return lines


RULES = Rules(warmup_steps = 50, window = 5)


def test_parse_reads_train_val_and_optional_gnorm():
    parsed = parse_log("\n".join(_healthy(100)) + "\nstep      110 | loss 3.4000 | ppl 30.0 | lr 3.00e-03 | 19.9k tok/s | 0.055B\n")
    assert parsed["train"][0]["step"] == 10 and parsed["train"][0]["gnorm"] == pytest.approx(0.9)
    assert parsed["train"][-1]["gnorm"] is None, "a log without gnorm (older train.py) must still parse"
    assert parsed["val"][0]["step"] == 100 and parsed["val"][0]["frac"] == pytest.approx(0.01)
    assert not parsed["failures"] and not parsed["done"]


def test_healthy_run_is_ok():
    verdict = assess(parse_log("\n".join(_healthy())), rules = RULES)
    assert verdict.status == "ok", verdict
    assert verdict.summary["last_step"] == 400 and verdict.summary["tok_per_s_median"] == pytest.approx(20e3)


def test_non_finite_loss_alarms():
    lines = _healthy(300) + [_train_line(310, float("nan"))]
    verdict = assess(parse_log("\n".join(lines)), rules = RULES)
    assert verdict.status == "alarm" and any("non-finite loss" in r for r in verdict.reasons)


def test_loss_spike_after_warmup_alarms_but_not_during_warmup():
    lines = _healthy(300)
    lines += [_train_line(310 + 10 * i, 5.0) for i in range(5)]      # a full window 50% above the best
    verdict = assess(parse_log("\n".join(lines)), rules = RULES)
    assert verdict.status == "alarm" and any("loss window mean" in r for r in verdict.reasons)
    # the same shape inside warmup is not judged (the loss legitimately moves there)
    early = [_train_line(10 * i, 6.0 - 0.1 * i) for i in range(1, 6)] + [_train_line(60 + 10 * i, 9.0) for i in range(5)]
    assert assess(parse_log("\n".join(early)), rules = Rules(warmup_steps = 200, window = 5)).status == "ok"


def test_grad_norm_spike_alarms_only_when_sustained():
    base = _healthy(300)
    sustained = base + [_train_line(310 + 10 * i, 3.3, gnorm = 12.0) for i in range(3)]
    assert assess(parse_log("\n".join(sustained)), rules = RULES).status == "alarm"
    single = base + [_train_line(310, 3.3, gnorm = 12.0), _train_line(320, 3.3), _train_line(330, 3.3)]
    assert assess(parse_log("\n".join(single)), rules = RULES).status == "ok"


def test_throughput_drop_watch_then_alarm():
    base = _healthy(300)
    assert assess(parse_log("\n".join(base + [_train_line(310, 3.3, tok = 14.0)])), rules = RULES).status == "watch"
    assert assess(parse_log("\n".join(base + [_train_line(310, 3.3, tok = 8.0)])), rules = RULES).status == "alarm"


def test_rising_validation_alarms():
    lines = _healthy(300)
    lines += [f"step {s:>7d} | val_loss {v:.4f} | val_ppl 1.0 | val_frac_near_certain 0.0100" for s, v in ((400, 3.30), (500, 3.35), (600, 3.40))]
    verdict = assess(parse_log("\n".join(lines)), rules = RULES)
    assert verdict.status == "alarm" and any("validation loss rose" in r for r in verdict.reasons)


def test_stall_needs_running_job_and_old_log():
    parsed = parse_log("\n".join(_healthy(200)))
    assert assess(parsed, rules = RULES, log_age_minutes = 45, job_running = True).status == "alarm"
    assert assess(parsed, rules = RULES, log_age_minutes = 45, job_running = False).status == "ok"
    assert assess(parsed, rules = RULES, log_age_minutes = 5, job_running = True).status == "ok"


def test_failure_and_done_markers():
    failed = parse_log("\n".join(_healthy(100)) + "\nTraceback (most recent call last):\n  File x\nRuntimeError: CUDA out of memory\n")
    assert assess(failed, rules = RULES).status == "failed"
    done = parse_log("\n".join(_healthy(100)) + "\nDone. 0.05B tokens in 0.1h\n")
    assert assess(done, rules = RULES).status == "done"


def test_flat_loss_is_a_watch_not_an_alarm():
    lines = _healthy(300) + [_train_line(310 + 10 * i, 3.25 + 0.002 * (i % 3)) for i in range(35)]
    verdict = assess(parse_log("\n".join(lines)), rules = RULES)
    assert verdict.status == "watch" and any("loss flat" in r for r in verdict.reasons)
