import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.monitor import LN_VOCAB_T5, Rules, assess, format_train_line, format_val_line, main, parse_log

# the fixture is the producer itself: train.py prints exactly these lines
def _train(step, loss, lr = 3e-3, tok = 20e3, gnorm = 0.9):
    return format_train_line(step = step, loss = loss, lr = lr, tok_per_sec = tok, tokens_total = step * 500_000, grad_norm_max = gnorm)


def _val(step, val, frac = 0.01):
    return format_val_line(step = step, val_loss = val, frac_near_certain = frac)


def _healthy(steps = 400, log_every = 10, start = 6.0, floor = 3.2, tok = 20e3):
    lines = []
    for i in range(1, steps // log_every + 1):
        step = i * log_every
        loss = floor + (start - floor) * math.exp(-step / 120) + 0.01 * math.sin(i)
        lines.append(_train(step, loss, tok = tok))
        if step % 100 == 0:
            lines.append(_val(step, loss + 0.05))
    return lines


RULES = Rules(warmup_steps = 50, window = 5, sanity_steps = 100)


def test_parse_reads_train_val_optional_gnorm_and_old_val_format():
    text = "\n".join(_healthy(100)) + "\nstep      110 | loss 3.4000 | ppl 30.0 | lr 3.00e-03 | 19.9k tok/s | 0.055B\n" \
           "step     200 | val_loss 3.4100 | val_ppl 30.3\n"
    parsed = parse_log(text = text)
    assert parsed["train"][0]["step"] == 10 and parsed["train"][0]["gnorm"] == pytest.approx(0.9)
    assert parsed["train"][-1]["gnorm"] is None, "a log without gnorm (older train.py) must still parse"
    assert parsed["val"][0]["frac"] == pytest.approx(0.01)
    assert parsed["val"][-1]["step"] == 200 and math.isnan(parsed["val"][-1]["frac"]), "a val line without the frac field must still count"
    assert not parsed["failures"] and not parsed["done"] and not parsed["timeout"]


def test_non_finite_renderings_parse_as_nan():
    lines = [_train(310, float("nan")), _train(320, 3.0, gnorm = float("nan")), _train(330, 3.0, gnorm = float("inf"))]
    parsed = parse_log(text = "\n".join(lines))
    assert math.isnan(parsed["train"][0]["loss"])
    assert math.isnan(parsed["train"][1]["gnorm"]) and math.isinf(parsed["train"][2]["gnorm"])


def test_healthy_run_is_ok():
    verdict = assess(parsed = parse_log(text = "\n".join(_healthy())), rules = RULES)
    assert verdict.status == "ok", verdict
    assert verdict.summary["last_step"] == 400 and verdict.summary["tok_per_s_median"] == pytest.approx(20e3)


def test_non_finite_loss_alarms():
    verdict = assess(parsed = parse_log(text = "\n".join(_healthy(300) + [_train(310, float("nan"))])), rules = RULES)
    assert verdict.status == "alarm" and any("non-finite loss" in r for r in verdict.reasons)


def test_non_finite_grad_norm_with_finite_loss_alarms():
    """The earliest signal: the norm blows up while the loss is still finite."""
    verdict = assess(parsed = parse_log(text = "\n".join(_healthy(300) + [_train(310, 3.3, gnorm = float("nan"))])), rules = RULES)
    assert verdict.status == "alarm" and any("non-finite grad norm" in r for r in verdict.reasons)


def test_loss_above_ln_vocab_after_sanity_steps_alarms():
    stuck = [_train(10 * i, LN_VOCAB_T5 + 0.5) for i in range(1, 12)]
    assert assess(parsed = parse_log(text = "\n".join(stuck)), rules = RULES).status == "alarm"
    assert assess(parsed = parse_log(text = "\n".join(stuck[:5])), rules = RULES).status == "ok", "before sanity_steps the loss may still be near uniform"


def test_trailing_loss_window_catches_a_sustained_rise_immediately():
    """The trailing window must include the newest lines: a 4x loss sustained
    for one full window alarms whatever the alignment (the first version
    strode by the window size and left up to window-1 newest lines unseen)."""
    base = _healthy(300)
    for extra in range(RULES.window, RULES.window + 4):          # every alignment of the tail
        lines = base + [_train(310 + 10 * i, 12.0) for i in range(extra)]
        verdict = assess(parsed = parse_log(text = "\n".join(lines)), rules = RULES)
        assert verdict.status == "alarm" and any("trailing loss window" in r for r in verdict.reasons), extra
    # a steep but healthy decrease never alarms
    decreasing = [_train(10 * i, 8.0 * math.exp(-i / 10) + 3.0) for i in range(1, 60)]
    assert assess(parsed = parse_log(text = "\n".join(decreasing)), rules = RULES).status == "ok"
    # a rise inside warmup is not judged
    early = [_train(10 * i, 6.0 - 0.1 * i) for i in range(1, 6)] + [_train(60 + 10 * i, 9.0) for i in range(5)]
    assert assess(parsed = parse_log(text = "\n".join(early)), rules = Rules(warmup_steps = 200, window = 5, sanity_steps = 500)).status == "ok"


def test_grad_norm_spike_alarms_only_when_sustained():
    base = _healthy(300)
    sustained = base + [_train(310 + 10 * i, 3.3, gnorm = 12.0) for i in range(2)]
    assert assess(parsed = parse_log(text = "\n".join(sustained)), rules = RULES).status == "alarm"
    single = base + [_train(310, 3.3, gnorm = 12.0), _train(320, 3.3)]
    assert assess(parsed = parse_log(text = "\n".join(single)), rules = RULES).status == "ok"


def test_throughput_drop_watch_then_alarm():
    base = _healthy(300)
    assert assess(parsed = parse_log(text = "\n".join(base + [_train(310, 3.3, tok = 14e3)])), rules = RULES).status == "watch"
    assert assess(parsed = parse_log(text = "\n".join(base + [_train(310, 3.3, tok = 8e3)])), rules = RULES).status == "alarm"


def test_validation_alarm_needs_consecutive_rises_and_magnitude():
    base = _healthy(300)
    noise = base + [_val(s, v) for s, v in ((400, 3.300), (500, 3.301), (600, 3.302))]      # rises, but within 1% of the best
    assert assess(parsed = parse_log(text = "\n".join(noise)), rules = RULES).status == "ok"
    real = base + [_val(s, v) for s, v in ((400, 3.30), (500, 3.36), (600, 3.42))]
    verdict = assess(parsed = parse_log(text = "\n".join(real)), rules = RULES)
    assert verdict.status == "alarm" and any("validation loss" in r for r in verdict.reasons)
    single_jump = base + [_val(s, v) for s, v in ((400, 3.30), (500, 3.45), (600, 3.44))]  # one jump, then a drop: not consecutive
    assert assess(parsed = parse_log(text = "\n".join(single_jump)), rules = RULES).status == "ok"


def test_stall_needs_running_job_and_old_log():
    parsed = parse_log(text = "\n".join(_healthy(200)))
    assert assess(parsed = parsed, rules = RULES, log_age_minutes = 45, job_running = True).status == "alarm"
    assert assess(parsed = parsed, rules = RULES, log_age_minutes = 45, job_running = False).status == "ok"
    assert assess(parsed = parsed, rules = RULES, log_age_minutes = 5, job_running = True).status == "ok"
    banner_only = parse_log(text = "===== SMOKE =====\nsome startup banner\n")
    assert assess(parsed = banner_only, rules = RULES, log_age_minutes = 45, job_running = True).status == "alarm"


def test_failure_markers_are_precise_and_keep_the_numeric_findings():
    benign = "\n".join(_healthy(300) + ["NCCL version 2.21.5+cuda12.4", "[rank0]:[W ProcessGroupNCCL.cpp:1250] WARNING: process group has NOT been destroyed"])
    assert assess(parsed = parse_log(text = benign), rules = RULES).status == "ok"
    failed = "\n".join(_healthy(300) + [_train(310, float("nan")), "Traceback (most recent call last):", "  File x", "RuntimeError: CUDA out of memory"])
    verdict = assess(parsed = parse_log(text = failed), rules = RULES)
    assert verdict.status == "failed" and any("non-finite loss" in r for r in verdict.reasons), "the numeric evidence must survive a failure"
    timeout = "\n".join(_healthy(300) + ["slurmstepd: error: *** JOB 1 ON n1 CANCELLED AT 2026-09-03T10:00:00 DUE TO TIME LIMIT ***"])
    assert assess(parsed = parse_log(text = timeout), rules = RULES).status == "timeout"
    done = "\n".join(_healthy(100)) + "\nDone. 0.05B tokens in 0.1h\n"
    assert assess(parsed = parse_log(text = done), rules = RULES).status == "done"


def test_cli_exit_codes_and_json(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text("\n".join(_healthy(300)) + "\n", encoding = "utf-8")
    assert main(argv = [str(log), "--warmup-steps", "50", "--window", "5", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok" and out["summary"]["last_step"] == 300

    log.write_text("\n".join(_healthy(300) + [_train(310, float("nan"))]) + "\n", encoding = "utf-8")
    assert main(argv = [str(log), "--warmup-steps", "50", "--window", "5", "--json"]) == 2
    out = json.loads(capsys.readouterr().out)          # NaN summary values must serialize (as null)
    assert out["status"] == "alarm" and out["summary"]["last_loss"] is None

    log.write_bytes(("\n".join(_healthy(100)) + "\n").encode() + b"caf\xe9 bad byte\n")
    assert main(argv = [str(log), "--warmup-steps", "50", "--window", "5"]) == 0, "one non-UTF8 byte must not break the monitor"
    capsys.readouterr()
    assert main(argv = [str(tmp_path / "missing.log"), "--json"]) == 3, "a monitor that cannot judge escalates, never reads as ok"
    assert json.loads(capsys.readouterr().out)["status"] == "monitor_error"
    assert main(argv = [str(log), "--window", "0", "--json"]) == 3
