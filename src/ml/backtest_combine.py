"""
Combination of the LSTM and persistence, evaluated from a finished backtest.

Reads the predictions a backtest saved (data/backtests/<run_id>/predictions/)
and evaluates, for every configuration found there, the forecast

    weight * LSTM ensemble mean + (1 - weight) * persistence

with the same metrics as src/ml/backtest.py (skill against persistence with a
95% block-bootstrap interval, peak size and timing per season). Nothing is
retrained: if a configuration is missing predictions for some origin, the
command stops and asks for the backtest to be finished first.

The weight is fixed before looking at the results (default 0.5, the plain
average; see "Criteria fixed before the full backtest" in
docs/REPRODUCTION.md). It is never tuned on 2018-2019, the seasons being
evaluated, so the combination is as out-of-sample as the backtest itself.

It also checks each configuration, alone and combined, against the criterion
for a model to set the API's risk level, written in the same section.

Usage
-----
    python -m src.ml.backtest_combine --run-id paper-full
    python -m src.ml.backtest_combine --run-id paper-full --weight 0.5

Output in data/backtests/<run_id>/: combination_w050.md and combination_w050.json
(the number is the weight in hundredths).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.core import epiweek
from src.ml import backtest as bt

log = logging.getLogger("backtest_combine")

# Criterion for a model to set the API's risk level (docs/REPRODUCTION.md):
# in every one of these seasons, skill with the lower 95% bound above zero and
# the forecast peak within +-25% of the observed peak.
CRITERION_SEASONS = ("2018", "2019")
MAX_PEAK_ERROR = 0.25


# =============================================================================
# Pure functions (no TensorFlow; covered by tests/unit/test_backtest_combine.py)
# =============================================================================

def combine(res: dict, weight: float) -> dict:
    """
    A backtest result (as returned by backtest.run_config) whose forecast is
    replaced by weight * forecast + (1 - weight) * persistence. weight 1 is
    the LSTM alone, 0 is persistence.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be between 0 and 1, got {weight}")
    out = dict(res)
    out["forecast"] = weight * np.asarray(res["forecast"]) + (1.0 - weight) * np.asarray(res["persistence"])
    return out


def meets_criterion(metrics: dict, seasons: tuple[str, ...] = CRITERION_SEASONS,
                    max_peak_error: float = MAX_PEAK_ERROR) -> bool:
    """
    Whether one configuration's metrics (backtest.evaluate) meet the criterion
    in every season. A season that was not evaluated counts as not met.
    """
    for s in seasons:
        m = metrics.get(s)
        if m is None or not (m["skill_ci95"][0] > 0 and abs(m["peak"]["relative"]) <= max_peak_error):
            return False
    return True


def prediction_path(cache: Path, name: str, origin: int) -> Path:
    """Where backtest.run_config saves one configuration's predictions for one origin."""
    return cache / f"{name}_o{origin:03d}.npy"


def origins_from_run(meta: dict, weeks: list[tuple[int, int]]) -> list[int]:
    """Week indexes of the origins a backtest used, from the labels in its run.json."""
    return [weeks.index(epiweek.parse_iso_label(label)) for label in meta["origins"]]


def available_configs(cache: Path, origins: list[int],
                      configs: tuple[tuple[str, int], ...] | None = None) -> list[tuple[str, int]]:
    """
    Configurations with predictions for every origin. A configuration with
    none is skipped (it was not part of the run); one with some but not all
    stops with an error, since evaluating it would mean retraining.
    """
    from src.ml.train import ALL_CONFIGS, config_name
    found = []
    for input_type, lag in configs or ALL_CONFIGS:
        name = config_name(input_type, lag)
        missing = [o for o in origins if not prediction_path(cache, name, o).exists()]
        if len(missing) == len(origins):
            continue
        if missing:
            raise SystemExit(f"{name} has no predictions for origin(s) {missing}; finish the backtest first "
                             f"(run it again with the same --run-id).")
        found.append((input_type, lag))
    return found


# =============================================================================
# Report
# =============================================================================

def _skill(s: dict) -> str:
    return f"{s['skill']:+.2f} ({s['skill_ci95'][0]:+.2f} to {s['skill_ci95'][1]:+.2f})"


def report(lstm: dict, combined: dict, meta: dict, weight: float) -> str:
    lines = [
        f"## Combination with persistence: backtest `{meta['run_id']}`",
        "",
        f"Forecast = {weight:g} x LSTM ensemble mean + {1 - weight:g} x persistence, from the predictions the "
        f"backtest saved ({meta['reps']} repetitions per refit, origins {meta['origins'][0]} to "
        f"{meta['origins'][-1]}); nothing retrained. Skill is against persistence at the same horizon.",
        "",
        "| Configuration | h | Period | LSTM skill (95% CI) | LSTM peak size | "
        "Combination skill (95% CI) | Combination peak size | Combination peak timing |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in lstm.items():
        c = combined[name]
        for period, s in m.items():
            if period == "horizon":
                continue
            k = c[period]
            lines.append(
                f"| {name} | {m['horizon']} | {period} | {_skill(s)} | {s['peak']['relative']:+.0%} | "
                f"{_skill(k)} | {k['peak']['relative']:+.0%} | {k['peak']['timing_weeks']:+d} wk |"
            )
    seasons = " and ".join(CRITERION_SEASONS)
    lines += [
        "",
        f"Criterion for a model to set the API's risk level (docs/REPRODUCTION.md): in {seasons}, "
        f"lower bound of the skill interval above zero and peak size within ±{MAX_PEAK_ERROR:.0%}.",
        "",
        "| Configuration | LSTM meets criterion | Combination meets criterion |",
        "|---|---|---|",
    ]
    for name in lstm:
        lines.append(f"| {name} | {'yes' if meets_criterion(lstm[name]) else 'no'} | "
                     f"{'yes' if meets_criterion(combined[name]) else 'no'} |")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    from src.ml import data as ml_data

    ap = argparse.ArgumentParser(prog="python -m src.ml.backtest_combine", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", required=True, help="a finished backtest under data/backtests")
    ap.add_argument("--weight", type=float, default=0.5,
                    help="weight of the LSTM; the rest goes to persistence (default 0.5, fixed in advance)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    run_dir = bt.OUT_DIR / args.run_id
    if not (run_dir / "run.json").exists():
        raise SystemExit(f"No backtest in {run_dir}; run python -m src.ml.backtest --run-id {args.run_id} first.")
    meta = json.loads((run_dir / "run.json").read_text())
    series = ml_data.load_from_raw()
    weeks = series.weeks
    origins = origins_from_run(meta, weeks)
    cache = run_dir / "predictions"
    configs = available_configs(cache, origins)
    if not configs:
        raise SystemExit(f"No predictions in {cache}.")

    lstm, combined = {}, {}
    for input_type, lag in configs:
        # Every prediction file exists (checked above), so run_config only reads them.
        res = bt.run_config(series, input_type, lag, origins, meta["reps"], meta["seed"], cache)
        lstm[res["name"]] = bt.evaluate(res, weeks)
        combined[res["name"]] = bt.evaluate(combine(res, args.weight), weeks)

    tag = f"combination_w{round(args.weight * 100):03d}"
    text = report(lstm, combined, meta, args.weight)
    (run_dir / f"{tag}.md").write_text(text + "\n", encoding="utf-8")
    (run_dir / f"{tag}.json").write_text(json.dumps({
        "run_id": meta["run_id"], "weight": args.weight,
        "criterion": {"seasons": list(CRITERION_SEASONS), "max_peak_error": MAX_PEAK_ERROR},
        "lstm": lstm, "combination": combined,
    }, indent=1))
    print(text)
    print(f"\nSaved to {run_dir / tag}.md and .json")


if __name__ == "__main__":
    main()
