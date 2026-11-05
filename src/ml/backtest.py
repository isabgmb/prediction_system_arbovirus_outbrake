"""
Rolling-origin evaluation (backtest) of the LSTM configurations.

The paper evaluates on one split: train up to 2019-10, test on the rest of
2019. That is one season, 42 strongly autocorrelated weeks. This module
evaluates over several seasons instead:

- At each origin (by default every 26 weeks from 2018-01) the models are
  retrained on every week before the origin and used until the next origin.
- A forecast for week t is issued h weeks earlier, where h is the
  configuration's horizon (its newest input, e.g. 4 weeks for EDI lag 6). It
  may only use a model trained on data observed by then (cutoff <= t - h).
  The paper's single split breaks this rule for its first h - 1 test weeks.
- Each configuration is compared with persistence at the same horizon
  (predict week t with the value observed at t - h): skill score
  1 - MSE(model) / MSE(persistence), positive when the model is better, with a
  95% moving-block bootstrap interval that respects the autocorrelation.
- Per season it also reports the error in the size and in the timing of the
  peak, which r and RMSE hide.

Usage
-----
    python -m src.ml.backtest                                   # all 10 configurations
    python -m src.ml.backtest --configs dengue:1,edi:6 --reps 5 --run-id quick

Predictions are saved per configuration and origin as they finish, so a run
that is interrupted continues where it stopped when started again with the
same --run-id.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core import epiweek
from src.ml.preprocessing import build_inputs, lag_offsets, log2_transform

log = logging.getLogger("backtest")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "backtests"


# =============================================================================
# Metrics (no TensorFlow; covered by tests/unit/test_backtest.py)
# =============================================================================

def _label(week: tuple[int, int]) -> str:
    return f"{week[0]}-{week[1]:02d}"


def horizon(lag: int) -> int:
    """Weeks between a forecast's issue and its target: the newest input."""
    return min(lag_offsets(lag))


def assign_models(targets: np.ndarray, cutoffs: np.ndarray, h: int) -> np.ndarray:
    """
    For each target week, the index of the newest model whose training data
    ends no later than the forecast's issue week (cutoff <= target - h),
    or -1 when no model qualifies. cutoffs must be increasing.
    """
    issue = np.asarray(targets) - h
    return np.searchsorted(np.asarray(cutoffs), issue, side="right") - 1


def skill_score(y: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> float:
    """1 - MSE(pred) / MSE(ref): 0 = as good as ref, 1 = perfect, < 0 = worse."""
    return float(1.0 - np.mean((y - pred) ** 2) / np.mean((y - ref) ** 2))


def block_bootstrap_ci(y: np.ndarray, pred: np.ndarray, ref: np.ndarray,
                       block: int = 8, n_boot: int = 2000, seed: int = 0,
                       level: float = 0.95) -> tuple[float, float]:
    """
    Moving-block bootstrap interval for skill_score. Weeks are resampled in
    contiguous blocks so that the autocorrelation within a block is kept,
    instead of treating weeks as independent.
    """
    n = len(y)
    block = min(block, n)
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    n_blocks = int(np.ceil(n / block))
    e_pred, e_ref = (y - pred) ** 2, (y - ref) ** 2
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = (rng.choice(starts, n_blocks)[:, None] + np.arange(block)).ravel()[:n]
        stats[b] = 1.0 - e_pred[idx].mean() / e_ref[idx].mean()
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


@dataclass
class PeakError:
    observed: float
    predicted: float
    relative: float        # predicted / observed - 1 (-0.5 = half the size)
    timing_weeks: int      # predicted peak week - observed peak week (> 0 = late)


def peak_error(y: np.ndarray, pred: np.ndarray) -> PeakError:
    """Size and timing error of the highest week in a season."""
    io, ip = int(np.argmax(y)), int(np.argmax(pred))
    return PeakError(float(y[io]), float(pred[ip]), float(pred[ip] / y[io] - 1), ip - io)


def summarize(y: np.ndarray, pred: np.ndarray, ref: np.ndarray, seed: int = 0) -> dict:
    """Metrics of one forecast series against observations and the reference."""
    low, high = block_bootstrap_ci(y, pred, ref, seed=seed)
    pk, pk_ref = peak_error(y, pred), peak_error(y, ref)
    return {
        "weeks": int(len(y)),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "r": float(np.corrcoef(y, pred)[0, 1]),
        "skill": skill_score(y, pred, ref),
        "skill_ci95": [low, high],
        "ref_mae": float(np.mean(np.abs(y - ref))),
        "ref_rmse": float(np.sqrt(np.mean((y - ref) ** 2))),
        "peak": pk.__dict__,
        "ref_peak": pk_ref.__dict__,
    }


# =============================================================================
# Training at each origin
# =============================================================================

def default_origins(weeks: list[tuple[int, int]], first: tuple[int, int], every: int, h_max: int) -> list[int]:
    """Week indexes of the origins: from `first`, every `every` weeks, while a forecast remains."""
    start = weeks.index(first)
    return list(range(start, len(weeks) - h_max, every))


def origin_seed(base: int, config_index: int, origin: int, rep: int) -> int:
    """Fixed seed per (configuration, origin, repetition); distinct from train.py's."""
    return 10_000_000 * base + 100_000 * config_index + 100 * origin + rep


def predict_origin(x: np.ndarray, y: np.ndarray, lag: int, origin: int, n_reps: int,
                   seed_of) -> np.ndarray:
    """
    Train n_reps models on targets lag .. origin-1 and return their
    predictions (log2) for every target lag .. N-1, shape (n_reps, N - lag).
    """
    from src.ml import model as ml_model
    targets = np.arange(lag, len(y))
    train = np.arange(lag, origin)
    x_all = build_inputs(x, targets, lag)
    x_train = build_inputs(x, train, lag)
    out = []
    for rep in range(n_reps):
        m = ml_model.train_rep(x_train, y[train], seed=seed_of(rep))
        out.append(ml_model.predict_sequence(m, x_all))
    return np.stack(out)


def run_config(series, input_type: str, lag: int, origins: list[int], n_reps: int,
               base_seed: int, cache: Path) -> dict:
    """Backtest one configuration; returns targets, forecasts and persistence."""
    from src.ml.train import ALL_CONFIGS, config_name
    name = config_name(input_type, lag)
    ci = ALL_CONFIGS.index((input_type, lag))
    h = horizon(lag)
    y = log2_transform(series.dengue, "dengue")
    x = log2_transform(series.series(input_type), input_type)
    n = len(y)

    per_origin = []
    for o in origins:
        f = cache / f"{name}_o{o:03d}.npy"
        if not f.exists():
            t0 = time.time()
            p = predict_origin(x, y, lag, o, n_reps, lambda rep: origin_seed(base_seed, ci, o, rep))
            np.save(cache / "tmp.npy", p)
            (cache / "tmp.npy").rename(f)     # atomic: an interruption never leaves half a file
            log.info("%s origin %s: %d reps in %.0fs", name, _label(series.weeks[o]), n_reps, time.time() - t0)
        per_origin.append(np.load(f))

    cutoffs = np.array(origins) - 1                      # last week each model saw
    targets = np.arange(origins[0] - 1 + h, n)           # first week the first model may forecast
    which = assign_models(targets, cutoffs, h)
    assert (which >= 0).all()
    # Ensemble mean in linear scale, as the paper's Fig. 2 and the API.
    pred = np.array([np.mean(2.0 ** per_origin[k][:, t - lag]) for t, k in zip(targets, which)])
    return {
        "name": name, "horizon": h, "targets": targets, "model": which,
        "observed": series.dengue[targets], "forecast": pred,
        "persistence": series.dengue[targets - h],
    }


def evaluate(res: dict, weeks: list[tuple[int, int]]) -> dict:
    """Overall and per-season metrics (season = epidemiological year)."""
    t, y, p, ref = res["targets"], res["observed"], res["forecast"], res["persistence"]
    seasons = np.array([weeks[i][0] for i in t])
    out = {"horizon": res["horizon"], "all": summarize(y, p, ref)}
    for s in sorted(set(seasons)):
        m = seasons == s
        if m.sum() >= 26:                                 # skip a season with too few weeks
            out[str(s)] = summarize(y[m], p[m], ref[m])
    return out


# =============================================================================
# Report
# =============================================================================

def report(metrics: dict, meta: dict) -> str:
    lines = [
        f"## Backtest `{meta['run_id']}`",
        "",
        f"Origins every {meta['every']} weeks from {meta['origins'][0]} to {meta['origins'][-1]} "
        f"({len(meta['origins'])} refits), {meta['reps']} repetitions per refit, ensemble mean. "
        f"Each forecast is issued `h` weeks before its target and compared with persistence at the "
        f"same horizon (the value observed `h` weeks earlier).",
        "",
        "| Configuration | h | Period | Weeks | MAE | Persist. MAE | RMSE | Persist. RMSE | r | "
        "Skill vs persist. (95% CI) | Peak size | Peak timing |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        for period, s in m.items():
            if period == "horizon":
                continue
            pk = s["peak"]
            lines.append(
                f"| {name} | {m['horizon']} | {period} | {s['weeks']} | {s['mae']:.2f} | {s['ref_mae']:.2f} | "
                f"{s['rmse']:.2f} | {s['ref_rmse']:.2f} | {s['r']:.2f} | "
                f"{s['skill']:+.2f} ({s['skill_ci95'][0]:+.2f} to {s['skill_ci95'][1]:+.2f}) | "
                f"{pk['relative']:+.0%} | {pk['timing_weeks']:+d} wk |"
            )
    lines += [
        "",
        "Skill > 0 means a lower squared error than persistence; an interval that contains 0 means the "
        "difference is not established. Peak size is the forecast's highest week relative to the observed "
        "highest week of the period (-50% = half the size); peak timing is how many weeks later (+) or "
        "earlier (-) it comes. Persistence is late by `h` weeks by construction.",
    ]
    return "\n".join(lines)


def plot(results: list[dict], weeks: list[tuple[int, int]], origins: list[int], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 3 * len(results)), squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        d = [epiweek.week_start(*weeks[i]) for i in res["targets"]]
        ax.plot(d, res["observed"], color="k", lw=1.2, label="Observed")
        ax.plot(d, res["forecast"], lw=1.5, label="LSTM (ensemble mean)")
        ax.plot(d, res["persistence"], lw=1, ls="--", color="gray", label=f"Persistence (h = {res['horizon']})")
        for o in origins:
            ax.axvline(epiweek.week_start(*weeks[o]), color="lightgray", lw=0.8, zorder=0)
        ax.set_title(res["name"], fontsize=11)
        ax.set_ylabel("Mean weekly cases")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    from src.ml import data as ml_data
    from src.ml import model as ml_model
    from src.ml.train import parse_configs

    ap = argparse.ArgumentParser(prog="python -m src.ml.backtest", description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", help="e.g. dengue:1,edi:6 (default: all 10)")
    ap.add_argument("--reps", type=int, default=10, help="repetitions per refit (default 10)")
    ap.add_argument("--first-origin", default="2018-01", help="epidemiological week YYYY-WW (default 2018-01)")
    ap.add_argument("--refit-every", type=int, default=26, help="weeks between refits (default 26)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-id", default=None, help="folder under data/backtests (reuse it to resume)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    ml_model.enable_determinism()
    configs = parse_configs(args.configs)
    series = ml_data.load_from_raw()
    weeks = series.weeks
    first = epiweek.parse_iso_label(args.first_origin)
    origins = default_origins(weeks, first, args.refit_every, max(horizon(lag) for _, lag in configs))
    run_id = args.run_id or time.strftime("bt-%Y%m%d-%H%M%S")
    out = OUT_DIR / run_id
    cache = out / "predictions"
    cache.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "reps": args.reps, "every": args.refit_every, "seed": args.seed,
            "origins": [_label(weeks[o]) for o in origins],
            "tensorflow": ml_model.tf().__version__}
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    log.info("Backtest %s: %d configurations, origins %s, %d reps", run_id, len(configs), meta["origins"], args.reps)

    results, metrics = [], {}
    for input_type, lag in configs:
        res = run_config(series, input_type, lag, origins, args.reps, args.seed, cache)
        results.append(res)
        metrics[res["name"]] = evaluate(res, weeks)
        s = metrics[res["name"]]["all"]
        log.info("%-12s skill %+.2f (%+.2f to %+.2f) | MAE %.2f vs %.2f persistence", res["name"], s["skill"],
                 *s["skill_ci95"], s["mae"], s["ref_mae"])

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "report.md").write_text(report(metrics, meta) + "\n")
    plot(results, weeks, origins, out / "backtest.png")
    print(report(metrics, meta))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
