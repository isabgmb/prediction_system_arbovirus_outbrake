# Reproducing the paper's LSTM results

How the Python port compares with the paper's MATLAB results (`Run1DLSTM_01.m`,
`ModelsPerform_1DLSTM_01.m`), what was changed to get there, and what is still open.

**Code version.** The results in this document are reproduced by the code at the git tag
`paper-reproduction-v1`: the faithful port of the paper's training, before any change meant to
improve it. Later changes to training are evaluated against these results and documented in their
own sections, each naming its commit; a change that does not pass is reverted, and its result is
kept here. To rerun with the code and the package versions of these runs:
`git checkout paper-reproduction-v1`, then `pip install -r requirements-lock.txt` in a new
virtual environment. The lock file was generated on Windows; on another system use
`requirements.txt` and expect differences in the last decimals.

## Result (run `lstm-paper-30r`)

10 configurations × 30 repetitions, seed 1, data from the raw files; 166 weeks for training
(2016-01 to 2019-10), test weeks 2019-11 to 2019-52 — the paper's split. 79 minutes on 2 CPU
cores, TensorFlow 2.21. Reproduce with `python -m src.ml.train --run-id <name>` and
`python -m src.ml.evaluate --run data/models/<name>`.

| Configuration | Paper r | Ours r | Paper RMSE | Ours RMSE | Naive baseline (same lead) r / RMSE |
|---|---|---|---|---|---|
| dengue lag 1 (week i−1) | 0.899 | 0.614 ± 0.029 | 4.18 | 7.68 ± 0.28 | 0.929 / 2.85 |
| dengue lag 3 (i−3…i−1) | 0.845 | 0.333 ± 0.066 | 6.02 | 9.58 ± 0.25 | 0.929 / 2.85 |
| dengue lag 4 (i−4…i−2) | 0.731 | −0.167 ± 0.061 | 7.28 | 10.58 ± 0.13 | 0.863 / 3.95 |
| dengue lag 5 (i−5…i−3) | 0.661 | −0.108 ± 0.072 | 7.71 | 10.43 ± 0.19 | 0.802 / 4.76 |
| dengue lag 6 (i−6…i−4) | 0.537 | −0.244 ± 0.055 | 8.45 | 10.50 ± 0.14 | 0.745 / 5.40 |
| EDI lag 1 | 0.851 | **0.883** ± 0.016 | 4.82 | 7.33 ± 0.39 | 0.929 / 2.85 |
| EDI lag 3 | 0.803 | **0.836** ± 0.025 | 4.88 | 7.26 ± 0.44 | 0.929 / 2.85 |
| EDI lag 4 | 0.784 | **0.824** ± 0.018 | 5.44 | 8.07 ± 0.43 | 0.863 / 3.95 |
| EDI lag 5 | 0.850 | 0.798 ± 0.015 | 4.64 | 8.45 ± 0.43 | 0.802 / 4.76 |
| EDI lag 6 | 0.879 | 0.794 ± 0.014 | 4.35 | 8.77 ± 0.43 | 0.745 / 5.40 |

r and RMSE are averaged over repetitions (± standard error, with N−1 as in MATLAB's `std`),
on mean weekly cases per neighborhood, as in the paper. A second run with deterministic ops
(`lstm-paper-30r-det`, TensorFlow 2.21, 1 CPU core) gave the same r and RMSE for all 10
configurations. The naive baseline predicts each week with the value observed
as many weeks earlier as the configuration's newest input — the same information the model has.

### What this shows

- **EDI (ovitrap) models reproduce the paper's correlations** (0.79–0.88 vs 0.78–0.88): they
  follow the 2019 epidemic curve. Their error is about 1.7× the paper's because they
  **underestimate the peak** — the predicted curve has the right shape at roughly half the
  height (outputs/lstm-paper-30r/Fig2_best_models.png).
- **Dengue-only models do not reproduce**: r 0.61 at lag 1, falling below zero at longer lags.
- **Naive baselines matter.** In the paper's own results, "same as last week" (r 0.93,
  RMSE 2.85) beats every model with a 1-week lead, and the paper's best model (dengue lag 1)
  correlates 0.98 with that baseline — it learned to repeat last week. The case for the system is
  the **4-week lead from ovitrap data**: EDI lag 6 beats "same as 4 weeks earlier" on correlation
  in both the paper (0.879 vs 0.745) and our run (0.794 vs 0.745), though not yet on RMSE in ours.

## What the first port got wrong (fixed)

1. **Training regime.** The port fed ~160 one-week sequences without carrying state, then
   predicted with a stateful copy warmed up over the full history — a regime never seen in
   training (one repetition gave RMSE 309 on a series whose maximum is 41). MATLAB's
   `trainNetwork` with a matrix input trains on **one continuous sequence**; the port now does too.
2. **Missing MATLAB defaults** that the script never sets but MATLAB applies: gradient clipping
   at 1 (`GradientThreshold`, set in the script but not ported), half mean-squared-error loss
   (`regressionLayer`), L2 regularization 1e-4, Adam ε 1e-8. In a 5-repetition probe the loss
   and L2 defaults raised test r from 0.40 to 0.57 (dengue lag 1) and 0.68 to 0.72 (EDI lag 6).
3. **Prediction** now equals MATLAB's `resetState` + `predictAndUpdateState` over training then
   test inputs (one pass over the joined sequence).
4. Data checks: the city-wide series from the raw files equals the database's
   (`python -m src.ml.data`), and `YTest` equals the paper's.

## Still open

The investigation was time-boxed. Hypotheses for the remaining gap, roughly in order of
likelihood:

- **Initialization and randomness.** MATLAB and Keras draw different initial weights and the
  dengue models are unstable across seeds (standard error 0.03–0.07 on r). The paper's 30
  repetitions may have landed differently.
- **Exact loss and clipping arithmetic.** MATLAB's regression loss normalization and its
  per-parameter clipping interact with Adam; small differences change how far a single long
  sequence is fitted (training-period fit is r ≈ 0.98 here, i.e. heavy memorization).
- **MATLAB version behaviour** (the results file is from 2022) — e.g. input normalization defaults
  in `sequenceInputLayer`.

Re-running `Run1DLSTM_01.m` in MATLAB with the training-period fit printed (and one fixed seed)
would settle most of these. Until then, the EDI models are the reproduced part of the paper.

Diagnostics run since (not committed as code) narrow this down. The port's forward pass equals a
step-by-step NumPy LSTM written from MATLAB's equations (max difference 1.6e-6), so prediction is
not the cause. Test performance peaks early and then degrades: for dengue lag 1, test r is about
0.93 at epochs 50–100 in all 7 repetitions checked and falls to about 0.62 at epoch 250, after the
learning-rate drop, while the training fit keeps improving. The paper's r of 0.899 lies on that
curve, around epoch 120–130. The gap is therefore how far each implementation overfits by epoch
250, not a porting bug; per-epoch curves from MATLAB would show whether its training overfits
more slowly.

## Rolling-origin evaluation

The paper's test is one season (2019-11 to 2019-52): 42 autocorrelated weeks. It also lets the
first `h - 1` test weeks of a configuration with horizon `h` use a model trained on data observed
after the forecast would have been issued. `python -m src.ml.backtest` evaluates instead with
rolling origins: models are retrained every 26 weeks from 2018-01, each forecast only uses a model
trained on data available when it is issued, and each configuration is compared with persistence
at the same horizon over the 2018 and 2019 seasons. It reports a skill score
(1 − MSE / MSE of persistence) with a 95% moving-block bootstrap interval, and the error in the
size and timing of each season's peak. The figures no longer show p-values, which assumed
independent weeks.

Preliminary run (`bt-demo`: 3 configurations, 5 repetitions per refit, 4 refits):

| Configuration | h | Season | MAE | Persistence MAE | Skill (95% CI) | Peak size | Peak timing |
|---|---|---|---|---|---|---|---|
| dengue lag 1 | 1 | 2018 | 4.41 | 1.51 | −10.7 (−17.9 to −2.4) | −48% | +11 wk |
| dengue lag 1 | 1 | 2019 | 3.05 | 1.70 | −1.7 (−5.9 to −0.0) | −15% | +2 wk |
| EDI lag 1 | 1 | 2018 | 4.52 | 1.51 | −10.9 (−19.2 to −1.8) | −47% | +11 wk |
| EDI lag 1 | 1 | 2019 | 2.44 | 1.70 | −0.9 (−2.9 to +0.1) | −3% | −1 wk |
| EDI lag 6 | 4 | 2018 | 5.70 | 3.17 | −2.8 (−5.4 to −0.6) | −57% | +14 wk |
| EDI lag 6 | 4 | 2019 | 2.69 | 3.62 | +0.4 (−0.2 to +0.8) | −41% | +0 wk |

No configuration beats persistence with confidence. In 2018 the models trained on 2016–2017 miss
the epidemic: the dengue lag 1 model receives last week's 22 cases and predicts 4, although it was
trained on weeks of up to 41 cases in 2016, which matches the trajectory memorization seen in the
diagnostics. EDI lag 6, the API's default, is the only one with positive skill in a season (2019),
but the interval includes zero and it predicts the peak at 59% of its size. A full run (all 10
configurations, 10 or more repetitions) should confirm these numbers before the API's default
model is chosen.

## Criteria fixed before the full backtest

Written before the full backtest of the paper's version (`paper-full`) was run, so that its
results cannot shape them. "Season" is an epidemiological year (2018 and 2019) as reported by
`python -m src.ml.backtest` with its default settings: origins every 26 weeks from 2018-01, at
least 10 repetitions per refit, each configuration at its own horizon `h`. Skill is always against
persistence at the same horizon.

**For a model to set the API's risk level.** A configuration, or its combination with persistence
(below), qualifies only if, in **both** seasons:

1. the lower bound of the 95% interval of its skill is above zero; and
2. its peak size is within ±25% (the highest forecast week is between 75% and 125% of the highest
   observed week).

The risk level compares the forecast peak with historical percentiles, so a model that beats
persistence on average but forecasts 60% of the peak still gives the wrong level when it matters
most; hence the second condition. Both seasons are required because they test different
situations (2018: models trained on 2016–2017; 2019: models that have seen 2018), and a model that
passes in one only may have been lucky. Until a configuration qualifies, the risk level comes from
the observed data, and LSTM forecasts are shown as experimental.
`python -m src.ml.backtest_combine` checks this criterion for every configuration of a backtest.

**For a change to training to stay in the code.** The change is evaluated with the same backtest
settings and compared with `paper-full`. It stays only if, for EDI lag 6 (the API's default) in
both seasons, its skill is above zero and above the reference's; otherwise the commit is reverted
and the result is recorded here. The results of every configuration evaluated are reported,
including those that got worse. This rule uses point estimates and is looser than the one above
on purpose: keeping a change only decides where the next experiment starts, while setting the risk
level affects what users see.

**Combination with persistence.** The combination evaluated from the saved predictions, without
retraining, is the plain average of the LSTM ensemble mean and persistence (weight 0.5). The
weight is fixed here and is not tuned on 2018–2019; any other weight would have to be chosen on
earlier seasons. The combination is held to the same criterion as the LSTM alone.

**Many comparisons.** Ten configurations, alone and combined, are 20 checks against the
criterion, so one passing by chance is possible even at 95%. A configuration that passes in one
backtest is adopted, but is evaluated again on later seasons (2020 onward) as they are ingested.
