# Uncertainty-Aware Energy Intelligence

An end-to-end 15-minute IoT energy pipeline that repairs corrupted sensor readings, forecasts the next 24 hours with calibrated uncertainty, and estimates savings from shifting flexible demand.

## Why this is a strong AI/ML project

It goes beyond “train an LSTM and report RMSE” by showing:

- 15-minute IoT-style readings with simulated missing, flat-line, repeated-random, and spike failures;
- weather enrichment from Open-Meteo's historical archive API;
- rolling-window, robust-z-score, and repeated-value quality rules;
- an LSTM reconstruction model for anomaly scoring;
- linear interpolation and forward-fill baselines versus an LSTM imputer;
- a 96-step Bayesian LSTM forecast with Monte Carlo Dropout and conformal interval widening;
- interval coverage, imputation error, detection precision/recall, and an energy-cost scenario.

The Bayesian method is **MC Dropout**. Dropout remains active during inference, and repeated stochastic forward passes approximate samples from the posterior predictive distribution. It is practical and explainable for a portfolio project, but it is not exact Bayesian inference.

## Dataset and simulation

The code downloads the public [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). Its minute-level active-power readings are aggregated to 15-minute demand. The pipeline deliberately injects known corruptions into a clean contiguous window so that detection and restoration can be evaluated against ground truth without corporate data.

Historical temperature, humidity, precipitation, and shortwave radiation are fetched from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) for the IIT Kanpur coordinates. `--offline-weather` uses a deterministic weather generator when internet access is unavailable.

## Run locally

```bash
cd bayesian-lstm-energy-forecast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py --max-days 60 --epochs 3 --mc-samples 50
```

For a fast local smoke test:

```bash
python pipeline.py --max-days 10 --epochs 1 --mc-samples 10 --offline-weather
```

Outputs are written to `artifacts/pipeline/`:

- `energy_quality_results.csv`: clean, corrupted, detected, and repaired readings;
- `forecast_intervals.csv`: actual values, forecast mean, and calibrated interval;
- `metrics.json`: detection, imputation, forecasting, and optimization metrics;
- `pipeline_overview.png`: anomaly repair and probabilistic forecast visualization;
- `optimization_summary.json`: baseline cost, optimized cost, and annualized savings scenario;
- `bayesian_forecaster.pt`: saved Bayesian LSTM weights.

The original point-forecast-only experiment remains available as `train.py`.

## Pipeline flow

1. Download minute-level active power and aggregate it to 15-minute `kW` readings.
2. Join weather data and create cyclic calendar, rolling mean/std, and lag features.
3. Inject labelled missing, flat-line, repeated-random, and spike failures for evaluation.
4. Detect failures using robust rolling rules and an LSTM reconstruction error threshold.
5. Restore flagged intervals with linear interpolation, forward-fill, and a masked LSTM imputer.
6. Train a forecasting LSTM on repaired energy, weather, calendar, rolling, and lag features.
7. Forecast 96 future 15-minute intervals using stochastic dropout passes.
8. Calibrate the 95% interval using held-out validation nonconformity.
9. Simulate shifting 20% flexible peak demand to off-peak periods using an explicit time-of-use tariff.

## Interview discussion points

- **Why Bayesian?** A point forecast can be overconfident. Predictive intervals communicate risk and support decisions such as reserve capacity or human review.
- **Why MC Dropout?** It is a low-cost approximate Bayesian technique that reuses a standard neural network and does not require a separate probabilistic inference engine.
- **Why inject anomalies?** Public datasets rarely contain reliable labels for every sensor failure. Synthetic, labelled corruption lets the pipeline report precision, recall, F1, and restoration error honestly.
- **Why repair before forecasting?** Sensor corruption becomes a false signal for the forecasting model. Restoration makes the downstream forecast less sensitive to bad telemetry.
- **What is calibrated?** A split-conformal correction uses held-out validation residuals to widen the raw MC-Dropout interval. Coverage and interval width should be reported together.
- **What is the limitation?** MC Dropout mainly captures model/epistemic uncertainty. It can underrepresent observation noise, regime changes, and uncertainty from future covariates.
- **How would you improve it?** Add real tariff/load metadata, compare against TCN/DeepAR/TFT baselines, evaluate rolling-origin backtests, and validate with real device outages.

## Resume-ready version

**Uncertainty-Aware Energy Intelligence | LSTM Imputation + Bayesian Forecasting**

- Built an end-to-end PyTorch pipeline for 15-minute energy telemetry, integrating Open-Meteo temperature, humidity, precipitation, and solar radiation with rolling statistics, lag features, and calendar covariates.
- Detected labelled missing, flat-line, repeated-random, and spike failures using robust rolling rules plus LSTM reconstruction error; compared linear, forward-fill, and masked-LSTM restoration using anomaly-only MAE/RMSE.
- Forecast 96 future intervals with a Monte Carlo-Dropout LSTM and validation-calibrated 95% intervals; evaluated coverage/width and simulated annual savings from shifting flexible peak demand to off-peak periods.

