# Pearls AQI Predictor — Project Report

**Project**: Serverless 3-day AQI forecasting system for Karachi, Pakistan
**Author**: Laiba Minhaj
**Live Dashboard**: https://pearls-aqi-predictor-khi.streamlit.app/
**Repository**: https://github.com/LaibaMinhaj/pearls-aqi-predictor

---

## 1. Project Overview

This project builds an end-to-end machine learning pipeline that predicts Karachi's Air Quality Index (AQI) — specifically, the **average AQI for each of the next 3 calendar days** — using a 100% serverless, cloud-based stack. The system covers the full ML lifecycle: automated data collection, feature engineering, model training and comparison, a cloud feature store and model registry, a live dashboard with explainability, and full CI/CD automation via GitHub Actions.

---

## 2. Data Source Selection

### APIs evaluated
Three candidate APIs were evaluated for historical and live AQI/weather data:

| API | Verdict | Reason |
|---|---|---|
| **AQICN** | Rejected for historical data | Free API only exposes a rolling 7-day history graph; full historical backfill requires separate Historical Data Platform registration (institutional access, not self-serve). Also, ground stations update on a multi-hour cycle, risking duplicate/flat readings if used for live features. |
| **OpenWeather** | Rejected | Historical Air Pollution API returned `401 Unauthorized` on the free tier despite documentation suggesting free-tier availability — likely a plan/subscription mismatch discovered during testing. |
| **Open-Meteo** | **Selected** | No API key required, no signup, genuinely free with no billing risk. Provides both historical (`archive-api`, `air-quality-api`, back to Jan 2023 for this location) and live/forecast data (`forecast` endpoint with `past_days`/`forecast_days` parameters) from a single, consistent, modeled (CAMS-based reanalysis) source. |

**Key decision**: Open-Meteo was used for **both historical backfill and live inference**, avoiding any inconsistency between training-time and inference-time data sources. This also sidesteps the AQICN ground-station staleness problem entirely, since no raw station feed is ever used.

**Known tradeoff**: Because Open-Meteo's AQI is a modeled estimate rather than a physical sensor reading, live "current AQI" values can diverge from consumer apps like AccuWeather that source from ground stations, particularly during rapidly changing conditions. This was documented as a deliberate reliability-vs-precision tradeoff rather than treated as a bug.

---

## 3. Data Collection and Cleaning

- **Historical backfill**: ~3.5 years of hourly data (January 2023 – present) for Karachi (24.8607°N, 67.0011°E), combining pollutant data (PM2.5, PM10, CO, NO2, SO2, O3, US AQI) and weather data (temperature, humidity, pressure, wind speed/direction, precipitation).
- **Chunked fetching**: Requests were split into ~90-day chunks with retry logic to handle API limits and transient failures, later corrected to avoid re-requesting boundary dates twice (off-by-one fix).
- **Validation performed** before proceeding to modeling:
  - Zero duplicate timestamps
  - Zero gaps in the hourly sequence (verified against a full expected date range)
  - Sensible value ranges (e.g., PM2.5 mean ~29 with a max of 339 during pollution spikes; temperature 8.9–41.6°C, consistent with Karachi's climate)
- **Result**: 31,392 clean hourly rows with zero missing values, confirmed via EDA before any feature engineering began.

---

## 4. Exploratory Data Analysis

Key findings from EDA on the raw historical data:
- `us_aqi` correlates most strongly with PM2.5 (r=0.73), followed by SO2, CO, and NO2
- Temperature, humidity, and wind speed correlate **negatively** with AQI — consistent with atmospheric dispersion physics (wind/heat disperse pollutants; cool, still, humid conditions trap them)
- AQI distribution: mean ~90, median 82, max 297 — Karachi spends most of its time in the "Moderate" to "Unhealthy for Sensitive Groups" range

These patterns matching known atmospheric science gave confidence the dataset was trustworthy before any modeling began.

---

## 5. Target Definition — A Critical Correction

The initial implementation defined targets as **point-in-time** AQI at exactly t+24h, t+48h, t+72h. Partway through the project, clarification from the mentor ("predict the average AQI per day") revealed the intended target was the **calendar-day average AQI** for Day 1, Day 2, and Day 3 — not a single-hour snapshot.

This was corrected by redefining targets as:

target_day1 = mean(AQI) over tomorrow's 24 hours
target_day2 = mean(AQI) over the day after tomorrow's 24 hours
target_day3 = mean(AQI) over three days from now's 24 hours


**Impact of this correction** (Ridge model, before further tuning):
| Horizon | Point-in-time RMSE | Daily-average RMSE | Point-in-time R² | Daily-average R² |
|---|---|---|---|---|
| Day 1 | 11.22 | 9.61 | 0.547 | 0.600 |
| Day 2 | 14.11 | 12.77 | 0.274 | 0.283 |
| Day 3 | 14.90 | 13.41 | 0.187 | 0.204 |

Averaging smoothed out hourly noise, meaningfully improving both error and explained variance at every horizon.

---

## 6. Feature Engineering

### Leakage prevention (core principle throughout)
- **Lag features** (1h, 3h, 6h, 12h, 24h, 48h, 72h) for AQI, pollutants, and weather use only past values (`.shift()`), never same-timestamp or future data.
- **Future weather features** use forecasted values at inference time; during training, actual historical observations at t+24h/48h/72h serve as a proxy (real historical forecasts don't exist). This is a documented, deliberate approximation.
- **Same-timestamp future pollutant values are never used as features** — only current/past pollutant readings and legitimately forecastable weather.

### Feature set evolution
**V1 (76 features)**: calendar features (hour, day of week, month, weekend flag), lag features, rolling 24h/72h means, AQI change rate, current weather/pollutants, future weather proxy.

**V2 (92 features)**, added after initial model comparison:
- Cyclical encoding: `sin`/`cos` of hour and month (captures that hour 23 and hour 0 are adjacent, which raw integers cannot)
- Rolling volatility features: 24h rolling standard deviation, min, and max for AQI and PM2.5 (captures recent *variability*, not just level)

**Impact of V2 features** (ensemble model):
| Horizon | V1 R² | V2 R² |
|---|---|---|
| Day 2 | 0.4901 | 0.5254 |
| Day 3 | 0.3703 | 0.4294 |

---

## 7. Modeling — Experiments and Results

### Model families compared
Seven model families were systematically evaluated using a chronological 70/15/15 train/validation/test split (never randomly shuffled, to avoid time-series leakage):

1. Ridge Regression
2. Random Forest
3. HistGradientBoostingRegressor
4. LightGBM (with early stopping)
5. XGBoost (with early stopping)
6. CatBoost (with early stopping)
7. SVR (with StandardScaler)

### Key experimental findings

**Feature-set ablation (Ridge, chronological order tested)**:
| Experiment | Description | 24h RMSE | Finding |
|---|---|---|---|
| Baseline | Lag features only | 11.89 | — |
| A2 | + current weather | 11.88 | Negligible improvement (lags already captured it) |
| B | + current pollutants | 11.43 | Meaningful improvement at 24h, minimal at 48h/72h (pollutant persistence decays) |
| C | + future weather proxy | 11.22 | Largest improvement, especially at longer horizons |

**Model family comparison** (daily-average targets, validation set): Ridge consistently outperformed Random Forest, HistGradientBoosting, and LightGBM across every horizon in initial comparisons, suggesting a predominantly linear relationship in the base feature set.

**Rejected techniques** (tested, documented, not adopted):
- **Recency-weighted training** (exponential decay favoring recent data): hurt Day 2/Day 3 performance — Karachi's AQI has seasonal patterns that recency weighting inadvertently discounts.
- **Recursive horizon-chaining** (using Day 1's out-of-fold prediction as an input feature for Day 2/Day 3): did not help — existing lag/rolling features already captured the relevant momentum signal, making the chained feature redundant.
- **Training-window regime** (recent-only 6-month window vs. full 3.5-year history): full history won decisively at every horizon (e.g., Day 3 R² 0.424 vs 0.207 for recent-only), confirming the value of seasonal coverage for longer-horizon forecasting.

**Ensembling**: Averaging predictions from Ridge, SVR, and CatBoost outperformed any single model at Day 1 and Day 3. Notably, using *individually hyperparameter-tuned* versions of each model in the ensemble slightly **underperformed** a default-parameter ensemble at Day 2 — tuning each model toward its own optimum reduced the prediction diversity that makes ensembling effective. Both configurations were kept, using whichever performed best per horizon.

### Final selected configuration
| Horizon | Approach | Ridge alpha | SVR (C, epsilon) | CatBoost (depth, lr) |
|---|---|---|---|---|
| Day 1 | Tuned ensemble | 10000 | 1.0, 0.1 | 6, 0.03 |
| Day 2 | Default-param ensemble | 10000 | 10.0, 0.5 | 6, 0.03 |
| Day 3 | Tuned ensemble | 1000 | 1.0, 0.1 | 4, 0.01 |

### Final performance (true held-out test set)
| Horizon | RMSE | MAE | R² |
|---|---|---|---|
| Day 1 | 9.17 | 6.46 | 0.636 |
| Day 2 | 12.44 | 9.16 | 0.320 |
| Day 3 | 13.11 | 9.65 | 0.239 |

**Note on validation vs. test discrepancy**: Validation-set R² for this configuration was notably higher (0.7482 / 0.5254 / 0.4319). Investigation confirmed this gap is driven by differing variance between the validation and test time periods — R² is inherently sensitive to the variance of the evaluation window, while RMSE (an absolute error measure) remained consistent and even improved slightly on the test set relative to earlier configurations. This distinction was verified directly by decomposing R² into its RMSE and variance components.

### Persistence baseline comparison
A naive persistence baseline (tomorrow's AQI = today's 24h average) was computed on the same final test set to confirm the model provides genuine value beyond simple autocorrelation:

| Horizon | Persistence RMSE | Persistence R² | Ensemble RMSE | Improvement |
|---|---|---|---|---|
| Day 1 | 14.30 | 0.11 | 9.17 | 36% lower RMSE |
| Day 2 | 16.60 | -0.21 | 12.44 | 25% lower RMSE |
| Day 3 | 18.10 | -0.45 | 13.11 | 28% lower RMSE |

Persistence's R² turns **negative** at Day 2/Day 3 on this dataset — meaning naive "tomorrow=today" performs worse than simply predicting the average. The ensemble model decisively and consistently outperforms this baseline at every horizon.

### Error analysis
The largest test-set prediction errors were examined directly. They clustered heavily around a single event in mid-March, where actual AQI spiked to 122–134 but the model predicted only 80–100. Weather conditions preceding this spike were hot (30–32°C), very dry (19–36% humidity), and lightly-to-moderately windy — a known meteorological precursor to dust-suspension events in Karachi. This represents an inherent limitation of lag-based forecasting: features built from past pollution levels cannot anticipate the *onset* of a sudden pollution event, only track its continuation once underway. This is documented as a genuine model limitation rather than a bug, and a candidate for future work (e.g., a dedicated "heat-dryness index" feature).

---

## 8. Feature Store and Model Registry (Hopsworks)

- **Feature Store**: Two feature groups — `karachi_aqi_features` (v1: 76 cols, v2: 92 cols after the richer-feature addition) and `karachi_aqi_targets` (calendar-day average targets) — both fully populated and used as the sole source of truth for training (no local CSV dependency in the final production pipeline).
- **Model Registry**: All 9 production model components (Ridge, SVR+scaler, CatBoost, × 3 horizons) are registered with attached performance metrics (RMSE, MAE, R²), enabling the dashboard to display live, registry-sourced accuracy rather than hardcoded values.

---

## 9. Dashboard and Explainability

- Built with **Streamlit**, showing current AQI, 3-day forecast cards with confidence ranges (± RMSE), a trend chart, and hazardous-AQI alert banners.
- **SHAP explanations** ("Why these predictions?") show the top features increasing/decreasing each day's forecast. Since the production model is an ensemble of three different algorithm types, SHAP is computed via the ensemble's Ridge component specifically (the one genuinely interpretable member) — this scope limitation is disclosed directly in the UI.

---

## 10. Automation (CI/CD)

- **Hourly** (GitHub Actions): fetches a rolling data window, computes features only for newly-complete rows (tracked via a committed state file), and incrementally upserts to the Feature Store — avoiding redundant reprocessing of unchanged data.
- **Daily** (GitHub Actions): retrains the full ensemble on the latest data and registers new model versions with fresh metrics.
- Both workflows include automatic retry logic (3 attempts, 30s apart) to handle transient Hopsworks connectivity issues.

---

## 11. Blockers Encountered and How They Were Fixed

| Blocker | Fix |
|---|---|
| OpenWeather historical API returned 401 | Switched to Open-Meteo entirely (historical + live) |
| Windows-specific Hopsworks client errors (`/tmp` path hardcoding, missing `pyarrow`/`confluent-kafka`) | Installed missing packages; manually created `C:\tmp` directory |
| Hopsworks defaulted to unavailable `DELTA` time-travel format | Explicitly set `time_travel_format="HUDI"` |
| Schema mismatches (target columns leaking into feature push; int32 vs int64 dtype conflicts; new v2 columns rejected by locked v1 schema) | Corrected exclusion lists; explicit `.astype("int64")` casts; created a new feature group version (v2) for schema changes |
| Exposed Hopsworks API key committed to a public README | Revoked and rotated the key immediately; scrubbed the file; treated as a priority security fix before continuing feature work |
| Hopsworks free-tier billing hit its monthly compute budget, freezing the account | Diagnosed root cause (hourly pipeline was reprocessing the full data window on every run); redesigned the pipeline to track state and only process/push genuinely new rows, cutting compute cost roughly 100x per run |
| GitHub Actions workflow silently did nothing (missing `if __name__ == "__main__":` call, then an indentation bug in that block) | Traced via raw workflow logs (not the collapsed UI view) to find the script was defining functions but never executing `main()`; fixed indentation |
| Transient Hopsworks connection drops (`Socket closed`, `Flight unavailable`) and a backend database error during model upload | Identified as Hopsworks-side infrastructure issues via server error messages; added retry logic (3 attempts) to both automation workflows |
| Streamlit Cloud deployment failures: protobuf version conflict between `streamlit` and `hopsworks`; Python 3.14 default incompatible with protobuf's C extension; missing `catboost` dependency; stale cached model downloads after retraining | Unpinned conflicting package versions; explicitly pinned Python 3.12 in Streamlit Cloud settings; added `catboost` to `requirements.txt`; manually rebooted the app container to clear stale caches |
| Local `.env` API key changes not taking effect | Diagnosed as a stale running process holding an old environment variable in memory; fixed by fully restarting the process and adding `load_dotenv(override=True)` |

---

## 12. Conclusion

This project delivers a fully automated, cloud-native AQI forecasting system that decisively outperforms a naive persistence baseline at every horizon, is backed by a systematic 7-model comparison with documented negative results, and includes honest analysis of its own limitations (validation/test variance sensitivity, sudden-event blind spots). Beyond the modeling work, the project involved diagnosing and resolving several genuine production engineering issues — a security incident, a billing/cost-control issue, and multiple infrastructure/deployment failures — each investigated to root cause and fixed rather than worked around.