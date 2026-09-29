# Industrial Maintenance Effectiveness Analyzer

A complete local React + FastAPI student application for observational maintenance analysis. The supplied CSV remains untouched in Downloads; `backend/data/supplied.csv` is a working copy. Its synthetic origin is shown in the application.

## Run on Windows PowerShell

From `D:\DS hackathon 2`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

In a second terminal:

```powershell
cd 'D:\DS hackathon 2\frontend'
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Select **Open supplied dataset**, then use **Train model** on the Decision Tree Model page. New CSV uploads require dataset setup. Local project history is excluded from Git; a reference trained pipeline and evaluation are included in `artifacts/`. API documentation: http://127.0.0.1:8001/docs.

Install dependencies using the commands above after cloning. No database, authentication, paid service, or LLM is required. Run one backend worker: training uses in-process background tasks and a lock. This local student application is not designed for public multi-user hosting. If the process exits during training, use dataset setup to invalidate the interrupted training state and train again.

## Dataset meaning

The user confirmed one row per machine-day, daily outcome totals, and `Maintenance_Performed = 1` as an actual maintenance event. Dates use YYYY-MM-DD. The bundled dataset has 10,950 records, 30 machines and five categories. Maintenance types are read from the data; mixed or unknown types must not be described as verified preventive-only effectiveness.

`Maintenance_Date` is never used as an observation date. The model uses `Observation_Date`. New datasets can specify daily, fixed-period, event, or unconfirmed semantics. Fixed-period rows require an actual observation date representing the period end and a confirmed constant period length. Before/after comparison currently requires daily observations; explicit interval rows must be disaggregated upstream, without inventing exposure. Event rows can support explanatory regression, but not equal-duration category comparisons.

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for units and assumptions. A project-specific dictionary is downloadable from Upload & Data Quality.

## Analysis and model methods

- Missing outcomes remain missing, exact duplicates are reported then removed, invalid outcomes become missing, and outliers are retained. Missing identifiers or inconsistent categories exclude rows. Ambiguous dates are resolved with an explicit selected date format. The original upload is preserved.
- Four chart sections: paired before/after downtime (or descriptive fallback), category downtime, event frequency versus downtime, and priority overview. The first three appear on Maintenance Analysis; the fourth is on Machine Priorities. The HTML report contains all four and embedded SVGs without external assets.
- Default paired windows are 30 calendar days before/after, excluding event day. Both require at least 80% valid daily coverage and a valid-count ratio of at least 90%. Overlapping event windows are all excluded, preventing observation reuse. Means use observed days; missing days are never zero. Percentage reduction is undefined if the before mean is zero.
- Events are deduplicated by machine/date/ID when IDs are complete; otherwise by machine/date. Frequency uses a common calendar horizon, with observed-day coverage reported separately. Counts do not imply complete event logging.
- Category averages weight machines equally. Priority rules use the category upper quartile, at least ten valid outcome records and four comparable machines. High maintenance and poor outcomes alone do not prove excessive maintenance. Effectiveness-review flags additionally require supported paired evidence of <=5% improvement. Recent-trend and maintenance-gap rules are not implemented; the transparent supported rules are displayed.
- Forecasts predict the next confirmed observation period, only across contiguous periods within a machine. Inputs are prior downtime, production, failures, category, historical event count and days since last event. No machine ID or target-period outcomes are predictors. Evaluation is a chronological holdout with identical target dates grouped together. This is rolling one-step evaluation: completed previous test-period measurements are available for subsequent one-step predictions.
- If chronological data is unavailable, confirmed semantics and at least five machines permit contemporaneous explanatory regression using category, same-period production and failures. Grouped holdout separates machines. This is not forecasting.
- A training-fitted Pipeline handles median numeric imputation and categorical encoding. The tree has maximum depth four and a minimum leaf support of max(5, 1% of training records). Complexity is fixed; no held-out tuning. Evaluate MAE, RMSE and R², training/test counts and errors, mean baseline, and previous-period baseline in forecast mode. Permutation importance is measured on held-out rows. Downloadable tree text and leaf sample counts are included in evaluation JSON.
- Filters apply to descriptive analysis, priorities, insights and report exports. Cleaned CSV, data dictionary, evaluation and model downloads represent the entire project. Model training is always on the full dataset and is labeled accordingly. A metadata version invalidates models when interpretation or events change.

## APIs and storage

`backend/data/projects/<uuid>/` stores immutable original CSV, cleaned CSV, metadata, quality report, latest analysis, evaluation and the full joblib pipeline. User-uploaded models are never accepted. Upload limit: 20 MB. Project UUID validation isolates paths; uploaded filenames are never used as storage paths. CORS is configured through `CORS_ORIGINS`; `DATA_DIR` can override the project store. `.env.example` documents values; set them as environment variables (no automatic dotenv loading).

| Endpoint | Purpose |
| --- | --- |
| GET/POST `/api/projects` | List projects / upload data |
| POST `/api/projects/supplied` | Create project from supplied working copy |
| GET `/api/projects/{id}` | Project metadata |
| PUT `/api/projects/{id}/setup` | Save semantics and invalidate model |
| POST `/api/projects/{id}/events` | Separate event CSV |
| GET `/api/projects/{id}/quality` | Validation and cleaned preview |
| POST `/api/projects/{id}/clean` | Re-run deterministic cleaning |
| GET `/api/projects/{id}/analysis` | Filtered descriptive results |
| GET `/api/projects/{id}/before-after` | Paired comparisons |
| GET `/api/projects/{id}/frequency` | Event counts/rates with support |
| GET `/api/projects/{id}/priorities` | Rule-based priority flags |
| POST `/api/projects/{id}/train` | Background training |
| GET `/api/projects/{id}/training-status` | Training status |
| GET `/api/projects/{id}/evaluation` | Evaluation and interpretation |
| POST `/api/projects/{id}/predict` | Compatible-model estimate |
| GET `/api/projects/{id}/insights` | Deterministic evidence-based findings |
| GET `/api/projects/{id}/download/{kind}` | cleaned, priorities, evaluation, pipeline, dictionary, report |

Analysis endpoints accept `machine`, `category`, `start`, `end` query filters. The combined analysis response includes the action plan. OpenAPI documents schemas and error responses.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm run build
```

Tests cover uploads, missing columns, invalid dates, non-imputed outcomes, repeated events, filtering, downloads, coverage, overlap, zero denominators, future-event exclusion, gaps, time/group holdout, unseen categories, pipeline reload and stale model rejection.

Eight backend tests pass. The production frontend build passes. With both servers running and Microsoft Edge installed, run the browser smoke test from `frontend` using `node smoke.mjs`. It exercises all seven pages, charts, prediction, a report download, and mobile overflow detection. Screenshots and a standalone report are saved in `artifacts/`. `backend/requirements-lock.txt` records the verified Python environment; `frontend/package-lock.json` pins the frontend installation.

The supplied-data run produced 181 confirmed events, 96 valid paired comparisons, 75 overlap exclusions and 10 coverage exclusions. Each of the three outcomes has 55 missing values. Forecast evaluation: 8,684 training samples and 2,182 test samples; test MAE 0.7255 h, RMSE 0.9432 h, R² 0.5914. Mean baseline MAE 1.1306 h; previous-period baseline MAE 0.7754 h. These are synthetic-data results, not evidence of real-factory effectiveness.
