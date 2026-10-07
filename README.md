# Cepheus: Screen Time and sleep analysis

A local, manually run Kedro pipeline linking Apple Screen Time on selected
Mac/iPhone/iPad devices with Garmin's following-morning sleep score.
Includes correlation charts, chronological ridge-versus-average backtesting,
an estimate for tonight, and Kedro-Viz pipeline inspection.

## Install and explore the synthetic demo

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run kedro run --env=demo
uv run kedro viz run --env=demo --host=127.0.0.1
open data/demo/08_reporting/report.html
```

The demo never calls Garmin or reads Apple stores. It uses deterministic,
explicitly synthetic data and a fixed demonstration time of
2026-10-07 20:15 Europe/Rome. Its outputs live under `data/demo`, separately
from personal data under `data/live`. It intentionally contains an artificial
usage/score association; it says nothing about your actual sleep.

The existing console entry point also runs the pipeline:

```sh
uv run cepheus-sleepanalysis-python --env=demo
```

Kedro-Viz defaults to port 4141. It shows registered pipelines, nodes, datasets
and supported previews (including Plotly charts). Opening Viz does not run
ingestion or authenticate Garmin. Do not use Viz's publish/deploy commands for
personal data. Viz itself may check PyPI for package updates.

## Set up Apple Screen Time

The pipeline uses [Screen Time Exporter](https://github.com/nichtlegacy/screentime)
as an independently installed tool. It does not implement its own Apple database
parser. This revision was inspected during integration planning:

```sh
uv tool install 'git+https://github.com/nichtlegacy/screentime@c25696a3ddac32058ae809f63c04e2bfc0c8dc0f'
mkdir -p ~/.config/screentime
```

Before the first collection, put the local-only settings from
[`conf/screentime.example.toml`](conf/screentime.example.toml) into
`~/.config/screentime/config.toml`. If an exporter configuration already exists,
merge these settings rather than overwrite device names or existing data paths.
Disable remote sinks and optional app-name lookups **before** running setup.
Unset any `INFLUX_URL`, `INFLUX_TOKEN`, `HA_URL`, `HA_TOKEN` or
`SCREENTIME_TIMEZONE` overrides.

```sh
screentime setup --no-agent
screentime doctor
screentime status
```

Grant your terminal Full Disk Access in macOS settings and reopen it when
required. Enable Screen Time and **Share Across Devices** on each selected
Apple device signed into the same account. Setup is deliberately manual:
`--no-agent` does not install a scheduled exporter.

Use `screentime status` to obtain stable device IDs. Check that this exporter
revision works with your macOS version. Apple retains only a few weeks of source
history: run collection regularly to archive it locally. Old Garmin scores alone
cannot reconstruct unavailable Screen Time.

Ingestion invokes `screentime run`, checks its JSON status (partial/skipped
collection is an error), then exports all retained sessions with
`screentime dump --format json`. It supplies the configured path through
`SCREENTIME_CONFIG`, checks privacy settings and uses no shell interpolation.
Sessions are archived idempotently by device/start/app; repeated collection
preserves older history and updates revised sessions.

## Set up Garmin

The [garminconnect](https://github.com/cyberjunky/python-garminconnect) package
uses unofficial Garmin Connect endpoints, which can change or be rate limited.
This is not an official Garmin developer-program integration.

```sh
uv run cepheus-garmin-login
# For a Garmin China account:
uv run cepheus-garmin-login --region=china
```

Enter credentials locally in the terminal; passwords and MFA input are hidden.
Tokens are stored outside the repository in `~/.config/cepheus/garmin`.
Later runs reuse them. For another token path, pass `--tokenstore PATH` and
update `study.garmin_tokenstore`. Set `study.garmin_region: china` if applicable.
Never put passwords or tokens in configuration, chat, source code or git.

The pipeline initially requests up to 180 daily sleep records. It reuses cached
older responses and refreshes the latest seven days to pick up sync corrections.
Missing records differ from failed requests; connection/rate-limit failures have
bounded retries and unresolved errors stop the run. Raw per-date responses remain
available for local inspection. The expected score path is
`dailySleepDTO.sleepScores.overall.value`; GMT sleep timestamps are converted
to the study timezone instead of trusting Garmin's local timestamp fields.

## Configure your study

Create ignored `conf/local/parameters.yml`. Parameters merge with base defaults,
so only overrides are necessary:

```yaml
study:
  device_ids: ["mac:YOUR-UUID", "YOUR-PHONE-UUID", "YOUR-IPAD-UUID"]
  device_platforms:
    "mac:YOUR-UUID": mac
    "YOUR-PHONE-UUID": iphone
    "YOUR-IPAD-UUID": ipad
  confirmed_coverage:
    - {device_id: "mac:YOUR-UUID", start: "2026-10-01", end: "2026-10-07"}
    - {device_id: "YOUR-PHONE-UUID", start: "2026-10-01", end: "2026-10-07"}
    - {device_id: "YOUR-IPAD-UUID", start: "2026-10-01", end: "2026-10-07"}
```

Replace all example IDs and dates with your actual selections. Omit devices you
do not want to study. **Coverage ranges are assertions by you**, not inferred
from session absence. Only confirm dates where Screen Time was enabled,
available and synced for that device. Inspect the exported data first and
extend coverage through today only once you have confirmed its availability.
An export can succeed while a remote phone has not synced; freshness of the
Mac collection does not prove remote-device completeness.

Collection provenance is retained in the session archive. A local hashed Garmin
account marker prevents accidental reuse of a data root with another account;
choose a new data root when changing accounts or Garmin region.

With no confirmed coverage, daily usage remains unknown, paired-night analysis
is unavailable, and tonight's forecast is not ready. A confirmed covered day
with no sessions is treated as zero usage; an unknown day is never zero-filled.
Updating coverage ranges does not discard the session archive.

Defaults in [`conf/base/parameters.yml`](conf/base/parameters.yml):

- Timezone `Europe/Rome`, fixed cutoff `20:00`.
- 180-day analysis window, seven-day Garmin refresh window.
- At least 60 eligible paired nights for ridge regression.
- Expanding-window backtest: at least 40 initial training nights and 20
  out-of-sample predictions.
- Fixed ridge penalty of 10; no hyperparameter search.
- Current Screen Time collection must be no more than 45 minutes old.

Change the data location via ignored `conf/local/globals.yml`, not by editing
`study.data_root` directly:

```yaml
data_root: data/my-study
```

It must be a dedicated subdirectory of project `data/`. Run hooks protect the
root with mode 0700 and an environment marker prevents demo/live mixing.
Changes to device selection or timezone require a new data root or deliberate
relocation of the old session archive; other feature/configuration changes
require preparation and training to be rerun.

## Run locally

```sh
# Collect, prepare, analyse, train, forecast and render the report:
uv run kedro run

# Inspect the real pipeline:
uv run kedro viz run --host=127.0.0.1
open data/live/08_reporting/report.html

# Rebuild all downstream outputs from cached raw data, without source access:
uv run kedro run --pipeline=offline
```

Individual pipelines are registered as `ingestion`, `preparation`, `analysis`,
`training`, `forecast`, and `reporting`. Their inputs and outputs are persisted
in the catalog. For example:

```sh
uv run kedro run --pipeline=ingestion
uv run kedro run --pipeline=offline
```

After fresh ingestion, rerun preparation and training before an isolated
forecast. Cached models with different configuration/data are rejected, as is
today's usage computed from an older collection.

### Temporal semantics

Exposure day **D** uses sessions in `[D 00:00, D 20:00)`, and is joined with
sleep scored on morning **D+1**. Tonight's forecast targets tomorrow morning,
not the score Garmin recorded this morning.

Intervals are timezone-aware and clipped using actual elapsed time, including
midnight and DST transitions. Per-device and per-platform usage count the union
of that device/platform's intervals. **Combined minutes** union all selected
devices, so simultaneous phone/laptop use is not double-counted.
**Summed device-minutes** deliberately adds per-device totals and can exceed
combined wall-clock minutes. Full-day combined usage is retained only for
completed covered days; it is not a regression feature.

The main-overnight heuristic requires sleep to end on Garmin's reported date,
start after the preceding day's cutoff and before morning noon, last 2–16
hours, and end before 18:00. It permits after-midnight sleep onset. Missing
scores/timestamps and ambiguous windows are excluded with reasons and warnings.
The heuristic is not a clinical sleep classifier; inspect actual records and
adapt it deliberately if your schedule involves shifts/daytime main sleep.

### Correlation and prediction

Eligible paired nights feed Pearson and Spearman correlations for combined,
evening, summed-device, per-device and per-platform cutoff usage. The report
includes sample counts, date coverage, excluded observations, weekday/weekend
summaries, time-series and scatter charts. Constant or short series are explicitly
unavailable. No independent-sample p-values, causal claims or mined lag searches
are presented.

Ridge features are combined minutes, evening minutes from 18:00 to cutoff, and
weekend indicator. Each backtest fits scaling/regression only on earlier paired
nights; its mean baseline uses earlier valid sleep scores, including scores
without confirmed Screen Time. MAE selects the model and RMSE is also reported.
Backtests use the latest cached source records, so historical corrections are
not an as-of-time reconstruction.

- **Before 20:00:** ingestion and analysis run, but forecast is `not_ready`.
- **No valid sleep history:** `insufficient_history`, no numerical estimate.
- **Fewer than 60 pairs:** with current coverage and valid sleep history, show
  the historical average as `baseline_only`, explicitly not usage-based.
- **At least 60 pairs:** use ridge only if its chronological backtest MAE is
  lower than the historical-average baseline; otherwise keep `baseline_only`.
- **Stale/unconfirmed current data:** `not_ready`, no numerical forecast.

Estimates are bounded to 0–100, with unclipped estimates and clipping flags
retained. Historical error is not a confidence interval or a guarantee. Inputs
outside the training range produce warnings. Forecasts are archived separately
from updated training data and compared with actual scores once available.

## Outputs and privacy

See [`conf/base/catalog.yml`](conf/base/catalog.yml) for all persistent outputs:

| Directory | Contents |
|---|---|
| `01_raw` | Exported sessions, historical session archive, dated Garmin responses |
| `02_intermediate` | Normalized sessions and sleep records |
| `04_feature` | Covered daily/cutoff usage and provenance |
| `05_model_input` | Exposure-day / next-morning aligned series |
| `06_models` | Local preprocessing/model artifact and metadata |
| `07_model_output` | Backtest metrics, predictions, tonight's forecast, forecast history |
| `08_reporting` | Coverage, correlations, Plotly charts and self-contained HTML report |

Data, models, reports, local configuration and tokens are excluded from git.
Kedro telemetry is disabled in `.telemetry` and in package settings.
Raw requests contain personal health data; keep the data root private.
Model artifacts use pickle: load only artifacts generated by this trusted local
project, never downloads from untrusted sources. The HTML report embeds Plotly
locally instead of using a CDN. Only the explicitly configured Garmin ingestion
contacts Garmin; Screen Time remote sinks/lookups are rejected.

This implementation has not had tests written or executed, nor live integrations
or a pipeline run assessed. Dependency resolution was performed; functional
assessment, Apple permissions and Garmin authentication are left to you.