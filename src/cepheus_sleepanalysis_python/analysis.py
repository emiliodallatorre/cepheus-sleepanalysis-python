import html
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress, spearmanr

from .common import Parameters

logger = logging.getLogger(__name__)


def correlate(aligned: pd.DataFrame) -> pd.DataFrame:
    paired = aligned.loc[aligned["eligible"]]
    features = [
        column for column in paired.columns
        if column.endswith("minutes")
    ]
    rows = []
    for feature in features:
        source = (
            aligned.loc[aligned["full_day_eligible"]]
            if feature.startswith("full_day") else paired
        )
        valid = source[[feature, "sleep_score"]].dropna()
        reason = (
            "insufficient_pairs" if len(valid) < 3
            else "constant_series" if valid.nunique().min() < 2 else "available"
        )
        pearson, spearman = None, None
        if reason == "available":
            pearson = float(np.corrcoef(valid[feature], valid["sleep_score"])[0, 1])
            spearman = float(spearmanr(valid[feature], valid["sleep_score"]).statistic)
        else:
            logger.warning("Correlation %s unavailable: %s", feature, reason)
        rows.append({
            "feature": feature, "pairs": len(valid), "pearson": pearson,
            "spearman": spearman, "status": reason,
        })
    return pd.DataFrame(rows, columns=["feature", "pairs", "pearson", "spearman", "status"])


def charts(
    aligned: pd.DataFrame, p: Parameters
) -> tuple[go.Figure, go.Figure]:
    label = "SYNTHETIC DEMO - " if p["demo"] else ""
    available = aligned.loc[
        aligned["usage_available"] & aligned["full_day_summed_device_minutes"].notna()
    ].sort_values("exposure_date")
    series = make_subplots(rows=2, cols=1, shared_xaxes=True)
    for column in aligned.columns:
        if column.startswith("full_day") and column.endswith("minutes"):
            series.add_trace(go.Scatter(
                x=available["exposure_date"], y=available[column],
                name=column, mode="lines",
            ), row=1, col=1)
    series.add_trace(go.Scatter(
        x=available["exposure_date"], y=available["sleep_score"],
        name="Following morning's sleep score", mode="lines+markers",
    ), row=2, col=1)
    series.update_yaxes(title_text="Full-day minutes (00:00-24:00)", row=1, col=1)
    series.update_yaxes(title_text="Sleep score", range=[0, 100], row=2, col=1)
    series.update_layout(
        title=f"{label}Exposure day D and sleep on morning D+1",
        height=650, template="plotly_white",
    )
    scatter = go.Figure()
    paired = available.loc[available["full_day_eligible"]]
    scatter.add_trace(go.Scatter(
        x=paired["full_day_summed_device_minutes"] / 60,
        y=paired["sleep_score"],
        customdata=paired[["exposure_date", "target_sleep_date"]].to_numpy(),
        name="Paired observations", mode="markers",
        hovertemplate="Usage day %{customdata[0]}<br>Sleep morning %{customdata[1]}"
        "<br>%{x:.2f} device-hours<br>Sleep score %{y}<extra></extra>",
    ))
    scatter.update_layout(
        title=f"{label}Screen Time vs sleep score ({len(paired)} paired nights)",
        xaxis_title="Screen Time 00:00-24:00 (summed device-hours)",
        yaxis_title="Following morning's sleep score",
        yaxis={"range": [0, 100]}, template="plotly_white", showlegend=True,
    )
    x = paired["full_day_summed_device_minutes"] / 60
    y = paired["sleep_score"]
    if len(paired) >= 3 and x.nunique() > 1 and y.nunique() > 1:
        fit = linregress(x, y)
        line_x = np.array([float(x.min()), float(x.max())])
        scatter.add_trace(go.Scatter(
            x=line_x, y=fit.intercept + fit.slope * line_x,
            mode="lines", name="Linear trend (descriptive)",
            line={"color": "#d35400", "width": 3},
            hovertemplate="%{x:.2f} device-hours<br>Fitted score %{y:.2f}<extra>Linear trend</extra>",
        ))
        summary = (
            f"Pearson r = {fit.rvalue:.3f} | R² = {fit.rvalue ** 2:.3f}"
            f"<br>Slope = {fit.slope:.2f} score points/device-hour"
            f" | n = {len(paired)}<br>In-sample association, not causation or forecast accuracy"
        )
    else:
        summary = "Linear trend unavailable: requires at least 3 pairs and nonconstant axes."
        logger.warning(summary)
    scatter.add_annotation(
        text=summary, xref="paper", yref="paper", x=0, y=1.02,
        xanchor="left", yanchor="bottom", align="left", showarrow=False,
    )
    scatter.update_layout(margin={"t": 160})
    if available.empty:
        series.add_annotation(
            text="No available Screen Time totals. Prepare usage data first.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
    if paired.empty:
        scatter.add_annotation(
            text="No eligible nights with both Screen Time and a sleep score.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
    return series, scatter


def report(
    aligned: pd.DataFrame, quality: dict[str, Any], correlations: pd.DataFrame,
    series: go.Figure, scatter: go.Figure, metrics: dict[str, Any],
    forecast: dict[str, Any], backtest: pd.DataFrame, history: pd.DataFrame,
    p: Parameters,
) -> str:
    paired = aligned.loc[aligned["full_day_eligible"]]
    groups = paired.groupby("weekend").agg(
        paired_nights=("sleep_score", "count"),
        mean_sleep_score=("sleep_score", "mean"),
        mean_full_day_device_minutes=("full_day_summed_device_minutes", "mean"),
    ).rename(index={0: "Weekday", 1: "Weekend"})
    label = "SYNTHETIC DEMO - " if p["demo"] else ""
    comparison = paired[[
        "exposure_date", "target_sleep_date", "full_day_summed_device_minutes", "sleep_score"
    ]].sort_values(["full_day_summed_device_minutes", "exposure_date"]).copy()
    comparison["full_day_summed_device_minutes"] = comparison["full_day_summed_device_minutes"] / 60
    comparison = comparison.rename(columns={
        "exposure_date": "Usage day",
        "target_sleep_date": "Sleep morning",
        "full_day_summed_device_minutes": "Screen Time 00:00-24:00 (device-hours)",
        "sleep_score": "Sleep score",
    })

    def block(value: dict[str, Any]) -> str:
        return "<pre>" + html.escape(json.dumps(value, indent=2, allow_nan=False)) + "</pre>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{label}Cepheus sleep analysis</title>
<style>
body {{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17212b}}
table {{border-collapse:collapse;font-size:14px;display:block;overflow:auto}}
th,td {{border:1px solid #ddd;padding:.5rem;text-align:right}}
pre {{background:#f3f5f7;padding:1rem;overflow:auto}}
.warning {{background:#fff2d1;padding:1rem;border-left:4px solid #c88b00}}
</style></head><body>
<h1>{label}Screen Time and sleep</h1>
<p>Timezone: {html.escape(p["timezone"])}. Plots use full-day usage, 00:00-24:00.
Forecast features still use midnight to {html.escape(p["cutoff"])}.
Usage on day D is paired with the sleep score on morning D+1.</p>
<p class="warning">Observational association is not causation or medical advice.
Garmin's score is a vendor-derived metric. Exercise, stress, caffeine, alcohol,
illness, travel and sleep habits can confound the relationship. Serial dependence
means ordinary independent-sample significance claims are inappropriate.
Device-minutes can overlap; the primary analysis and predictor sum device usage,
counting simultaneous devices twice. Combined minutes also show their interval union.
With observed_sum coverage, missing device records count as zero observed minutes,
not proven inactivity; incomplete sync can bias both correlation and forecasts.</p>
<h2>Tonight's estimate</h2>{block(forecast)}
<p>No numerical confidence interval is claimed. Backtest MAE describes historical
error, not a guarantee for tonight. Before cutoff or with unavailable/stale
inputs, no forecast is issued.</p>
<h2>Coverage and exclusions</h2>{block(quality)}
<h2>Screen Time and sleep over time</h2>
<p>Only dates with available Screen Time totals are plotted, including zero
observed usage. Earlier Garmin-only dates and unfinished calendar days are omitted.</p>
{series.to_html(full_html=False, include_plotlyjs=True)}
<h2>Screen Time vs sleep score</h2>
<p>Each point pairs summed Screen Time from 00:00 through 24:00 on day D with the sleep
score on morning D+1. Simultaneous devices count separately. Only eligible
completed days are shown. This is a retrospective calendar-day comparison:
usage after sleep onset can be included, so it is not strictly pre-sleep exposure
and cannot be used for a forecast made at 20:00.</p>
<p>The descriptive least-squares line summarizes this sample. Pearson r gives
the direction and strength of linear association; R² is the fraction of
in-sample score variation explained by that line. Neither is evidence of causation
or held-out forecast accuracy. The paired table is sorted by Screen Time.</p>
{scatter.to_html(full_html=False, include_plotlyjs=False)}
<h3>Paired values</h3>
{comparison.to_html(index=False, escape=True, float_format=lambda value: f"{value:.2f}")}
<h2>Correlation on eligible paired nights</h2>
{correlations.to_html(index=False, escape=True, na_rep="Unavailable")}
<p>No p-values or causal effect estimates are presented. The primary comparison
is full-day usage versus next-morning score; cutoff metrics are also listed for
comparison with forecast features. This is not an exploratory lag search.
Constant series and small samples are explicitly marked unavailable.</p>
<h2>Weekday versus weekend</h2>{groups.to_html(escape=True)}
<h2>Chronological model selection</h2>{block(metrics)}
<p>Ridge uses summed device-minutes, summed evening minutes and weekend indicator. Every
backtest prediction fits scaling and regression on earlier paired nights only.
Its baseline uses earlier valid sleep scores. The same 0..100 clipping rule
applies to backtesting and forecasts. Selection results are not an independent
final test set. Source corrections may revise retrospective training data.</p>
<h3>Recent backtest predictions</h3>{backtest.tail(30).to_html(index=False, escape=True)}
<h2>Saved forecast history and subsequent actuals</h2>
{history.tail(30).to_html(index=False, escape=True, na_rep="Not yet available")}
<h2>Recent aligned observations</h2>
{aligned.tail(30).to_html(index=False, escape=True, na_rep="Missing")}
<p>This report embeds Plotly locally, does not fetch a CDN, and is intended to
remain on this Mac. Raw data, reports, forecasts and model artifacts must not
be committed or uploaded.</p></body></html>"""
