import html
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from .common import Parameters

logger = logging.getLogger(__name__)


def correlate(aligned: pd.DataFrame) -> pd.DataFrame:
    paired = aligned.loc[aligned["eligible"]]
    features = [
        column for column in paired.columns
        if column.endswith("minutes") and not column.startswith("full_day")
    ]
    rows = []
    for feature in features:
        valid = paired[[feature, "sleep_score"]].dropna()
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
    series = make_subplots(rows=2, cols=1, shared_xaxes=True)
    for column in aligned.columns:
        if column.endswith("minutes") and not column.startswith("full_day"):
            series.add_trace(go.Scatter(
                x=aligned["exposure_date"], y=aligned[column],
                name=column, mode="lines",
            ), row=1, col=1)
    series.add_trace(go.Scatter(
        x=aligned["exposure_date"], y=aligned["sleep_score"],
        name="Following morning's sleep score", mode="lines+markers",
    ), row=2, col=1)
    series.update_yaxes(title_text="Minutes before cutoff", row=1, col=1)
    series.update_yaxes(title_text="Sleep score", range=[0, 100], row=2, col=1)
    series.update_layout(
        title=f"{label}Exposure day D and sleep on morning D+1",
        height=650, template="plotly_white",
    )
    scatter = go.Figure()
    for weekend, name in ((0, "Weekday"), (1, "Weekend")):
        subset = aligned.loc[aligned["eligible"] & aligned["weekend"].eq(weekend)]
        scatter.add_trace(go.Scatter(
            x=subset["combined_minutes"], y=subset["sleep_score"],
            text=subset["exposure_date"], name=name, mode="markers",
            hovertemplate="%{text}<br>%{x:.1f} min<br>Score %{y}<extra>%{fullData.name}</extra>",
        ))
    scatter.update_layout(
        title=f"{label}Cutoff usage versus following morning's sleep",
        xaxis_title="Combined unique wall-clock minutes",
        yaxis_title="Sleep score", template="plotly_white",
    )
    return series, scatter


def report(
    aligned: pd.DataFrame, quality: dict[str, Any], correlations: pd.DataFrame,
    series: go.Figure, scatter: go.Figure, metrics: dict[str, Any],
    forecast: dict[str, Any], backtest: pd.DataFrame, history: pd.DataFrame,
    p: Parameters,
) -> str:
    paired = aligned.loc[aligned["eligible"]]
    groups = paired.groupby("weekend").agg(
        paired_nights=("sleep_score", "count"),
        mean_sleep_score=("sleep_score", "mean"),
        mean_combined_minutes=("combined_minutes", "mean"),
    ).rename(index={0: "Weekday", 1: "Weekend"})
    label = "SYNTHETIC DEMO - " if p["demo"] else ""

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
<p>Timezone: {html.escape(p["timezone"])}. Feature window: local midnight to
{html.escape(p["cutoff"])}. Exposure day D predicts the sleep score on morning D+1.</p>
<p class="warning">Observational association is not causation or medical advice.
Garmin's score is a vendor-derived metric. Exercise, stress, caffeine, alcohol,
illness, travel and sleep habits can confound the relationship. Serial dependence
means ordinary independent-sample significance claims are inappropriate.
Device-minutes can overlap; combined minutes count their interval union.</p>
<h2>Tonight's estimate</h2>{block(forecast)}
<p>No numerical confidence interval is claimed. Backtest MAE describes historical
error, not a guarantee for tonight. Before cutoff or with unconfirmed/stale
inputs, no forecast is issued.</p>
<h2>Coverage and exclusions</h2>{block(quality)}
{series.to_html(full_html=False, include_plotlyjs=True)}
<h2>Correlation on eligible paired nights</h2>
{correlations.to_html(index=False, escape=True, na_rep="Unavailable")}
<p>No p-values or causal effect estimates are presented. The primary comparison
is same-day cutoff usage versus next-morning score, not an exploratory lag search.
Constant series and small samples are explicitly marked unavailable.</p>
{scatter.to_html(full_html=False, include_plotlyjs=False)}
<h2>Weekday versus weekend</h2>{groups.to_html(escape=True)}
<h2>Chronological model selection</h2>{block(metrics)}
<p>Ridge uses combined minutes, evening minutes and weekend indicator. Every
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
