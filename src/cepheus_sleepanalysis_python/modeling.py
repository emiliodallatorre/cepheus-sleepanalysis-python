import hashlib
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    Parameters, atomic_json, cutoff, fingerprint, load_json, now, validate,
)

logger = logging.getLogger(__name__)
FEATURES = ["combined_minutes", "evening_minutes", "weekend"]
BACKTEST_COLUMNS = [
    "exposure_date", "target_sleep_date", "training_through",
    "actual_score", "regression_score", "baseline_score"
]


def data_fingerprint(aligned: pd.DataFrame, sleep: pd.DataFrame) -> str:
    paired = aligned.loc[aligned["eligible"], [
        "exposure_date", "target_sleep_date", *FEATURES, "sleep_score"
    ]].sort_values("exposure_date")
    scores = sleep.loc[sleep["sleep_status"].eq("eligible"), [
        "target_sleep_date", "sleep_score"
    ]].sort_values("target_sleep_date")
    return hashlib.sha256(
        (paired.to_json(orient="records") + scores.to_json(orient="records")).encode()
    ).hexdigest()


def _regressor(p: Parameters) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(p["ridge_alpha"]))),
    ])


def train(
    aligned: pd.DataFrame, sleep: pd.DataFrame, p: Parameters
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    validate(p)
    if not aligned["config_fingerprint"].eq(fingerprint(p)).all():
        raise ValueError("Cached features have different configuration; rerun preparation.")
    paired = aligned.loc[aligned["eligible"]].sort_values("exposure_date")
    scores = sleep.loc[
        sleep["sleep_status"].eq("eligible")
        & sleep["target_sleep_date"].le(now(p).date().isoformat())
    ].sort_values("target_sleep_date")
    artifact: dict[str, Any] = {
        "config_fingerprint": fingerprint(p),
        "data_fingerprint": data_fingerprint(aligned, sleep),
        "trained_at": now(p).isoformat(),
        "features": FEATURES,
        "paired_nights": len(paired),
        "baseline_nights": len(scores),
        "baseline_score": float(scores["sleep_score"].mean()) if len(scores) else None,
        "training_from": paired["exposure_date"].min() if len(paired) else None,
        "training_through": paired["exposure_date"].max() if len(paired) else None,
        "selected_method": "historical_mean",
        "regressor": None,
        "reason": "insufficient_paired_history",
        "feature_bounds": {},
        "demo": p["demo"],
    }
    metrics: dict[str, Any] = {
        "paired_nights": len(paired),
        "baseline_nights": len(scores),
        "minimum_paired_nights": int(p["min_paired_nights"]),
        "backtest_nights": 0,
        "regression_mae": None, "baseline_mae": None,
        "regression_rmse": None, "baseline_rmse": None,
        "evaluation": "Retrospective expanding-window model-selection backtest. "
        "Revised source records may differ from what was available at the time.",
    }
    predictions = []
    if len(paired) >= int(p["min_paired_nights"]):
        for index in range(int(p["initial_training_nights"]), len(paired)):
            past, target = paired.iloc[:index], paired.iloc[[index]]
            regressor = _regressor(p)
            regressor.fit(past[FEATURES], past["sleep_score"])
            baseline_history = scores.loc[
                scores["target_sleep_date"] < target["target_sleep_date"].iloc[0]
            ]
            predictions.append({
                "exposure_date": target["exposure_date"].iloc[0],
                "target_sleep_date": target["target_sleep_date"].iloc[0],
                "training_through": past["exposure_date"].iloc[-1],
                "actual_score": float(target["sleep_score"].iloc[0]),
                "regression_score": float(np.clip(regressor.predict(target[FEATURES])[0], 0, 100)),
                "baseline_score": float(baseline_history["sleep_score"].mean()),
            })
        backtest = pd.DataFrame(predictions, columns=BACKTEST_COLUMNS)
        metrics["backtest_nights"] = len(backtest)
        if len(backtest) >= int(p["min_backtest_nights"]):
            for method in ("regression", "baseline"):
                error = backtest[f"{method}_score"] - backtest["actual_score"]
                metrics[f"{method}_mae"] = float(error.abs().mean())
                metrics[f"{method}_rmse"] = float(np.sqrt((error ** 2).mean()))
            regressor = _regressor(p)
            regressor.fit(paired[FEATURES], paired["sleep_score"])
            artifact["regressor"] = regressor
            artifact["feature_bounds"] = {
                feature: [float(paired[feature].min()), float(paired[feature].max())]
                for feature in FEATURES
            }
            improved = metrics["regression_mae"] < metrics["baseline_mae"]
            artifact["selected_method"] = "ridge" if improved else "historical_mean"
            artifact["reason"] = (
                "regression_improves_backtest_mae" if improved else "baseline_wins_backtest"
            )
    metrics.update({
        "selected_method": artifact["selected_method"],
        "reason": artifact["reason"],
        "training_from": artifact["training_from"],
        "training_through": artifact["training_through"],
        "config_fingerprint": artifact["config_fingerprint"],
        "data_fingerprint": artifact["data_fingerprint"],
    })
    artifact["model_id"] = hashlib.sha256(
        (artifact["config_fingerprint"] + artifact["data_fingerprint"]).encode()
    ).hexdigest()[:16]
    logger.info(
        "Selected %s from %s paired nights: %s",
        artifact["selected_method"], len(paired), artifact["reason"]
    )
    return artifact, metrics, pd.DataFrame(predictions, columns=BACKTEST_COLUMNS)


def predict(
    usage: pd.DataFrame, aligned: pd.DataFrame, sleep: pd.DataFrame,
    raw: dict[str, Any], model: dict[str, Any], metrics: dict[str, Any], p: Parameters
) -> dict[str, Any]:
    validate(p)
    if model["config_fingerprint"] != fingerprint(p):
        raise ValueError("Cached model configuration differs; rerun training.")
    if model["data_fingerprint"] != data_fingerprint(aligned, sleep):
        raise ValueError("Cached model data differs; rerun training.")
    if metrics["data_fingerprint"] != model["data_fingerprint"]:
        raise ValueError("Cached metrics and model differ; rerun training.")
    timestamp = now(p)
    today = timestamp.date()
    current = usage.loc[usage["exposure_date"] == today.isoformat()]
    result: dict[str, Any] = {
        "generated_at": timestamp.isoformat(),
        "exposure_date": today.isoformat(),
        "target_sleep_date": (today + timedelta(days=1)).isoformat(),
        "timezone": p["timezone"], "cutoff_time": p["cutoff"],
        "synthetic": p["demo"], "status": "not_ready", "method": None,
        "predicted_sleep_score": None, "unclipped_estimate": None,
        "clipped": False, "baseline_score": None,
        "paired_nights": model["paired_nights"],
        "baseline_nights": model["baseline_nights"],
        "training_from": model["training_from"],
        "training_through": model["training_through"],
        "model_id": model["model_id"],
        "observed_features": {}, "experimental_regression_score": None,
        "experimental_unclipped_estimate": None,
        "backtest_mae": None, "reason": "", "warnings": [
            "Association, not causation; this is not a medical prediction."
        ],
    }
    if timestamp < cutoff(today, p):
        result["reason"] = "before_cutoff"
        return result
    collected = pd.Timestamp(raw["collected_at"])
    age = (timestamp - collected).total_seconds() / 60
    if (
        raw["demo"] != p["demo"] or not raw["successful_collection"]
        or age < 0 or age > float(p["max_collection_age_minutes"])
        or collected < cutoff(today, p)
    ):
        result["reason"] = "stale_or_unavailable_collection"
        return result
    if len(current) != 1 or not current["window_complete"].iloc[0]:
        result["reason"] = "today_window_unavailable"
        return result
    if (
        current["screen_collected_at"].iloc[0] != raw["collected_at"]
        or current["config_fingerprint"].iloc[0] != fingerprint(p)
    ):
        raise ValueError("Today's cached usage is stale; rerun preparation and training.")
    if not current["coverage_confirmed"].iloc[0] or current[FEATURES].isna().any().any():
        result["reason"] = "today_device_coverage_unknown"
        return result
    if pd.Timestamp(model["trained_at"]).date() != today:
        raise ValueError("Model is not trained for today's forecast; rerun training.")
    features = current[FEATURES]
    result["observed_features"] = {
        column: float(current[column].iloc[0])
        for column in current.columns
        if column.endswith("minutes") and pd.notna(current[column].iloc[0])
    }
    result["observed_features"]["weekend"] = int(features["weekend"].iloc[0])
    result["baseline_score"] = model["baseline_score"]
    if model["baseline_score"] is None:
        result.update(status="insufficient_history", reason="no_valid_sleep_history")
        return result
    estimate = model["baseline_score"]
    if model["regressor"] is not None:
        learned = float(model["regressor"].predict(features)[0])
        result["experimental_unclipped_estimate"] = learned
        result["experimental_regression_score"] = float(np.clip(learned, 0, 100))
        outside = [
            feature for feature in FEATURES
            if not model["feature_bounds"][feature][0]
            <= features[feature].iloc[0] <= model["feature_bounds"][feature][1]
        ]
        if outside:
            result["warnings"].append(f"Features outside training range: {', '.join(outside)}")
        if model["selected_method"] == "ridge":
            estimate = learned
    result.update(
        status="regression_ready" if model["selected_method"] == "ridge" else "baseline_only",
        method=model["selected_method"],
        reason=model["reason"],
        predicted_sleep_score=float(np.clip(estimate, 0, 100)),
        unclipped_estimate=float(estimate),
        clipped=not 0 <= estimate <= 100,
        backtest_mae=metrics[
            "regression_mae" if model["selected_method"] == "ridge" else "baseline_mae"
        ],
    )
    if result["status"] == "baseline_only":
        result["warnings"].append("Historical average only; not a usage-based estimate.")
    return result


def record_forecast(
    forecast: dict[str, Any], sleep: pd.DataFrame, p: Parameters
) -> pd.DataFrame:
    root = Path(p["data_root"]) / "07_model_output" / "history"
    record_id = hashlib.sha256(
        (forecast["generated_at"] + forecast["model_id"]).encode()
    ).hexdigest()
    path = root / f"{record_id}.json"
    if path.exists() and load_json(path) != forecast:
        raise ValueError("Refusing to overwrite a historical forecast with revised data.")
    atomic_json(path, forecast)
    records = []
    actuals = sleep.set_index("target_sleep_date")["sleep_score"]
    for saved in sorted(root.glob("*.json")):
        row = load_json(saved)
        records.append({
            "generated_at": row["generated_at"],
            "exposure_date": row["exposure_date"],
            "target_sleep_date": row["target_sleep_date"],
            "status": row["status"], "method": row["method"],
            "predicted_sleep_score": row["predicted_sleep_score"],
            "actual_sleep_score": actuals.get(row["target_sleep_date"], np.nan),
            "model_id": row["model_id"], "synthetic": row["synthetic"],
        })
    frame = pd.DataFrame(records)
    for column in ("predicted_sleep_score", "actual_sleep_score"):
        frame[column] = pd.to_numeric(frame[column])
    frame["absolute_error"] = (
        frame["predicted_sleep_score"] - frame["actual_sleep_score"]
    ).abs()
    return frame
