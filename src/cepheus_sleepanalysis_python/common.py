import hashlib
import json
import logging
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
Parameters = dict[str, Any]


def now(p: Parameters) -> datetime:
    value = (
        datetime.fromisoformat(p["demo_now"])
        if p["demo"]
        else datetime.now(ZoneInfo(p["timezone"]))
    )
    if value.tzinfo is None:
        raise ValueError("Run timestamp must be timezone-aware.")
    return value.astimezone(ZoneInfo(p["timezone"]))


def cutoff(day: date, p: Parameters) -> datetime:
    return datetime.combine(
        day, time.fromisoformat(p["cutoff"]), ZoneInfo(p["timezone"])
    )


def days(p: Parameters) -> list[date]:
    today = now(p).date()
    count = int(p["history_days"])
    if count < 1:
        raise ValueError("history_days must be positive.")
    return [today - timedelta(days=i) for i in reversed(range(count))]


def fingerprint(p: Parameters) -> str:
    fields = (
        "timezone", "cutoff", "device_ids", "device_platforms", "confirmed_coverage", "demo",
        "min_paired_nights", "initial_training_nights", "min_backtest_nights",
        "ridge_alpha",
    )
    return hashlib.sha256(
        json.dumps({key: p[key] for key in fields}, sort_keys=True).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate(p: Parameters) -> None:
    ZoneInfo(p["timezone"])
    clock = time.fromisoformat(p["cutoff"])
    if clock.tzinfo is not None or clock == time(0):
        raise ValueError("cutoff must be a non-midnight local wall-clock time.")
    ids = p["device_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Configure a nonempty, unique study.device_ids allowlist.")
    platforms = p["device_platforms"]
    if set(platforms) != set(ids) or not set(platforms.values()) <= {"mac", "iphone", "ipad"}:
        raise ValueError("Map each selected device_id to mac, iphone or ipad.")
    if p["garmin_region"] not in ("global", "china"):
        raise ValueError("garmin_region must be global or china.")
    if int(p["retry_attempts"]) < 1 or int(p["refresh_days"]) < 1:
        raise ValueError("retry_attempts and refresh_days must be positive.")
    if float(p["request_pause_seconds"]) < 0:
        raise ValueError("request_pause_seconds cannot be negative.")
    if float(p["max_collection_age_minutes"]) <= 0:
        raise ValueError("max_collection_age_minutes must be positive.")
    if int(p["initial_training_nights"]) < 2 or int(p["min_backtest_nights"]) < 1:
        raise ValueError("Backtesting requires >=2 training nights and >=1 test night.")
    if int(p["min_paired_nights"]) < (
        int(p["initial_training_nights"]) + int(p["min_backtest_nights"])
    ):
        raise ValueError("min_paired_nights must cover training and backtest windows.")
    if float(p["ridge_alpha"]) <= 0:
        raise ValueError("ridge_alpha must be positive.")
    for item in p["confirmed_coverage"]:
        if item["device_id"] not in ids:
            raise ValueError("Coverage references a device outside the allowlist.")
        if date.fromisoformat(item["start"]) > date.fromisoformat(item["end"]):
            raise ValueError("Coverage start must not follow its end.")
