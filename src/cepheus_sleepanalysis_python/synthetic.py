from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .common import Parameters, days, fingerprint, now


def _daily_usage(p: Parameters) -> list[float]:
    rng = np.random.default_rng(731)
    return rng.uniform(35, 180, len(days(p))).tolist()


def screen(p: Parameters) -> dict[str, Any]:
    rows = []
    zone = ZoneInfo(p["timezone"])
    for day, minutes in zip(days(p), _daily_usage(p), strict=True):
        for device, name, factor, hour in (
            ("demo-mac", "Synthetic Mac", 1.0, 18),
            ("demo-phone", "Synthetic iPhone", 0.5, 18),
            ("demo-tablet", "Synthetic iPad", 0.2, 10),
        ):
            start = datetime.combine(day, time(hour), zone)
            # Clip to the shared cutoff so demo training/inference windows agree.
            end = min(
                start + timedelta(minutes=float(minutes * factor)),
                datetime.combine(day, time.fromisoformat(p["cutoff"]), zone),
            )
            if end <= start:
                continue
            rows.append({
                "device_id": device, "device": name,
                "bundle_id": "synthetic.example", "category": "Synthetic",
                "start": start.isoformat(), "end": end.isoformat(),
                "duration_s": (end - start).total_seconds(),
            })
    return {
        "source": "synthetic", "source_version": "1",
        "collected_at": now(p).isoformat(), "timezone": p["timezone"],
        "device_ids": p["device_ids"], "config_fingerprint": fingerprint(p),
        "successful_collection": True, "demo": True, "sessions": rows,
    }


def sleep(p: Parameters) -> dict[str, Any]:
    zone = ZoneInfo(p["timezone"])
    rng = np.random.default_rng(732)
    records = []
    for day, minutes in zip(days(p), _daily_usage(p), strict=True):
        target = day + timedelta(days=1)
        if target > now(p).date():
            continue
        start = datetime.combine(day, time(23), zone)
        end = datetime.combine(target, time(7), zone)
        score = int(np.clip(92 - 0.12 * min(minutes, 120) + rng.normal(0, 3), 0, 100))
        records.append({
            "requested_date": target.isoformat(),
            "fetched_at": now(p).isoformat(), "status": "fetched",
            "response": {"dailySleepDTO": {
                "calendarDate": target.isoformat(),
                "sleepStartTimestampGMT": int(start.timestamp() * 1000),
                "sleepEndTimestampGMT": int(end.timestamp() * 1000),
                "sleepTimeSeconds": int(end.timestamp() - start.timestamp()),
                "sleepScores": {"overall": {"value": score}},
            }},
        })
    return {
        "source": "synthetic", "source_version": "1", "demo": True,
        "timezone": p["timezone"], "collected_at": now(p).isoformat(),
        "records": records,
    }
