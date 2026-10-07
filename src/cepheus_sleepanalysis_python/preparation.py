import logging
import math
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .common import Parameters, cutoff, days, fingerprint, now, validate

logger = logging.getLogger(__name__)
SESSION_COLUMNS = [
    "device_id", "device", "bundle_id", "category", "start", "end", "duration_s"
]
SLEEP_COLUMNS = [
    "target_sleep_date", "sleep_score", "sleep_start", "sleep_end",
    "sleep_seconds", "sleep_status", "fetched_at"
]


def _timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Source timestamps must be present and timezone-aware.")
    return stamp.tz_convert("UTC")


def normalize_sessions(raw: dict[str, Any], p: Parameters) -> pd.DataFrame:
    validate(p)
    if raw["demo"] != p["demo"] or raw["timezone"] != p["timezone"]:
        raise ValueError("Screen Time source environment/timezone mismatch.")
    if raw["device_ids"] != p["device_ids"]:
        raise ValueError("Recollect Screen Time after changing the device allowlist.")
    records = []
    for row in raw["sessions"]:
        if row["device_id"] not in p["device_ids"]:
            continue
        start, end = _timestamp(row["start"]), _timestamp(row["end"])
        duration = row["duration_s"]
        if (
            end <= start or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration < 0
        ):
            raise ValueError(f"Invalid interval for device {row['device_id']}.")
        records.append({
            "device_id": row["device_id"],
            "device": row["device"],
            "bundle_id": row["bundle_id"],
            "category": row["category"],
            "start": start,
            "end": end,
            "duration_s": (end - start).total_seconds(),
        })
    frame = pd.DataFrame(records, columns=SESSION_COLUMNS)
    for column in ("start", "end"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for device in p["device_ids"]:
        if not frame["device_id"].eq(device).any():
            logger.warning(
                "No archived sessions for selected device %s; verify ID and coverage.", device
            )
    return frame.drop_duplicates(["device_id", "start", "bundle_id"]).sort_values("start")


def normalize_sleep(raw: dict[str, Any], p: Parameters) -> pd.DataFrame:
    validate(p)
    if raw["demo"] != p["demo"] or raw["timezone"] != p["timezone"]:
        raise ValueError("Garmin source environment/timezone mismatch.")
    records = []
    for item in raw["records"]:
        requested = date.fromisoformat(item["requested_date"])
        response = item["response"]
        if not isinstance(response, dict):
            raise ValueError(f"Malformed Garmin response for {requested}.")
        dto = response.get("dailySleepDTO")
        if response and "dailySleepDTO" not in response:
            raise ValueError(f"Unrecognized Garmin sleep response schema for {requested}.")
        row = {
            "target_sleep_date": requested.isoformat(),
            "sleep_score": None,
            "sleep_start": pd.NaT,
            "sleep_end": pd.NaT,
            "sleep_seconds": None,
            "sleep_status": "missing_record",
            "fetched_at": item["fetched_at"],
        }
        if dto is not None:
            if not isinstance(dto, dict):
                raise ValueError(f"Malformed dailySleepDTO for {requested}.")
            reported = dto.get("calendarDate")
            if reported is None:
                raise ValueError(f"Missing Garmin calendarDate for {requested}.")
            if reported != requested.isoformat():
                raise ValueError(f"Garmin requested/reported date mismatch: {requested}.")
            scores = dto.get("sleepScores")
            if scores is None:
                scores = {}
            if not isinstance(scores, dict):
                raise ValueError(f"Malformed Garmin sleepScores for {requested}.")
            overall = scores.get("overall")
            if overall is None:
                overall = {}
            if not isinstance(overall, dict):
                raise ValueError(f"Malformed Garmin overall score for {requested}.")
            score = overall.get("value")
            if score is not None and (
                isinstance(score, bool) or not isinstance(score, (int, float))
                or not 0 <= score <= 100
            ):
                raise ValueError(f"Invalid Garmin sleep score for {requested}.")
            row["sleep_score"] = score
            row["sleep_seconds"] = dto.get("sleepTimeSeconds")
            duration = row["sleep_seconds"]
            if duration is not None and (
                isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0
            ):
                raise ValueError(f"Invalid sleep duration for {requested}.")
            start = dto.get("sleepStartTimestampGMT")
            end = dto.get("sleepEndTimestampGMT")
            row["sleep_status"] = "missing_score" if score is None else "missing_timestamps"
            if start is not None and end is not None:
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in (start, end)
                ):
                    raise ValueError(f"Invalid GMT sleep timestamps for {requested}.")
                row["sleep_start"] = pd.to_datetime(start, unit="ms", utc=True)
                row["sleep_end"] = pd.to_datetime(end, unit="ms", utc=True)
                local_start = row["sleep_start"].tz_convert(p["timezone"])
                local_end = row["sleep_end"].tz_convert(p["timezone"])
                exposure = requested - timedelta(days=1)
                overnight = (
                    row["sleep_end"] > row["sleep_start"]
                    and local_end.date() == requested
                    and local_start.date() in (exposure, requested)
                    and local_start >= cutoff(exposure, p)
                    and local_start < datetime.combine(
                        requested, time(12), ZoneInfo(p["timezone"])
                    )
                    and local_end.hour < 18
                    and 2 <= (row["sleep_end"] - row["sleep_start"]).total_seconds() / 3600 <= 16
                )
                row["sleep_status"] = (
                    "eligible" if overnight and score is not None
                    else "missing_score" if overnight else "ambiguous_sleep_window"
                )
        if row["sleep_status"] != "eligible":
            logger.warning("Sleep %s excluded: %s", requested, row["sleep_status"])
        records.append(row)
    frame = pd.DataFrame(records, columns=SLEEP_COLUMNS)
    for column in ("sleep_start", "sleep_end", "fetched_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["sleep_score"] = pd.to_numeric(frame["sleep_score"])
    if frame["target_sleep_date"].duplicated().any():
        raise ValueError("Duplicate Garmin target dates.")
    return frame.sort_values("target_sleep_date")


def union_minutes(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    first, last = intervals[0]
    seconds = 0.0
    for start, end in intervals[1:]:
        if start > last:
            seconds += (last - first).total_seconds()
            first, last = start, end
        else:
            last = max(last, end)
    return (seconds + (last - first).total_seconds()) / 60


def _window(
    frame: pd.DataFrame, start: datetime, end: datetime
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    left, right = pd.Timestamp(start), pd.Timestamp(end)
    matches = frame.loc[(frame["start"] < right) & (frame["end"] > left)]
    return [
        (max(row.start, left), min(row.end, right))
        for row in matches.itertuples()
    ]


def _covered(device: str, day: date, p: Parameters) -> bool:
    return any(
        item["device_id"] == device
        and date.fromisoformat(item["start"]) <= day <= date.fromisoformat(item["end"])
        for item in p["confirmed_coverage"]
    )


def build_usage(
    sessions: pd.DataFrame, raw: dict[str, Any], p: Parameters
) -> pd.DataFrame:
    validate(p)
    timestamp = now(p)
    collected = _timestamp(raw["collected_at"]).to_pydatetime()
    records = []
    for day in days(p):
        start = datetime.combine(day, time(0), ZoneInfo(p["timezone"]))
        stop = cutoff(day, p)
        tomorrow = datetime.combine(day + timedelta(days=1), time(0), start.tzinfo)
        coverage = {device: _covered(device, day, p) for device in p["device_ids"]}
        ready = timestamp >= stop and collected >= stop
        record = {
            "exposure_date": day.isoformat(),
            "target_sleep_date": (day + timedelta(days=1)).isoformat(),
            "weekend": int(day.weekday() >= 5),
            "screen_collected_at": raw["collected_at"],
            "config_fingerprint": fingerprint(p),
            "coverage_confirmed": all(coverage.values()),
            "window_complete": ready,
            "coverage_status": "confirmed" if all(coverage.values()) else "unknown",
            "combined_minutes": None,
            "evening_minutes": None,
            "summed_device_minutes": None,
            "full_day_combined_minutes": None,
        }
        intervals = []
        for device in p["device_ids"]:
            subset = sessions.loc[sessions["device_id"] == device]
            clipped = _window(subset, start, stop)
            record[f"device__{device}__minutes"] = (
                union_minutes(clipped) if coverage[device] and ready else None
            )
            intervals.extend(clipped)
        for platform in sorted(set(p["device_platforms"].values())):
            devices = [
                device for device in p["device_ids"]
                if p["device_platforms"][device] == platform
            ]
            subset = sessions.loc[sessions["device_id"].isin(devices)]
            record[f"platform__{platform}__minutes"] = (
                union_minutes(_window(subset, start, stop))
                if ready and all(coverage[device] for device in devices) else None
            )
        if all(coverage.values()) and ready:
            evening_start = min(
                stop, datetime.combine(day, time(18), start.tzinfo)
            )
            record["combined_minutes"] = union_minutes(intervals)
            record["evening_minutes"] = union_minutes(
                _window(sessions, evening_start, stop)
            )
            record["summed_device_minutes"] = sum(
                record[f"device__{device}__minutes"] for device in p["device_ids"]
            )
            if collected >= tomorrow and timestamp >= tomorrow:
                record["full_day_combined_minutes"] = union_minutes(
                    _window(sessions, start, tomorrow)
                )
        records.append(record)
    return pd.DataFrame(records)


def align(
    usage: pd.DataFrame, sleep: pd.DataFrame, raw: dict[str, Any], p: Parameters
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = usage.merge(sleep, on="target_sleep_date", how="left", validate="one_to_one")
    frame["sleep_status"] = frame["sleep_status"].fillna("not_available")
    frame["eligible"] = (
        frame["coverage_confirmed"] & frame["window_complete"]
        & frame["sleep_status"].eq("eligible") & frame["sleep_score"].notna()
        & frame["target_sleep_date"].le(now(p).date().isoformat())
    )
    quality = {
        "generated_at": now(p).isoformat(),
        "synthetic": p["demo"],
        "timezone": p["timezone"],
        "cutoff": p["cutoff"],
        "device_ids": p["device_ids"],
        "exposure_from": usage["exposure_date"].min(),
        "exposure_to": usage["exposure_date"].max(),
        "days": len(frame),
        "paired_nights": int(frame["eligible"].sum()),
        "unknown_coverage_days": int((~frame["coverage_confirmed"]).sum()),
        "incomplete_windows": int((~frame["window_complete"]).sum()),
        "sleep_status_counts": {
            str(key): int(value) for key, value in frame["sleep_status"].value_counts().items()
        },
        "screen_collected_at": raw["collected_at"],
        "coverage_basis": "Explicit user-confirmed ranges; absent sessions alone are not zero.",
        "sleep_window_rule": "Main overnight heuristic: 2..16h, ends on reported morning "
        "before 18:00, starts after exposure cutoff and before morning noon.",
    }
    if not frame["eligible"].any():
        logger.warning("No eligible paired nights. Review coverage and sleep exclusions.")
    return frame, quality
