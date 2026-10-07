import argparse
import getpass
import hashlib
import json
import logging
import os
import subprocess
import time
import tomllib
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
from garminconnect.exceptions import GarminConnectNotFoundError

from .common import Parameters, atomic_json, days, fingerprint, load_json, now, validate

logger = logging.getLogger(__name__)


def login() -> None:
    parser = argparse.ArgumentParser(description="Authenticate Garmin locally with MFA.")
    parser.add_argument("--tokenstore", default="~/.config/cepheus/garmin")
    parser.add_argument("--region", choices=["global", "china"], default="global")
    args = parser.parse_args()
    path = Path(args.tokenstore).expanduser()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    client = Garmin(
        input("Garmin email: ").strip(),
        getpass.getpass("Garmin password: "),
        is_cn=args.region == "china",
        prompt_mfa=lambda: getpass.getpass("Garmin MFA code: ").strip(),
    )
    client.login(str(path))
    print(f"Garmin authentication saved locally in {path}.")


def _command(arguments: list[str], config: Path, timeout: int = 300) -> str:
    try:
        result = subprocess.run(
            arguments, check=True, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "SCREENTIME_CONFIG": str(config)},
        )
    except subprocess.CalledProcessError as error:
        logger.error("Screen Time command failed: %s", error.stderr or error.stdout)
        raise
    if result.stderr.strip():
        logger.warning("Screen Time diagnostic: %s", result.stderr.strip())
    return result.stdout


def collect_screen(p: Parameters) -> dict[str, Any]:
    validate(p)
    if p["demo"]:
        from .synthetic import screen

        return screen(p)
    config = Path(p["screentime_config"]).expanduser()
    with config.open("rb") as stream:
        settings = tomllib.load(stream)
    if settings.get("lookup", {}).get("enabled", True):
        raise ValueError("Disable Screen Time [lookup] enabled before local ingestion.")
    for section in ("influx", "home_assistant"):
        sink = settings.get(section, {})
        if sink.get("enabled") or sink.get("url"):
            raise ValueError(f"Disable the Screen Time {section} sink.")
    if any(
        os.environ.get(key)
        for key in ("INFLUX_URL", "INFLUX_TOKEN", "HA_URL", "HA_TOKEN")
    ):
        raise ValueError("Unset remote Screen Time sink environment variables.")
    if settings.get("timezone") != p["timezone"]:
        raise ValueError("Screen Time config timezone must match study.timezone.")
    if os.environ.get("SCREENTIME_TIMEZONE", p["timezone"]) != p["timezone"]:
        raise ValueError("Unset or correct SCREENTIME_TIMEZONE before collection.")
    cache = Path(p["data_root"]) / "01_raw" / "screen_archive.json"
    previous = load_json(cache) if cache.exists() else None
    if previous and (
        previous["timezone"] != p["timezone"]
        or previous["device_ids"] != p["device_ids"]
        or previous["demo"] != p["demo"]
    ):
        raise ValueError(
            "Screen archive configuration changed. Use a new data_root or explicitly "
            "move the old archive before collecting with the new configuration."
        )
    command = [p["screentime_executable"]]
    result = json.loads(_command([*command, "run"], config))
    if result.get("status") != "success" or result.get("errors"):
        raise RuntimeError(f"Screen Time collection did not succeed: {result}")
    rows = json.loads(_command([*command, "dump", "--format", "json"], config))
    if not isinstance(rows, list):
        raise ValueError("Screen Time dump must return a JSON array.")
    rows = [row for row in rows if row["device_id"] in p["device_ids"]]
    combined = {
        (row["device_id"], row["start"], row["bundle_id"]): row
        for row in [*(previous["sessions"] if previous else []), *rows]
    }
    payload = {
        "source": "screentime-exporter",
        "source_version": p["screentime_revision"],
        "collected_at": now(p).isoformat(),
        "timezone": p["timezone"],
        "device_ids": p["device_ids"],
        "config_fingerprint": fingerprint(p),
        "sessions": list(combined.values()),
        "successful_collection": True,
        "collection_details": result,
        "collections": [
            *(previous.get("collections", []) if previous else []),
            {
                "collected_at": now(p).isoformat(),
                "config_fingerprint": fingerprint(p),
                "details": result,
            },
        ],
        "demo": False,
    }
    atomic_json(cache, payload)
    return payload


def collect_sleep(p: Parameters) -> dict[str, Any]:
    validate(p)
    if p["demo"]:
        from .synthetic import sleep

        return sleep(p)
    path = Path(p["garmin_tokenstore"]).expanduser()
    if not (path / "garmin_tokens.json").exists():
        raise GarminConnectAuthenticationError(
            "Authenticate first with uv run cepheus-garmin-login."
        )
    client = Garmin(is_cn=p["garmin_region"] == "china")
    client.login(str(path))
    cache_dir = Path(p["data_root"]) / "01_raw" / "garmin"
    if not client.display_name:
        raise GarminConnectAuthenticationError("Garmin login did not identify an account.")
    identity = {
        "account_hash": hashlib.sha256(client.display_name.encode()).hexdigest(),
        "region": p["garmin_region"],
    }
    marker = cache_dir / ".account.json"
    if marker.exists() and load_json(marker) != identity:
        raise ValueError("Garmin account changed; select a new data_root.")
    atomic_json(marker, identity)
    records = []
    timestamp = now(p)
    for day in days(p):
        cache = cache_dir / f"{day.isoformat()}.json"
        refresh = day >= timestamp.date() - timedelta(days=int(p["refresh_days"]))
        if cache.exists() and not refresh:
            records.append(load_json(cache))
            continue
        for attempt in range(int(p["retry_attempts"])):
            try:
                response = client.get_sleep_data(day.isoformat())
                status = "fetched"
                break
            except GarminConnectNotFoundError:
                logger.warning("Garmin has no sleep record for %s.", day)
                response, status = {}, "missing"
                break
            except (
                GarminConnectConnectionError, GarminConnectTooManyRequestsError
            ) as error:
                if attempt + 1 == int(p["retry_attempts"]):
                    raise
                logger.warning(
                    "Garmin %s on %s; retry %s.", type(error).__name__, day, attempt + 1
                )
                time.sleep(
                    60 if isinstance(error, GarminConnectTooManyRequestsError)
                    else min(60, 2 ** (attempt + 1))
                )
        record = {
            "requested_date": day.isoformat(),
            "fetched_at": timestamp.isoformat(),
            "status": status,
            "response": response,
        }
        atomic_json(cache, record)
        records.append(record)
        time.sleep(float(p["request_pause_seconds"]))
    return {
        "source": "garminconnect",
        "source_version": version("garminconnect"),
        "collected_at": timestamp.isoformat(),
        "timezone": p["timezone"],
        "demo": False,
        "records": records,
    }
