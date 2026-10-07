"""Local Screen Time and sleep analysis."""

import os

os.environ["KEDRO_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

__version__ = "0.1.0"


def main() -> None:
    from .__main__ import main as run

    run()
