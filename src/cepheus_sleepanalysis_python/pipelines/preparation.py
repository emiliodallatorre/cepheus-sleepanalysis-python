from kedro.pipeline import Pipeline, node

from ..preparation import align, build_usage, normalize_sessions, normalize_sleep


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            normalize_sessions, ["raw_screen", "params:study"], "sessions",
            name="normalize_screen_sessions",
        ),
        node(
            normalize_sleep, ["raw_sleep", "params:study"], "sleep",
            name="normalize_sleep_scores",
        ),
        node(
            build_usage, ["sessions", "raw_screen", "params:study"], "usage",
            name="aggregate_cutoff_usage",
        ),
        node(
            align, ["usage", "sleep", "raw_screen", "params:study"],
            ["aligned", "quality"], name="align_exposure_and_next_morning_sleep",
        ),
    ])
