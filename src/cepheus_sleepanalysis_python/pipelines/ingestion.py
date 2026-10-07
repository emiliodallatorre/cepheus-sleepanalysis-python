from kedro.pipeline import Pipeline, node

from ..sources import collect_screen, collect_sleep


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(collect_screen, "params:study", "raw_screen", name="collect_screen_sessions"),
        node(collect_sleep, "params:study", "raw_sleep", name="collect_garmin_sleep"),
    ])
