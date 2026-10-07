from kedro.pipeline import Pipeline, node

from ..modeling import train


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            train, ["aligned", "sleep", "params:study"], ["model", "metrics", "backtest"],
            name="backtest_and_select_sleep_model",
        ),
    ])
