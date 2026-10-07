from kedro.pipeline import Pipeline, node

from ..modeling import predict, record_forecast


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            predict,
            ["usage", "aligned", "sleep", "raw_screen", "model", "metrics", "params:study"],
            "forecast", name="forecast_tonights_sleep",
        ),
        node(
            record_forecast, ["forecast", "sleep", "params:study"], "forecast_history",
            name="retain_forecast_and_compare_actual",
        ),
    ])
