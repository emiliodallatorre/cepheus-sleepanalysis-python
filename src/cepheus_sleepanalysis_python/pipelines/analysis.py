from kedro.pipeline import Pipeline, node

from ..analysis import charts, correlate


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(correlate, "aligned", "correlations", name="measure_correlations"),
        node(
            charts, ["aligned", "params:study"],
            ["timeseries_chart", "scatter_chart"], name="plot_usage_and_sleep",
        ),
    ])
