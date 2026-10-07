from kedro.pipeline import Pipeline, node

from ..analysis import report


def create_pipeline() -> Pipeline:
    return Pipeline([
        node(
            report,
            [
                "aligned", "quality", "correlations", "timeseries_chart", "scatter_chart",
                "metrics", "forecast", "backtest", "forecast_history", "params:study",
            ],
            "report", name="render_local_analysis_report",
        ),
    ])
