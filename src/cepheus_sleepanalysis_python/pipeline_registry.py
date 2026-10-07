from kedro.pipeline import Pipeline

from .pipelines import analysis, forecast, ingestion, preparation, reporting, training


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {
        "ingestion": ingestion.create_pipeline(),
        "preparation": preparation.create_pipeline(),
        "analysis": analysis.create_pipeline(),
        "training": training.create_pipeline(),
        "forecast": forecast.create_pipeline(),
        "reporting": reporting.create_pipeline(),
    }
    offline = (
        pipelines["preparation"] + pipelines["analysis"] + pipelines["training"]
        + pipelines["forecast"] + pipelines["reporting"]
    )
    pipelines["offline"] = offline
    pipelines["__default__"] = pipelines["ingestion"] + offline
    return pipelines
