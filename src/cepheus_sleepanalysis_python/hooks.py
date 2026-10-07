import os
from pathlib import Path
from typing import Any

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog

from .common import atomic_json, load_json, validate


class LocalDataHooks:
    @hook_impl
    def before_pipeline_run(
        self, run_params: dict[str, Any], catalog: DataCatalog
    ) -> None:
        parameters = catalog.load("params:study")
        validate(parameters)
        project = Path(run_params["project_path"])
        root = Path(parameters["data_root"])
        if not root.is_absolute():
            root = project / root
        root = root.resolve()
        data_dir = (project / "data").resolve()
        if root == data_dir or not root.is_relative_to(data_dir):
            raise ValueError("data_root must be a dedicated subdirectory of project data/.")
        marker = root / ".environment.json"
        identity = {"synthetic": parameters["demo"]}
        if marker.exists() and load_json(marker) != identity:
            raise ValueError("Refusing to mix synthetic and live data in the same data_root.")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        atomic_json(marker, identity)
