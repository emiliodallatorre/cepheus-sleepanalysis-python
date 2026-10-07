from .hooks import LocalDataHooks

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "merge_strategy": {"parameters": "soft"},
}
HOOKS = (LocalDataHooks(),)
DISABLE_HOOKS_FOR_PLUGINS = ("kedro-telemetry",)
