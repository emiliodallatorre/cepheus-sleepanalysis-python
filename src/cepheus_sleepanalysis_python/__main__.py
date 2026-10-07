from pathlib import Path

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main() -> None:
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    find_run_command(package_name)()


if __name__ == "__main__":
    main()
