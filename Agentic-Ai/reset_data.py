from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Runtime artifacts safe to clear for a fresh run.
RUNTIME_DIRS = [
    DATA_DIR / "phase2_runs",
    DATA_DIR / "phase3_runs",
    DATA_DIR / "task_graph_logs",
    DATA_DIR / "memory",
]

# Generated outputs that can be deleted between runs.
RUNTIME_FILES = [
    DATA_DIR / "character_db_auto.json",
    DATA_DIR / "last_script_auto.txt",
    DATA_DIR / "scene_manifest_auto.json",
    DATA_DIR / "story_manifest_auto.json",
    DATA_DIR / "timing_manifest.json",
]

RUN_COUNTER_PATHS = [
    DATA_DIR / "run_counter.json",
    DATA_DIR / "phase3_run_counter.json",
]


def clear_directory_contents(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return

    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=False)
        else:
            item.unlink()


def reset_run_counter(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump({"last_run": 0}, file_obj, indent=2)
        file_obj.write("\n")


def main() -> None:
    print("Resetting runtime data...")

    for directory in RUNTIME_DIRS:
        clear_directory_contents(directory)
        print(f"Cleared directory: {directory.relative_to(PROJECT_ROOT)}")

    for runtime_file in RUNTIME_FILES:
        if runtime_file.exists():
            runtime_file.unlink()
            print(f"Deleted file: {runtime_file.relative_to(PROJECT_ROOT)}")

    for counter_path in RUN_COUNTER_PATHS:
        reset_run_counter(counter_path)
        print(f"Reset counter: {counter_path.relative_to(PROJECT_ROOT)}")
    print("Done. Project is ready for a fresh run.")


if __name__ == "__main__":
    main()
