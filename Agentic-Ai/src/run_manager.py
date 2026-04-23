"""
run_manager.py  –  Phase 2 Run Manager

Directory layout per run
------------------------
data/
  phase2_runs/
    run01/
      audio/
        scene1/   scene2/   scene3/ ...
      bgm/
        scene1_tense_freesound.mp3
        scene2_mysterious_freesound.mp3
      timing_manifest_run01.json
    run02/
      audio/ ...
      bgm/   ...
      timing_manifest_run02.json
  task_graph_logs/
    run01_task_graph.json
    run02_task_graph.json
  run_counter.json
"""

import json
from pathlib import Path
from typing import Dict


_COUNTER_FILE = Path("data/run_counter.json")
_RUNS_DIR     = Path("data/phase2_runs")
_LOGS_DIR     = Path("data/task_graph_logs")


def _read_counter() -> int:
    try:
        data = json.loads(_COUNTER_FILE.read_text(encoding="utf-8"))
        return int(data.get("last_run", 0))
    except Exception:
        return 0


def _write_counter(n: int) -> None:
    _COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _COUNTER_FILE.write_text(
        json.dumps({"last_run": n}, indent=2), encoding="utf-8"
    )


def get_next_run(scene_ids: list) -> Dict[str, object]:
    """
    Increment the run counter, create all directories, return run_info.

    Returns
    -------
    {
      "run_number": 2,
      "run_tag":    "run02",
      "run_dir":    Path("data/phase2_runs/run02"),
      "audio_dir":  Path("data/phase2_runs/run02/audio"),
      "bgm_dir":    Path("data/phase2_runs/run02/bgm"),
      "scene_dirs": { 1: Path(".../audio/scene1"), ... },
      "log_path":   Path("data/task_graph_logs/run02_task_graph.json"),
    }
    """
    n         = _read_counter() + 1
    run_tag   = f"run{n:02d}"
    run_dir   = _RUNS_DIR / run_tag
    audio_dir = run_dir / "audio"
    bgm_dir   = run_dir / "bgm"          # run-specific BGM folder

    # Per-scene audio folders
    scene_dirs = {}
    for sid in scene_ids:
        scene_folder = audio_dir / f"scene{sid}"
        scene_folder.mkdir(parents=True, exist_ok=True)
        scene_dirs[sid] = scene_folder

    # BGM folder for this run
    bgm_dir.mkdir(parents=True, exist_ok=True)

    # Task graph logs folder
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    _write_counter(n)

    return {
        "run_number": n,
        "run_tag":    run_tag,
        "run_dir":    run_dir,
        "audio_dir":  audio_dir,
        "bgm_dir":    bgm_dir,
        "scene_dirs": scene_dirs,
        "log_path":   _LOGS_DIR / f"{run_tag}_task_graph.json",
    }


def save_task_graph_log(
    log_path: Path,
    run_info: Dict,
    scene_manifest: Dict,
    character_db: Dict,
) -> None:
    """
    Write task graph log at the START of every run.
    Contains: run metadata + full scene manifest + character db.
    Saved even if the run crashes halfway.
    """
    payload = {
        "run_tag":        run_info["run_tag"],
        "run_number":     run_info["run_number"],
        "run_dir":        str(run_info["run_dir"]),
        "bgm_dir":        str(run_info["bgm_dir"]),
        "scene_manifest": scene_manifest,
        "character_db":   character_db,
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] Task graph log saved -> {log_path}")