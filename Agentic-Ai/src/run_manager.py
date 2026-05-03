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

_PHASE3_RUNS_DIR     = Path("data/phase3_runs")
_PHASE3_COUNTER_FILE = Path("data/phase3_run_counter.json")


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


# ── Phase 3 run management ───────────────────────────────────────────────────

def _read_phase3_counter() -> int:
    try:
        data = json.loads(_PHASE3_COUNTER_FILE.read_text(encoding="utf-8"))
        return int(data.get("last_run", 0))
    except Exception:
        return 0


def _write_phase3_counter(n: int) -> None:
    _PHASE3_COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PHASE3_COUNTER_FILE.write_text(
        json.dumps({"last_run": n}, indent=2), encoding="utf-8"
    )


def get_next_phase3_run(
    scene_ids: list,
    align_to_phase2_run: str | None = None,
) -> Dict[str, object]:
    """
    Allocate a Phase 3 run directory.

    If `align_to_phase2_run` is provided (e.g. "run01"), Phase 3 reuses that
    same tag so artifacts from both phases are easy to correlate side-by-side.
    Otherwise, increments its own independent counter.

    Layout
    ------
    data/phase3_runs/runXX/
      images/
      clips/
      composed/
      final_output_runXX.mp4
      phase3_state_runXX.json
      phase3_outputs_runXX.json
      image_prompts_runXX.json
      ffmpeg_commands_runXX.log
      subtitles_runXX.srt
    """
    if align_to_phase2_run:
        run_tag = align_to_phase2_run
        try:
            run_number = int(run_tag.replace("run", ""))
        except ValueError:
            run_number = -1
    else:
        run_number = _read_phase3_counter() + 1
        run_tag = f"run{run_number:02d}"
        _write_phase3_counter(run_number)

    run_dir     = _PHASE3_RUNS_DIR / run_tag
    images_dir  = run_dir / "images"
    clips_dir   = run_dir / "clips"
    composed_dir = run_dir / "composed"

    for d in (images_dir, clips_dir, composed_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "run_number":   run_number,
        "run_tag":      run_tag,
        "run_dir":      run_dir,
        "images_dir":   images_dir,
        "clips_dir":    clips_dir,
        "composed_dir": composed_dir,
        "ffmpeg_log":   run_dir / f"ffmpeg_commands_{run_tag}.log",
        "final_path":   run_dir / f"final_output_{run_tag}.mp4",
        "state_path":   run_dir / f"phase3_state_{run_tag}.json",
        "outputs_path": run_dir / f"phase3_outputs_{run_tag}.json",
        "prompts_path": run_dir / f"image_prompts_{run_tag}.json",
        "subtitles_path": run_dir / f"subtitles_{run_tag}.srt",
    }


def find_latest_timing_manifest() -> Path | None:
    """Return the newest timing_manifest_runXX.json under data/phase2_runs."""
    if not _RUNS_DIR.exists():
        return None
    candidates = list(_RUNS_DIR.glob("run*/timing_manifest_run*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def derive_phase2_run_tag(timing_manifest_path: Path | str) -> str | None:
    """Extract 'runXX' from a timing manifest path."""
    p = Path(timing_manifest_path)
    parent = p.parent.name
    if parent.startswith("run"):
        return parent
    return None