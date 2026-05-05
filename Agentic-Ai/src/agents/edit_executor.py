"""
edit_executor.py  –  Phase 5 Edit Execution Engine

Receives a classified EditIntent and dispatches the appropriate pipeline
re-execution:
  - script  → re-invoke Phase 1, cascade through Phase 2 + 3
  - audio   → re-invoke Phase 2 for affected scenes
  - video_frame → re-generate images + recompose via Phase 3
  - video   → re-run Phase 3 composition only (FFmpeg params)

Before executing, the caller (LangGraph workflow) snapshots the current state.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.edit_intent_classifier import EditIntent
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]  # Agentic-Ai/
DATA_DIR = ROOT / "data"


def execute(
    intent: EditIntent,
    current_state: Dict[str, Any],
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute an edit based on the classified intent.

    Parameters
    ----------
    intent : EditIntent
        The classified edit intent from the intent classifier.
    current_state : dict
        The current pipeline state (story, scene, character, timing manifests).
    run_config : dict, optional
        Pipeline run configuration (quality, backend, subtitles, etc.)

    Returns
    -------
    dict
        Execution result with keys: success, description, target, changes, errors
    """
    config = run_config or {}
    target = intent.target.lower()

    dispatch = {
        "script":      _execute_script_edit,
        "audio":       _execute_audio_edit,
        "video_frame": _execute_video_frame_edit,
        "video":       _execute_video_edit,
    }

    handler = dispatch.get(target)
    if not handler:
        return {
            "success": False,
            "description": f"Unknown target: {target}",
            "target": target,
            "changes": [],
            "errors": [f"No handler for target '{target}'"],
        }

    try:
        return handler(intent, current_state, config)
    except Exception as exc:
        LOGGER.error("Edit execution failed: %s", exc, exc_info=True)
        return {
            "success": False,
            "description": f"Execution failed: {exc}",
            "target": target,
            "changes": [],
            "errors": [str(exc)],
        }


# ── Script Edit (Phase 1 re-run, cascades to Phase 2 + 3) ───────────────────

def _execute_script_edit(
    intent: EditIntent,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-invoke Phase 1 with optional prompt modifier, then cascade."""
    prompt_modifier = intent.parameters.get("prompt_modifier", "")
    original_prompt = config.get("prompt", "A short animated story")

    if prompt_modifier:
        new_prompt = f"{original_prompt}. {prompt_modifier}"
    else:
        new_prompt = original_prompt

    scenes = config.get("scenes", 3)

    # Run Phase 1
    cmd = [
        sys.executable, "-m", "src.main",
        "--mode", "auto",
        "--prompt", new_prompt,
        "--scenes", str(scenes),
        "--auto-approve",
    ]
    code = _run_subprocess(cmd, "Phase 1 (script regeneration)")

    changes = ["Regenerated story, script, and character manifests"]

    if code == 0:
        # Cascade: Phase 2
        cmd2 = [sys.executable, "-m", "src.main_phase2"]
        code2 = _run_subprocess(cmd2, "Phase 2 (audio cascade)")
        if code2 == 0:
            changes.append("Regenerated audio and timing manifest")

            # Cascade: Phase 3
            cmd3 = _build_phase3_cmd(config)
            code3 = _run_subprocess(cmd3, "Phase 3 (video cascade)")
            if code3 == 0:
                changes.append("Regenerated video composition")
            else:
                return _error_result("Phase 3 cascade failed", intent, code3)
        else:
            return _error_result("Phase 2 cascade failed", intent, code2)

    return {
        "success": code == 0,
        "description": f"Script regenerated with modifier: {prompt_modifier or 'none'}",
        "target": "script",
        "intent": intent.intent,
        "changes": changes,
        "errors": [] if code == 0 else [f"Phase 1 exited with code {code}"],
    }


# ── Audio Edit (Phase 2 re-run for affected scenes) ─────────────────────────

def _execute_audio_edit(
    intent: EditIntent,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-invoke Phase 2 for voice/BGM changes."""
    changes: List[str] = []

    if intent.intent == "change_voice_tone":
        tone = intent.parameters.get("tone", "default")
        changes.append(f"Voice tone changed to: {tone}")
    elif intent.intent == "add_background_music":
        mood = intent.parameters.get("mood", "neutral")
        changes.append(f"Background music mood set to: {mood}")
    elif intent.intent == "adjust_volume":
        adj = intent.parameters.get("adjustment", "louder")
        changes.append(f"Volume adjusted: {adj}")
    elif intent.intent == "change_scene_mood":
        mood = intent.parameters.get("mood", "neutral")
        changes.append(f"Scene mood changed to: {mood}")
    else:
        changes.append(f"Audio edit: {intent.intent}")

    # Re-run Phase 2
    cmd = [sys.executable, "-m", "src.main_phase2"]
    code = _run_subprocess(cmd, "Phase 2 (audio edit)")

    if code == 0:
        # Re-run Phase 3 to incorporate new audio
        cmd3 = _build_phase3_cmd(config)
        code3 = _run_subprocess(cmd3, "Phase 3 (recompose after audio edit)")
        if code3 == 0:
            changes.append("Video recomposed with updated audio")
        else:
            changes.append(f"Phase 3 recompose failed (exit code {code3})")

    return {
        "success": code == 0,
        "description": f"Audio edit: {intent.intent}",
        "target": "audio",
        "intent": intent.intent,
        "changes": changes,
        "errors": [] if code == 0 else [f"Phase 2 exited with code {code}"],
    }


# ── Video Frame Edit (re-generate images, apply filters, recompose) ──────────

def _execute_video_frame_edit(
    intent: EditIntent,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-generate or filter scene images, then recompose."""
    changes: List[str] = []
    scene_id = intent.parameters.get("scene_id")

    if intent.intent == "apply_filter":
        # Apply OpenCV filter to existing scene images
        filter_name = intent.parameters.get("filter", "sepia")
        changed_files = _apply_filter_to_scene_images(filter_name, scene_id)
        changes.append(
            f"Applied {filter_name} filter to {len(changed_files)} image(s)"
        )

        # Recompose video with filtered images
        cmd3 = _build_phase3_cmd(config, scene_id=scene_id)
        code = _run_subprocess(cmd3, "Phase 3 (recompose after filter)")
        if code == 0:
            changes.append("Video recomposed with filtered images")
        else:
            changes.append(f"Phase 3 recompose failed (exit code {code})")

        return {
            "success": True,
            "description": f"Applied {filter_name} filter",
            "target": "video_frame",
            "intent": intent.intent,
            "changes": changes,
            "errors": [],
        }

    elif intent.intent in ("make_scene_darker", "make_scene_brighter"):
        brightness = intent.parameters.get("brightness", 0.3)
        if intent.intent == "make_scene_darker":
            factor = max(0.3, 1.0 + brightness)  # brightness is negative for darker
        else:
            factor = min(2.0, 1.0 + brightness)

        changed_files = _apply_filter_to_scene_images(
            "brightness", scene_id, factor=factor
        )
        changes.append(
            f"Adjusted brightness (factor={factor:.2f}) on {len(changed_files)} image(s)"
        )

        cmd3 = _build_phase3_cmd(config, scene_id=scene_id)
        code = _run_subprocess(cmd3, "Phase 3 (recompose after brightness)")
        if code == 0:
            changes.append("Video recomposed")

        return {
            "success": True,
            "description": f"Brightness adjusted: {intent.intent}",
            "target": "video_frame",
            "intent": intent.intent,
            "changes": changes,
            "errors": [],
        }

    elif intent.intent == "change_character_design":
        # Re-run Phase 3 image generation for affected character
        cmd3 = _build_phase3_cmd(config, scene_id=scene_id)
        code = _run_subprocess(cmd3, "Phase 3 (character design change)")
        changes.append("Regenerated scene images and recomposed video")
        return {
            "success": code == 0,
            "description": f"Character design changed",
            "target": "video_frame",
            "intent": intent.intent,
            "changes": changes,
            "errors": [] if code == 0 else [f"Phase 3 exited with code {code}"],
        }

    # Generic video_frame edit: re-run Phase 3
    cmd3 = _build_phase3_cmd(config, scene_id=scene_id)
    code = _run_subprocess(cmd3, "Phase 3 (video frame edit)")
    changes.append("Regenerated scene images and recomposed video")

    return {
        "success": code == 0,
        "description": f"Video frame edit: {intent.intent}",
        "target": "video_frame",
        "intent": intent.intent,
        "changes": changes,
        "errors": [] if code == 0 else [f"Phase 3 exited with code {code}"],
    }


# ── Video Edit (Phase 3 composition only) ───────────────────────────────────

def _execute_video_edit(
    intent: EditIntent,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-run Phase 3 composition/export with updated parameters."""
    changes: List[str] = []
    scene_id = intent.parameters.get("scene_id")

    if intent.intent == "remove_subtitle":
        config["enable_subtitles"] = False
        changes.append("Subtitles disabled")
    elif intent.intent == "speed_up_scene":
        factor = intent.parameters.get("speed_factor", 1.5)
        changes.append(f"Scene speed increased by {factor}x")
    elif intent.intent == "slow_down_scene":
        factor = intent.parameters.get("speed_factor", 0.5)
        changes.append(f"Scene speed decreased to {factor}x")
    else:
        changes.append(f"Video edit: {intent.intent}")

    # Re-run Phase 3
    cmd = _build_phase3_cmd(config, scene_id=scene_id)
    code = _run_subprocess(cmd, "Phase 3 (video edit)")
    if code == 0:
        changes.append("Video recomposed successfully")

    return {
        "success": code == 0,
        "description": f"Video edit: {intent.intent}",
        "target": "video",
        "intent": intent.intent,
        "changes": changes,
        "errors": [] if code == 0 else [f"Phase 3 exited with code {code}"],
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_subprocess(cmd: List[str], label: str) -> int:
    """Run a subprocess and return exit code. Logs output."""
    LOGGER.info("[edit_executor] Running %s: %s", label, " ".join(cmd))
    print(f"[edit_executor] Running {label}...")

    env = _subprocess_env()
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"  [{label}] {line.rstrip()}")
    code = process.wait()
    LOGGER.info("[edit_executor] %s exited with code %d", label, code)
    return code


def _subprocess_env() -> Dict[str, str]:
    """Build env for child processes (UTF-8 + ffmpeg PATH)."""
    import os
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def _build_phase3_cmd(
    config: Dict[str, Any],
    scene_id: Optional[int] = None,
) -> List[str]:
    """Build the Phase 3 subprocess command."""
    quality = config.get("quality", "balanced")
    backend = config.get("backend", "auto")
    subtitles = config.get("enable_subtitles", True)

    cmd = [
        sys.executable, "-m", "src.main_phase3",
        "--quality", quality,
        "--backend", backend,
    ]
    if subtitles:
        cmd.append("--enable-subtitles")
    if scene_id is not None:
        cmd.extend(["--scene-id", str(scene_id)])
    return cmd


def _apply_filter_to_scene_images(
    filter_name: str,
    scene_id: Optional[int] = None,
    **kwargs: Any,
) -> List[str]:
    """
    Apply an OpenCV filter to scene images in the latest Phase 3 run.

    Returns list of modified file paths.
    """
    from src.utils.image_filters import FILTER_REGISTRY

    func = FILTER_REGISTRY.get(filter_name)
    if not func:
        LOGGER.warning("Unknown filter: %s", filter_name)
        return []

    # Find latest Phase 3 run images
    phase3_runs = DATA_DIR / "phase3_runs"
    if not phase3_runs.exists():
        return []

    run_dirs = sorted(phase3_runs.glob("run*"), reverse=True)
    if not run_dirs:
        return []

    images_dir = run_dirs[0] / "images"
    if not images_dir.exists():
        return []

    changed: List[str] = []
    for img_path in sorted(images_dir.glob("*.png")):
        # If scene_id is specified, only process matching images
        if scene_id is not None:
            if f"scene_{scene_id:02d}" not in img_path.name:
                continue

        try:
            result = func(str(img_path), **kwargs)
            changed.append(result)
            print(f"  [filter] Applied {filter_name} to {img_path.name}")
        except Exception as exc:
            LOGGER.warning("Filter %s failed on %s: %s", filter_name, img_path, exc)

    return changed


def _error_result(msg: str, intent: EditIntent, code: int) -> Dict[str, Any]:
    return {
        "success": False,
        "description": msg,
        "target": intent.target,
        "intent": intent.intent,
        "changes": [],
        "errors": [f"{msg} (exit code {code})"],
    }


def collect_current_asset_paths() -> List[str]:
    """
    Collect all asset paths from the current pipeline outputs.
    Used by StateManager to snapshot assets before edits.
    """
    assets: List[str] = []

    # Phase 1 outputs
    for name in ("story_manifest_auto.json", "scene_manifest_auto.json",
                 "character_db_auto.json", "last_script_auto.txt"):
        p = DATA_DIR / name
        if p.exists():
            assets.append(str(p))

    # Latest Phase 2 timing manifest
    phase2_runs = DATA_DIR / "phase2_runs"
    if phase2_runs.exists():
        timing = sorted(phase2_runs.glob("run*/timing_manifest_run*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if timing:
            assets.append(str(timing[0]))

    # Latest Phase 3 final video
    phase3_runs = DATA_DIR / "phase3_runs"
    if phase3_runs.exists():
        videos = sorted(phase3_runs.glob("run*/final_output_run*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if videos:
            assets.append(str(videos[0]))

    return assets


def collect_current_state() -> Dict[str, Any]:
    """
    Load the current pipeline state from data/ JSON files.
    Returns a dict with story_manifest, scene_manifest, character_db keys.
    """
    state: Dict[str, Any] = {}

    mappings = {
        "story_manifest":  DATA_DIR / "story_manifest_auto.json",
        "scene_manifest":  DATA_DIR / "scene_manifest_auto.json",
        "character_db":    DATA_DIR / "character_db_auto.json",
    }
    for key, path in mappings.items():
        if path.exists():
            try:
                state[key] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                state[key] = {}
        else:
            state[key] = {}

    # Include latest timing manifest
    phase2_runs = DATA_DIR / "phase2_runs"
    if phase2_runs.exists():
        timing = sorted(phase2_runs.glob("run*/timing_manifest_run*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if timing:
            try:
                state["timing_manifest"] = json.loads(
                    timing[0].read_text(encoding="utf-8")
                )
                state["timing_manifest_path"] = str(timing[0])
            except (json.JSONDecodeError, OSError):
                pass

    # Include latest video path
    phase3_runs = DATA_DIR / "phase3_runs"
    if phase3_runs.exists():
        videos = sorted(phase3_runs.glob("run*/final_output_run*.mp4"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if videos:
            state["latest_video"] = str(videos[0])

    return state
