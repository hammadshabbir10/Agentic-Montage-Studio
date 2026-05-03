"""
phase3_contracts.py  –  Phase 3 Input / Output Contracts and Validation

Defines the JSON contracts that Phase 3 consumes (from Phase 1 + Phase 2)
and produces. Provides explicit, fail-fast validation so any missing field
is reported before we spend time on ffmpeg / image generation.

Inputs
------
1. scene_manifest        (Phase 1 output)
2. character_db          (Phase 1 output)
3. timing_manifest       (Phase 2 output)

Outputs (per scene)
-------------------
- scene image            (PNG)
- Ken Burns clip         (MP4, no audio)
- composed scene clip    (MP4, with voice + BGM)

Final outputs
-------------
- final_output_runXX.mp4
- phase3_state_runXX.json
- phase3_outputs_runXX.json
- image_prompts_runXX.json
- subtitles_runXX.srt (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Required keys per object ─────────────────────────────────────────────────
_REQUIRED_TIMING_SCENE_KEYS = {
    "scene_id", "audio_file", "start_ms", "end_ms", "duration_ms", "lines",
}
_REQUIRED_TIMING_LINE_KEYS = {
    "speaker", "line", "start_ms", "end_ms", "duration_ms",
}
_REQUIRED_SCENE_MANIFEST_KEYS = {"scenes"}
_REQUIRED_SCENE_KEYS = {"scene_id", "location", "dialogue"}


class Phase3ValidationError(Exception):
    """Raised when an input artifact does not satisfy the Phase 3 contract."""


# ── Validation helpers ───────────────────────────────────────────────────────

def _missing(required: set, present: set) -> set:
    return required - present


def _ensure_keys(name: str, payload: Dict[str, Any], required: set) -> None:
    if not isinstance(payload, dict):
        raise Phase3ValidationError(f"{name}: expected dict, got {type(payload).__name__}")
    missing = _missing(required, set(payload.keys()))
    if missing:
        raise Phase3ValidationError(
            f"{name}: missing required keys: {sorted(missing)}"
        )


def _ensure_file_exists(label: str, path_str: str) -> None:
    if not path_str:
        raise Phase3ValidationError(f"{label}: empty path")
    if not Path(path_str).exists():
        raise Phase3ValidationError(f"{label}: file not found at {path_str!r}")


def validate_scene_manifest(manifest: Dict[str, Any]) -> None:
    _ensure_keys("scene_manifest", manifest, _REQUIRED_SCENE_MANIFEST_KEYS)
    scenes = manifest.get("scenes", [])
    if not scenes:
        raise Phase3ValidationError("scene_manifest: has no scenes")
    for idx, scene in enumerate(scenes, start=1):
        _ensure_keys(f"scene_manifest.scenes[{idx}]", scene, _REQUIRED_SCENE_KEYS)


def validate_timing_manifest(timing: Dict[str, Any]) -> None:
    if "scenes" not in timing:
        raise Phase3ValidationError("timing_manifest: missing 'scenes'")
    scenes = timing.get("scenes", [])
    if not scenes:
        raise Phase3ValidationError("timing_manifest: has no scenes")
    for idx, scene in enumerate(scenes, start=1):
        _ensure_keys(f"timing_manifest.scenes[{idx}]", scene, _REQUIRED_TIMING_SCENE_KEYS)
        _ensure_file_exists(
            f"timing_manifest.scenes[{idx}].audio_file",
            scene.get("audio_file", ""),
        )
        # bgm_file may be missing for stub scenes — only validate when set
        bgm_path = scene.get("bgm_file", "")
        if bgm_path:
            _ensure_file_exists(
                f"timing_manifest.scenes[{idx}].bgm_file", bgm_path,
            )
        for line_idx, line in enumerate(scene.get("lines", []), start=1):
            _ensure_keys(
                f"timing_manifest.scenes[{idx}].lines[{line_idx}]",
                line,
                _REQUIRED_TIMING_LINE_KEYS,
            )


def validate_character_db(char_db: Dict[str, Any]) -> None:
    if "characters" not in char_db:
        raise Phase3ValidationError("character_db: missing 'characters'")
    if not isinstance(char_db["characters"], list):
        raise Phase3ValidationError("character_db.characters: must be a list")


# ── Phase 3 task model ───────────────────────────────────────────────────────

@dataclass
class ScenePlan:
    """Per-scene plan that Phase 3 consumes."""
    scene_id: int
    location: str
    duration_ms: int
    audio_file: str
    bgm_file: str
    mood: str
    visual_cues: List[str]
    speakers: List[str]
    lines: List[Dict[str, Any]]
    prompt: str = ""
    image_path: str = ""
    clip_path: str = ""
    composed_path: str = ""
    image_backend: str = ""
    image_prompt: str = ""

    @property
    def duration_sec(self) -> float:
        return max(2.0, self.duration_ms / 1000.0)


def build_scene_plans(
    scene_manifest: Dict[str, Any],
    timing_manifest: Dict[str, Any],
) -> List[ScenePlan]:
    """
    Cross-join scene_manifest scenes with timing_manifest scenes (by scene_id)
    and produce ScenePlan objects ready for Phase 3 execution.
    """
    timing_by_id = {s["scene_id"]: s for s in timing_manifest.get("scenes", [])}
    plans: List[ScenePlan] = []

    for scene in scene_manifest.get("scenes", []):
        sid = scene.get("scene_id")
        timing = timing_by_id.get(sid)
        if timing is None:
            raise Phase3ValidationError(
                f"scene {sid}: present in scene_manifest but missing from timing_manifest"
            )
        cues = [
            d.get("visual_cue", "").strip()
            for d in scene.get("dialogue", [])
            if d.get("visual_cue")
        ]
        speakers = list(dict.fromkeys(d.get("speaker", "") for d in scene.get("dialogue", [])))
        plans.append(
            ScenePlan(
                scene_id=sid,
                location=scene.get("location", ""),
                duration_ms=int(timing.get("duration_ms", 0)),
                audio_file=timing.get("audio_file", ""),
                bgm_file=timing.get("bgm_file", ""),
                mood=timing.get("mood", "neutral"),
                visual_cues=cues,
                speakers=[s for s in speakers if s],
                lines=timing.get("lines", []),
            )
        )

    return plans


def validate_phase3_inputs(
    scene_manifest: Dict[str, Any],
    timing_manifest: Dict[str, Any],
    character_db: Dict[str, Any],
) -> List[ScenePlan]:
    """
    Run all validations and return the cross-joined ScenePlan list.
    Raises Phase3ValidationError on any contract violation.
    """
    validate_scene_manifest(scene_manifest)
    validate_timing_manifest(timing_manifest)
    validate_character_db(character_db)
    return build_scene_plans(scene_manifest, timing_manifest)
