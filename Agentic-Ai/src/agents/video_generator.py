"""
video_generator.py  –  thin MCP-style wrapper around scene_visualizer.

Phase 3 generates a single still per scene via scene_visualizer (HF + Pollinations);
the actual animation is added by src/utils/video_compose.ken_burns_clip.
This file exists so older callers and unit tests can request a per-scene image
through the MCP capability "generate_scene_image".
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.io.phase3_contracts import ScenePlan
from src.agents import scene_visualizer


def run(
    task: Dict[str, object],
    tool_client,
    frames_dir: str,
    run_tag: str | None = None,
    character_db: Dict | None = None,
    backend: str = "auto",
    quality: str = "balanced",
    seed: int | None = None,
) -> Dict[str, object]:
    """
    Generate a single image for the supplied scene task using the same
    backend chain as Phase 3 (HF -> Pollinations).
    """
    scene_id = int(task.get("scene_id", 0))
    plan = ScenePlan(
        scene_id=scene_id,
        location=str(task.get("location", "")),
        duration_ms=int(task.get("duration_ms", 4000)),
        audio_file=str(task.get("audio_file", "")),
        bgm_file=str(task.get("bgm_file", "")),
        mood=str(task.get("mood", "neutral")),
        visual_cues=[
            d.get("visual_cue", "")
            for d in task.get("dialogue", [])
            if d.get("visual_cue")
        ],
        speakers=list(dict.fromkeys(
            d.get("speaker", "") for d in task.get("dialogue", []) if d.get("speaker")
        )),
        lines=list(task.get("dialogue", [])),
    )

    images_dir = Path(frames_dir)
    result = scene_visualizer.generate_scene_image(
        plan=plan,
        character_db=character_db or {"characters": []},
        images_dir=images_dir,
        backend=backend,
        quality=quality,
        seed=seed,
    )

    return {
        "scene_id":   scene_id,
        "image_path": result["image_path"],
        "backend":    result["backend"],
        "prompt":     result["prompt"],
    }
