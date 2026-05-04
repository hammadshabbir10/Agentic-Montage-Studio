"""
main_phase3.py  –  Phase 3 Entry Point (Video Generation & Composition)

Inputs
------
- scene_manifest_*.json  (Phase 1)
- character_db_*.json    (Phase 1)
- timing_manifest_*.json (Phase 2, latest auto-detected if not provided)

Outputs
-------
data/phase3_runs/runXX/
  images/scene_NN_*.png
  clips/scene_NN_kb.mp4
  composed/scene_NN_composed.mp4
  final_output_runXX.mp4
  phase3_state_runXX.json
  phase3_outputs_runXX.json
  image_prompts_runXX.json
  ffmpeg_commands_runXX.log
  subtitles_runXX.srt           (if --enable-subtitles)

Usage
-----
  python -m src.main_phase3
  python -m src.main_phase3 --quality cinematic --enable-subtitles
  python -m src.main_phase3 --backend pollinations
  python -m src.main_phase3 --timing data/phase2_runs/run01/timing_manifest_run01.json
  python -m src.main_phase3 --scene-id 3       # partial rerun
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.io.phase3_contracts import Phase3ValidationError
from src.mcp.tool_client import ToolClient
from src.mcp.tool_registry import ToolRegistry
from src.memory.vector_store import MemoryStore
from src.run_manager import (
    derive_phase2_run_tag,
    find_latest_timing_manifest,
    get_next_phase3_run,
)
from src.workflows.langgraph_phase3 import build_graph


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _print_banner(title: str, width: int = 60) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Project Montage – Phase 3: Video Composition")
    parser.add_argument("--manifest", type=str, default="data/scene_manifest_auto.json",
                        help="Path to scene_manifest JSON from Phase 1")
    parser.add_argument("--characters", type=str, default="data/character_db_auto.json",
                        help="Path to character_db JSON from Phase 1")
    parser.add_argument("--story", type=str, default="data/story_manifest_auto.json",
                        help="Path to story_manifest JSON from Phase 1 (used for title card)")
    parser.add_argument("--timing", type=str, default="",
                        help="Path to timing_manifest JSON from Phase 2 "
                             "(auto-detect latest if not provided)")
    parser.add_argument("--phase2-run", type=str, default="",
                        help="Specific Phase 2 run tag (e.g. run01) to align Phase 3 outputs to")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "hf", "pollinations"],
                        help="Image generation backend (auto = HF then Pollinations)")
    parser.add_argument("--quality", type=str, default="balanced",
                        choices=["fast", "balanced", "cinematic"],
                        help="Quality profile for resolution / fps / codec settings")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for deterministic image generation")
    parser.add_argument("--scene-id", type=int, default=None,
                        help="Partial rerun: regenerate only this scene id")
    parser.add_argument("--enable-subtitles", action="store_true",
                        help="Burn dialogue subtitles into the final MP4")
    parser.add_argument("--transition-sec", type=float, default=0.35,
                        help="Crossfade duration between scenes in seconds (0 disables fades)")
    parser.add_argument("--scene-image-only", action="store_true",
                        help="Use one image per scene (disable speaker-focused line images)")
    parser.add_argument("--disable-cinematic", action="store_true",
                        help="Disable cinematic look (color grade + grain + vignette + letterbox)")
    parser.add_argument("--disable-title-card", action="store_true",
                        help="Disable the intro title card")
    parser.add_argument("--disable-end-card", action="store_true",
                        help="Disable the closing end card")
    parser.add_argument("--title-card-sec", type=float, default=3.0,
                        help="Duration of the intro title card in seconds")
    parser.add_argument("--end-card-sec", type=float, default=3.0,
                        help="Duration of the closing end card in seconds")
    parser.add_argument("--motion-engine", type=str, default="auto",
                        choices=["auto", "pil", "zoompan"],
                        help="Ken Burns engine: auto (PIL if available), pil, or zoompan")
    parser.add_argument(
        "--strict-character-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, enforces stronger character identity consistency "
            "(reuses canonical speaker image per scene). Use "
            "--no-strict-character-consistency to allow more visual variation."
        ),
    )
    args = parser.parse_args()

    _print_banner("PHASE 3 — Video Generation & Composition  [STARTING]")
    print(f"[Phase 3] Loading scene manifest : {args.manifest}")
    print(f"[Phase 3] Loading characters     : {args.characters}")

    # Resolve timing manifest
    timing_path = Path(args.timing) if args.timing else find_latest_timing_manifest()
    if not timing_path or not Path(timing_path).exists():
        raise SystemExit(
            "[Phase 3] ERROR: timing manifest not found. Run Phase 2 first or pass --timing."
        )
    print(f"[Phase 3] Using timing manifest  : {timing_path}")

    scene_manifest  = _load_json(args.manifest)
    timing_manifest = _load_json(str(timing_path))
    character_db    = _load_json(args.characters)
    story_manifest  = _load_json(args.story) if Path(args.story).exists() else {}

    scenes = scene_manifest.get("scenes", [])
    if not scenes:
        raise SystemExit("[Phase 3] ERROR: scene_manifest has no scenes.")
    scene_ids = [s.get("scene_id") for s in scenes]

    # Allocate run dir aligned to Phase 2 run for traceability
    align_tag = args.phase2_run or derive_phase2_run_tag(timing_path) or ""
    run_info = get_next_phase3_run(scene_ids, align_to_phase2_run=align_tag or None)
    print(f"[Phase 3] Run tag : {run_info['run_tag'].upper()}  (aligned to Phase 2: {align_tag or 'no'})")
    print(f"[Phase 3] Run dir : {run_info['run_dir']}")
    print(
        f"[Phase 3] Backend : {args.backend}    Quality : {args.quality}    "
        f"Seed : {args.seed}    Transition : {args.transition_sec:.2f}s    "
        f"SpeakerFocus : {not args.scene_image_only}    "
        f"Cinematic : {not args.disable_cinematic}    "
        f"TitleCard : {not args.disable_title_card}    "
        f"EndCard : {not args.disable_end_card}    "
        f"MotionEngine : {args.motion_engine}    "
        f"StrictCharacterConsistency : {args.strict_character_consistency}"
    )

    # Infrastructure
    registry     = ToolRegistry()
    memory_store = MemoryStore(persist_dir="data/memory")
    tool_client  = ToolClient(
        registry,
        memory_store=memory_store,
        image_dir="data/image_assets",
    )

    # Build LangGraph state
    state: Dict[str, Any] = {
        "scene_manifest":   scene_manifest,
        "timing_manifest":  timing_manifest,
        "character_db":     character_db,
        "story_manifest":   story_manifest,
        "tool_client":      tool_client,
        "memory_store":     memory_store,
        "phase3_state":     {"status": "processing"},
        "errors":           [],
        # paths
        "run_tag":          run_info["run_tag"],
        "run_dir":          str(run_info["run_dir"]),
        "images_dir":       str(run_info["images_dir"]),
        "clips_dir":        str(run_info["clips_dir"]),
        "composed_dir":     str(run_info["composed_dir"]),
        "final_path":       str(run_info["final_path"]),
        "state_path":       str(run_info["state_path"]),
        "outputs_path":     str(run_info["outputs_path"]),
        "prompts_path":     str(run_info["prompts_path"]),
        "subtitles_save":   str(run_info["subtitles_path"]),
        "ffmpeg_log":       str(run_info["ffmpeg_log"]),
        # options
        "backend":          args.backend,
        "quality":          args.quality,
        "seed":             args.seed,
        "only_scene_id":    args.scene_id,
        "enable_subtitles": bool(args.enable_subtitles),
        "transition_sec":   max(0.0, args.transition_sec),
        "speaker_focus":    (not args.scene_image_only),
        "cinematic":        (not args.disable_cinematic),
        "enable_title_card": (not args.disable_title_card),
        "enable_end_card":   (not args.disable_end_card),
        "title_card_sec":    max(1.0, args.title_card_sec),
        "end_card_sec":      max(1.0, args.end_card_sec),
        "motion_engine":     args.motion_engine,
        "strict_character_consistency": bool(args.strict_character_consistency),
    }

    graph = build_graph()
    try:
        result = graph.invoke(state)
    except Phase3ValidationError as exc:
        print(f"\n[Phase 3] CONTRACT ERROR: {exc}")
        raise SystemExit(2)

    # Persist phase3 outputs
    _write_json(Path(state["state_path"]), result.get("phase3_state", {}))
    _write_json(Path(state["outputs_path"]), {
        "image_results":     result.get("image_results", []),
        "clip_results":      result.get("clip_results", []),
        "composed_results":  result.get("composed_results", []),
        "final_video_path":  result.get("final_video_path", ""),
        "subtitles_path":    result.get("subtitles_path", ""),
        "options": {
            "backend":          args.backend,
            "quality":          args.quality,
            "seed":             args.seed,
            "only_scene_id":    args.scene_id,
            "enable_subtitles": bool(args.enable_subtitles),
            "transition_sec":   max(0.0, args.transition_sec),
            "speaker_focus":    (not args.scene_image_only),
            "cinematic":        (not args.disable_cinematic),
            "enable_title_card": (not args.disable_title_card),
            "enable_end_card":   (not args.disable_end_card),
            "title_card_sec":   max(1.0, args.title_card_sec),
            "end_card_sec":     max(1.0, args.end_card_sec),
            "motion_engine":    args.motion_engine,
            "strict_character_consistency": bool(args.strict_character_consistency),
        },
        "inputs": {
            "scene_manifest":   args.manifest,
            "character_db":     args.characters,
            "timing_manifest":  str(timing_path),
        },
    })

    _print_banner(f"PHASE 3 — COMPLETE  [{run_info['run_tag'].upper()}]")
    print(f"  Final video    : {result.get('final_video_path', 'N/A')}")
    print(f"  Image prompts  : {state['prompts_path']}")
    print(f"  ffmpeg log     : {state['ffmpeg_log']}")
    print(f"  Outputs JSON   : {state['outputs_path']}")
    if result.get("subtitles_path"):
        print(f"  Subtitles      : {result['subtitles_path']}")

    # Backend telemetry summary
    by_backend: Dict[str, int] = {}
    for r in result.get("image_results", []):
        by_backend[r["backend"]] = by_backend.get(r["backend"], 0) + 1
    if by_backend:
        print("\n  Image backends used:")
        for b, n in by_backend.items():
            print(f"    {n:3d} × {b}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
