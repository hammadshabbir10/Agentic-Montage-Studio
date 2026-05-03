"""
langgraph_phase3.py  –  Phase 3 LangGraph workflow (Video Composition)

Pipeline
--------
scene_parser
   -> image_gen
      -> motion          (Ken Burns animation per scene)
         -> compose      (mix Ken Burns clip + voice + BGM per scene)
            -> mux       (concat all scenes into final_output.mp4)
               -> subtitles_optional
                  -> memory_commit
                     -> END
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.agents import scene_visualizer
from src.io.phase3_contracts import (
    Phase3ValidationError,
    ScenePlan,
    validate_phase3_inputs,
)
from src.utils import video_compose
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


# ── State ───────────────────────────────────────────────────────────────────
class Phase3State(TypedDict, total=False):
    scene_manifest:    Dict[str, Any]
    timing_manifest:   Dict[str, Any]
    character_db:      Dict[str, Any]

    plans:             List[ScenePlan]
    image_results:     List[Dict[str, Any]]
    clip_results:      List[Dict[str, Any]]
    composed_results:  List[Dict[str, Any]]
    final_video_path:  str
    subtitles_path:    str

    errors:            List[str]
    phase3_state:      Dict[str, Any]
    tool_client:       object
    memory_store:      object

    run_tag:           str
    run_dir:           str
    images_dir:        str
    clips_dir:         str
    composed_dir:      str
    final_path:        str
    state_path:        str
    outputs_path:      str
    prompts_path:      str
    subtitles_save:    str
    ffmpeg_log:        str

    backend:           str   # hf | pollinations | auto
    quality:           str   # fast | balanced | cinematic
    seed:              Optional[int]
    only_scene_id:     Optional[int]
    enable_subtitles:  bool


def _print_section(title: str) -> None:
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


# ── Nodes ───────────────────────────────────────────────────────────────────

def scene_parser_node(state: Phase3State) -> Phase3State:
    print("\n" + "=" * 60)
    print("  PHASE 3 — Video Generation & Composition")
    print("=" * 60)
    print("[Phase 3] scene_parser: validating contracts...")

    try:
        plans = validate_phase3_inputs(
            scene_manifest=state["scene_manifest"],
            timing_manifest=state["timing_manifest"],
            character_db=state["character_db"],
        )
    except Phase3ValidationError as exc:
        state.setdefault("errors", []).append(str(exc))
        raise

    state["plans"] = plans
    state["image_results"]    = []
    state["clip_results"]     = []
    state["composed_results"] = []
    state.setdefault("phase3_state", {"status": "processing"})
    state["phase3_state"]["scenes_total"] = len(plans)

    print(f"[Phase 3] scene_parser: {len(plans)} scene plan(s) ready.")
    for p in plans:
        # Safety: prefer real audio duration over possibly stale timing manifest
        # so final video never cuts early even if Phase 2 timing is inaccurate.
        real_audio_ms = int(video_compose.probe_duration_sec(p.audio_file) * 1000)
        if real_audio_ms > 0 and abs(real_audio_ms - p.duration_ms) > 250:
            p.duration_ms = real_audio_ms
        print(f"           Scene {p.scene_id:02d}  {p.duration_sec:6.2f}s "
              f"mood={p.mood:11s} speakers={p.speakers}")
    return state


def image_gen_node(state: Phase3State) -> Phase3State:
    _print_section("Phase 3 — Scene Image Generation")
    plans: List[ScenePlan] = state["plans"]
    images_dir = Path(state["images_dir"])

    results = scene_visualizer.generate_all_scene_images(
        plans=plans,
        character_db=state["character_db"],
        images_dir=images_dir,
        backend=state.get("backend", "auto"),
        quality=state.get("quality", "balanced"),
        seed=state.get("seed"),
        only_scene_id=state.get("only_scene_id"),
    )
    state["image_results"] = results

    # Save image prompts and backend telemetry
    prompts_payload = [
        {
            "scene_id":  r["scene_id"],
            "prompt":    r["prompt"],
            "backend":   r["backend"],
            "image_path": r["image_path"],
            "width":     r["width"],
            "height":    r["height"],
        }
        for r in results
    ]
    Path(state["prompts_path"]).write_text(
        json.dumps(prompts_payload, indent=2), encoding="utf-8"
    )
    print(f"[Phase 3] image_gen: prompts saved → {state['prompts_path']}")
    return state


def motion_node(state: Phase3State) -> Phase3State:
    _print_section("Phase 3 — Ken Burns Motion")
    plans: List[ScenePlan] = state["plans"]
    clips_dir = Path(state["clips_dir"])
    profile = video_compose.get_profile(state.get("quality", "balanced"))
    log_path = Path(state["ffmpeg_log"])

    clip_results: List[Dict[str, Any]] = []
    only = state.get("only_scene_id")

    for plan in plans:
        if only is not None and plan.scene_id != only:
            continue
        if not plan.image_path:
            raise RuntimeError(f"motion: scene {plan.scene_id} has no image")

        clip_path = clips_dir / f"scene_{plan.scene_id:02d}_kb.mp4"
        # Alternate zoom direction for variety
        direction = "in" if plan.scene_id % 2 == 1 else "out"
        video_compose.ken_burns_clip(
            image_path=plan.image_path,
            out_path=clip_path,
            duration_sec=plan.duration_sec,
            profile=profile,
            direction=direction,
            log_path=log_path,
        )
        plan.clip_path = str(clip_path)
        clip_results.append({
            "scene_id":  plan.scene_id,
            "clip_path": str(clip_path),
            "duration_sec": plan.duration_sec,
            "direction": direction,
        })
        print(f"[Phase 3] motion: scene {plan.scene_id:02d} → {clip_path.name}")

    state["clip_results"] = clip_results
    return state


def compose_node(state: Phase3State) -> Phase3State:
    _print_section("Phase 3 — Per-Scene A/V Composition")
    plans: List[ScenePlan] = state["plans"]
    composed_dir = Path(state["composed_dir"])
    profile = video_compose.get_profile(state.get("quality", "balanced"))
    log_path = Path(state["ffmpeg_log"])

    composed_results: List[Dict[str, Any]] = []
    only = state.get("only_scene_id")

    for plan in plans:
        if only is not None and plan.scene_id != only:
            continue
        if not plan.clip_path:
            # In partial-rerun mode, allow reading existing clip
            existing = Path(state["clips_dir"]) / f"scene_{plan.scene_id:02d}_kb.mp4"
            if existing.exists():
                plan.clip_path = str(existing)
            else:
                raise RuntimeError(f"compose: scene {plan.scene_id} has no clip")

        out_path = composed_dir / f"scene_{plan.scene_id:02d}_composed.mp4"
        video_compose.compose_scene(
            clip_path=plan.clip_path,
            voice_path=plan.audio_file,
            bgm_path=plan.bgm_file,
            out_path=out_path,
            profile=profile,
            log_path=log_path,
        )
        plan.composed_path = str(out_path)
        composed_results.append({
            "scene_id":      plan.scene_id,
            "composed_path": str(out_path),
            "audio_file":    plan.audio_file,
            "bgm_file":      plan.bgm_file,
        })
        print(f"[Phase 3] compose: scene {plan.scene_id:02d} → {out_path.name}")

    state["composed_results"] = composed_results
    return state


def mux_node(state: Phase3State) -> Phase3State:
    _print_section("Phase 3 — Final Video Concatenation")
    plans: List[ScenePlan] = state["plans"]

    # If a partial rerun was requested, we still want to produce a final video
    # using existing composed clips for scenes not regenerated.
    composed_dir = Path(state["composed_dir"])
    ordered_clips: List[str] = []
    for plan in plans:
        candidate = plan.composed_path or str(composed_dir / f"scene_{plan.scene_id:02d}_composed.mp4")
        if not Path(candidate).exists():
            raise RuntimeError(f"mux: missing composed clip for scene {plan.scene_id}: {candidate}")
        ordered_clips.append(candidate)

    final_path = Path(state["final_path"])
    log_path = Path(state["ffmpeg_log"])
    video_compose.concat_scenes(ordered_clips, out_path=final_path, log_path=log_path)
    state["final_video_path"] = str(final_path)
    print(f"[Phase 3] mux: final video → {final_path}")
    return state


def subtitles_node(state: Phase3State) -> Phase3State:
    if not state.get("enable_subtitles"):
        return state
    _print_section("Phase 3 — Subtitle Burn-In")
    srt_path = Path(state["subtitles_save"])
    video_compose.build_srt(state["timing_manifest"], srt_path)
    state["subtitles_path"] = str(srt_path)
    print(f"[Phase 3] subtitles: SRT written → {srt_path}")

    profile = video_compose.get_profile(state.get("quality", "balanced"))
    log_path = Path(state["ffmpeg_log"])
    final_in = state["final_video_path"]
    final_out = str(Path(final_in).with_name(Path(final_in).stem + "_subbed.mp4"))
    video_compose.burn_subtitles(
        video_path=final_in,
        srt_path=str(srt_path),
        out_path=Path(final_out),
        profile=profile,
        log_path=log_path,
    )
    state["final_video_path"] = final_out
    print(f"[Phase 3] subtitles: subbed video → {final_out}")
    return state


def memory_commit_node(state: Phase3State) -> Phase3State:
    _print_section("Phase 3 — Memory Commit")
    payload = {
        "phase":                "phase3",
        "run_tag":              state.get("run_tag"),
        "final_video_path":     state.get("final_video_path"),
        "image_results":        state.get("image_results", []),
        "composed_results":     state.get("composed_results", []),
        "subtitles_path":       state.get("subtitles_path", ""),
    }
    try:
        client = state.get("tool_client")
        if client is not None:
            client.invoke_by_capability("commit_memory", payload)
            print("[Phase 3] memory_commit: done.")
        else:
            print("[Phase 3] memory_commit: skipped (no tool_client).")
    except Exception as exc:
        LOGGER.warning("memory_commit failed (non-fatal): %s", exc)
        print(f"[Phase 3] memory_commit: skipped ({exc})")

    if state.get("phase3_state"):
        state["phase3_state"]["status"] = "completed"
    return state


# ── Graph builder ───────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(Phase3State)

    graph.add_node("scene_parser",  scene_parser_node)
    graph.add_node("image_gen",     image_gen_node)
    graph.add_node("motion",        motion_node)
    graph.add_node("compose",       compose_node)
    graph.add_node("mux",           mux_node)
    graph.add_node("subtitles",     subtitles_node)
    graph.add_node("memory_commit", memory_commit_node)

    graph.set_entry_point("scene_parser")
    graph.add_edge("scene_parser",  "image_gen")
    graph.add_edge("image_gen",     "motion")
    graph.add_edge("motion",        "compose")
    graph.add_edge("compose",       "mux")
    graph.add_edge("mux",           "subtitles")
    graph.add_edge("subtitles",     "memory_commit")
    graph.add_edge("memory_commit", END)

    return graph.compile()
