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
    story_manifest:    Dict[str, Any]

    plans:             List[ScenePlan]
    image_results:     List[Dict[str, Any]]
    line_image_results: Dict[int, List[Dict[str, Any]]]
    portrait_bank:     Dict[str, Dict[str, Any]]
    clip_results:      List[Dict[str, Any]]
    composed_results:  List[Dict[str, Any]]
    title_card_path:   str
    end_card_path:     str
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
    transition_sec:    float
    speaker_focus:     bool
    cinematic:         bool
    enable_title_card: bool
    enable_end_card:   bool
    title_card_sec:    float
    end_card_sec:      float
    motion_engine:     str  # auto | pil | zoompan
    strict_character_consistency: bool


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
    state["line_image_results"] = {}
    state["portrait_bank"] = {}
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
    portraits_dir = images_dir / "character_bank"
    speaker_focus = bool(state.get("speaker_focus", True))
    active_speakers: set[str] = set()
    for plan in plans:
        if state.get("only_scene_id") is not None and plan.scene_id != state["only_scene_id"]:
            continue
        for s in plan.speakers:
            if s:
                active_speakers.add(str(s).strip().upper())

    portrait_bank = scene_visualizer.generate_character_portrait_bank(
        character_db=state["character_db"],
        portraits_dir=portraits_dir,
        backend=state.get("backend", "auto"),
        quality=state.get("quality", "balanced"),
        seed=state.get("seed"),
        speaker_names=active_speakers,
    )
    state["portrait_bank"] = portrait_bank

    results: List[Dict[str, Any]] = []
    if not speaker_focus:
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
    line_images_by_scene: Dict[int, List[Dict[str, Any]]] = {}

    if speaker_focus:
        for plan in plans:
            if state.get("only_scene_id") is not None and plan.scene_id != state["only_scene_id"]:
                continue
            line_images = scene_visualizer.generate_scene_line_images(
                plan=plan,
                character_db=state["character_db"],
                images_dir=images_dir,
                backend=state.get("backend", "auto"),
                quality=state.get("quality", "balanced"),
                seed=state.get("seed"),
                portrait_bank=portrait_bank,
                strict_character_consistency=bool(
                    state.get("strict_character_consistency", True)
                ),
            )
            line_images_by_scene[plan.scene_id] = line_images
    state["line_image_results"] = line_images_by_scene

    # Save image prompts and backend telemetry
    prompts_payload = [
        {
            "scene_id":  r["scene_id"],
            "kind": "scene",
            "prompt":    r["prompt"],
            "backend":   r["backend"],
            "image_path": r["image_path"],
            "width":     r["width"],
            "height":    r["height"],
        }
        for r in results
    ]
    for sid, line_entries in line_images_by_scene.items():
        for e in line_entries:
            prompts_payload.append(
                {
                    "scene_id": sid,
                    "kind": "line",
                    "line_index": e.get("line_index"),
                    "speaker": e.get("speaker"),
                    "prompt": e.get("prompt"),
                    "backend": e.get("backend"),
                    "image_path": e.get("image_path"),
                    "width": e.get("width"),
                    "height": e.get("height"),
                    "start_ms": e.get("start_ms"),
                    "end_ms": e.get("end_ms"),
                    "duration_ms": e.get("duration_ms"),
                }
            )
    for _, anchor in portrait_bank.items():
        prompts_payload.append(
            {
                "scene_id": None,
                "kind": "character_anchor",
                "speaker": anchor.get("name"),
                "backend": anchor.get("backend"),
                "image_path": anchor.get("portrait_path"),
                "seed": anchor.get("seed"),
                "anchor_traits": anchor.get("anchor_traits", ""),
                "width": anchor.get("width"),
                "height": anchor.get("height"),
            }
        )
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
    cinematic = bool(state.get("cinematic", True))
    motion_engine = str(state.get("motion_engine", "auto"))

    clip_results: List[Dict[str, Any]] = []
    only = state.get("only_scene_id")
    line_images_by_scene = state.get("line_image_results", {})
    speaker_focus = bool(state.get("speaker_focus", True))

    for plan in plans:
        if only is not None and plan.scene_id != only:
            continue
        if not plan.image_path:
            raise RuntimeError(f"motion: scene {plan.scene_id} has no image")

        clip_path = clips_dir / f"scene_{plan.scene_id:02d}_kb.mp4"
        line_images = line_images_by_scene.get(plan.scene_id, [])
        if speaker_focus and line_images:
            video_compose.build_scene_clip_from_line_images(
                scene_id=plan.scene_id,
                line_images=line_images,
                out_path=clip_path,
                profile=profile,
                mood=plan.mood,
                cinematic=cinematic,
                motion_engine=motion_engine,
                log_path=log_path,
            )
            preset_name = "per_line"
        else:
            preset = video_compose.pick_motion_preset(
                visual_cue=plan.visual_cues[0] if plan.visual_cues else "",
                mood=plan.mood,
                scene_id=plan.scene_id,
                line_index=0,
            )
            video_compose.ken_burns_clip(
                image_path=plan.image_path,
                out_path=clip_path,
                duration_sec=plan.duration_sec,
                profile=profile,
                motion_preset=preset,
                cinematic=cinematic,
                log_path=log_path,
                engine=motion_engine,
            )
            preset_name = preset.name
        plan.clip_path = str(clip_path)
        clip_results.append({
            "scene_id":  plan.scene_id,
            "clip_path": str(clip_path),
            "duration_sec": plan.duration_sec,
            "motion_preset": preset_name,
            "cinematic": cinematic,
        })
        print(f"[Phase 3] motion: scene {plan.scene_id:02d} → {clip_path.name} ({preset_name})")

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

    profile = video_compose.get_profile(state.get("quality", "balanced"))
    log_path = Path(state["ffmpeg_log"])
    clips_dir = Path(state["clips_dir"])

    # Tier 5: title + end cards as bookends.
    story = state.get("story_manifest", {}).get("story", {}) if state.get("story_manifest") else {}
    title_text = (story.get("title") or "Agentic Montage Studio").strip()
    title_subtitle = (
        story.get("logline")
        or story.get("genre")
        or "An Agentic Montage Studio Production"
    ).strip()
    if not title_text or title_text.lower() == "untitled story":
        title_text = "Agentic Montage Studio"
    if (
        not title_subtitle
        or "could not be parsed" in title_subtitle.lower()
        or "details unavailable" in title_subtitle.lower()
    ):
        title_subtitle = "An Agentic Montage Studio Production"

    title_card_path: Optional[Path] = None
    end_card_path: Optional[Path] = None

    if state.get("enable_title_card", True):
        title_card_path = clips_dir / "title_card.mp4"
        video_compose.build_title_card(
            title=title_text,
            subtitle=title_subtitle,
            out_path=title_card_path,
            profile=profile,
            duration_sec=float(state.get("title_card_sec", 3.0)),
            log_path=log_path,
        )
        ordered_clips.insert(0, str(title_card_path))
        state["title_card_path"] = str(title_card_path)
        print(f"[Phase 3] mux: title card → {title_card_path.name}")

    if state.get("enable_end_card", True):
        end_card_path = clips_dir / "end_card.mp4"
        video_compose.build_end_card(
            title="The End",
            subtitle="Created with Agentic Montage Studio",
            out_path=end_card_path,
            profile=profile,
            duration_sec=float(state.get("end_card_sec", 3.0)),
            log_path=log_path,
        )
        ordered_clips.append(str(end_card_path))
        state["end_card_path"] = str(end_card_path)
        print(f"[Phase 3] mux: end card → {end_card_path.name}")

    final_path = Path(state["final_path"])
    transition_sec = float(state.get("transition_sec", 0.35))
    if transition_sec > 0:
        video_compose.concat_scenes_with_crossfade(
            ordered_clips,
            out_path=final_path,
            profile=profile,
            transition_sec=transition_sec,
            log_path=log_path,
        )
    else:
        video_compose.concat_scenes(ordered_clips, out_path=final_path, log_path=log_path)
    state["final_video_path"] = str(final_path)
    print(f"[Phase 3] mux: final video → {final_path}")
    return state


def subtitles_node(state: Phase3State) -> Phase3State:
    if not state.get("enable_subtitles"):
        return state
    _print_section("Phase 3 — Subtitle Burn-In")
    srt_path = Path(state["subtitles_save"])
    title_offset_sec = (
        float(state.get("title_card_sec", 3.0))
        if state.get("enable_title_card", True)
        else 0.0
    )
    video_compose.build_srt(
        state["timing_manifest"],
        srt_path,
        scene_transition_sec=float(state.get("transition_sec", 0.0)),
        global_offset_sec=title_offset_sec,
    )
    state["subtitles_path"] = str(srt_path)
    print(f"[Phase 3] subtitles: SRT written → {srt_path} (offset {title_offset_sec:.2f}s)")

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
        "portrait_bank":        state.get("portrait_bank", {}),
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
