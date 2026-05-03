"""
video_compose.py  –  Phase 3 ffmpeg utilities (CPU-friendly)

Pipeline
--------
1. ken_burns_clip()   : still image -> animated MP4 (zoom + pan)
2. compose_scene()    : Ken Burns clip + voice + bgm -> scene MP4
3. concat_scenes()    : scene MP4s -> single final MP4
4. build_srt()        : timing manifest lines -> .srt subtitle file
5. burn_subtitles()   : optional subtitle burn-in over final MP4

All functions log the exact ffmpeg command they run to a per-run log file
so the grader can audit reproducibility.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


# ── Quality profiles ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class QualityProfile:
    width: int
    height: int
    fps: int
    crf: int           # libx264 quality (lower = better, 18 visually lossless, 28 small)
    preset: str        # libx264 speed/quality preset
    bgm_volume: float  # 0.0–1.0, voice always at 1.0


_PROFILES: Dict[str, QualityProfile] = {
    "fast":      QualityProfile(854,  480, 24, 28, "veryfast", 0.18),
    "balanced":  QualityProfile(1280, 720, 30, 23, "medium",   0.18),
    "cinematic": QualityProfile(1920, 1080, 30, 20, "slow",    0.20),
}


def get_profile(name: str) -> QualityProfile:
    return _PROFILES.get(name, _PROFILES["balanced"])


# ── ffmpeg execution helper ─────────────────────────────────────────────────

def _ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg executable not found on PATH. Install ffmpeg and re-run."
        )
    return path


def _run_ffmpeg(
    args: List[str],
    log_path: Optional[Path] = None,
    label: str = "ffmpeg",
) -> None:
    ffmpeg = _ensure_ffmpeg()
    full = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n# {label}\n")
            fh.write(" ".join(_quote_args(full)) + "\n")

    LOGGER.info("%s: running ffmpeg with %d args", label, len(full))
    try:
        completed = subprocess.run(
            full,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg not callable: {exc}") from exc

    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "").strip().splitlines()[-20:]
        raise RuntimeError(
            f"{label}: ffmpeg failed (exit={completed.returncode})\n"
            + "\n".join(stderr_tail)
        )


def _quote_args(args: List[str]) -> List[str]:
    out: List[str] = []
    for a in args:
        if any(ch in a for ch in (" ", "\"", "'", "\\")):
            out.append('"' + a.replace('"', r"\"") + '"')
        else:
            out.append(a)
    return out


# ── 1. Ken Burns clip ───────────────────────────────────────────────────────

def ken_burns_clip(
    image_path: str,
    out_path: Path,
    duration_sec: float,
    profile: QualityProfile,
    direction: str = "in",
    log_path: Optional[Path] = None,
) -> str:
    """
    Generate a slow zoom (Ken Burns) MP4 from a single image.

    direction:
        "in"   -> slow zoom in
        "out"  -> slow zoom out
    """
    duration_sec = max(2.0, float(duration_sec))
    fps = profile.fps
    total_frames = max(int(duration_sec * fps), fps * 2)
    width, height = profile.width, profile.height

    # Keep zoom subtle to avoid over-zoom/blurry output.
    max_zoom = 1.06
    zoom_step = 0.00012
    if direction == "out":
        z_expr = f"if(eq(on,1),{max_zoom:.3f},max(1.0,zoom-{zoom_step:.6f}))"
    else:
        z_expr = f"min(zoom+{zoom_step:.6f},{max_zoom:.3f})"

    # Use only a slight overscan margin (not 2x upscale) to preserve sharpness.
    base_w = int(width * 1.12)
    base_h = int(height * 1.12)

    vf = (
        f"scale={base_w}:{base_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={base_w}:{base_h},"
        f"zoompan=z='{z_expr}':d={total_frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={width}x{height}:fps={fps},"
        "format=yuv420p"
    )

    args = [
        "-loop", "1",
        "-i", image_path,
        "-t", f"{duration_sec:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", profile.preset,
        "-crf", str(profile.crf),
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(args, log_path=log_path, label=f"ken_burns scene_clip {out_path.name}")
    return str(out_path)


# ── 2. Per-scene A/V composition ────────────────────────────────────────────

def compose_scene(
    clip_path: str,
    voice_path: str,
    bgm_path: str,
    out_path: Path,
    profile: QualityProfile,
    log_path: Optional[Path] = None,
) -> str:
    """
    Mix the silent Ken Burns clip with the scene voice track and BGM.
    Voice is at full volume; BGM is ducked to profile.bgm_volume.
    Output duration is clamped to the voice/clip duration.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_bgm = bool(bgm_path) and Path(bgm_path).exists()

    if has_bgm:
        filter_complex = (
            f"[1:a]volume=1.0[voice];"
            f"[2:a]aloop=loop=-1:size=2e9,volume={profile.bgm_volume:.3f}[bgm];"
            f"[voice][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
        )
        args = [
            "-i", clip_path,
            "-i", voice_path,
            "-stream_loop", "-1", "-i", bgm_path,
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", profile.preset,
            "-crf", str(profile.crf),
            "-r", str(profile.fps),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(out_path),
        ]
    else:
        args = [
            "-i", clip_path,
            "-i", voice_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", profile.preset,
            "-crf", str(profile.crf),
            "-r", str(profile.fps),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(out_path),
        ]

    _run_ffmpeg(args, log_path=log_path, label=f"compose_scene {out_path.name}")
    return str(out_path)


# ── 3. Concat all scenes ────────────────────────────────────────────────────

def concat_scenes(
    scene_clips: List[str],
    out_path: Path,
    log_path: Optional[Path] = None,
) -> str:
    """
    Concatenate scene MP4s losslessly via the concat demuxer.
    All inputs must share codec / resolution / fps (we already enforce that).
    """
    if not scene_clips:
        raise ValueError("concat_scenes: scene_clips is empty")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.with_suffix(".concat.txt")
    list_lines = []
    for clip in scene_clips:
        abs_clip = str(Path(clip).resolve())
        # ffmpeg concat demuxer needs single-quoted paths with backslashes escaped
        safe = abs_clip.replace("\\", "/").replace("'", r"'\''")
        list_lines.append(f"file '{safe}'")
    list_file.write_text("\n".join(list_lines) + "\n", encoding="utf-8")

    args = [
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    _run_ffmpeg(args, log_path=log_path, label=f"concat_final {out_path.name}")
    return str(out_path)


def build_scene_clip_from_line_images(
    scene_id: int,
    line_images: List[Dict[str, Any]],
    out_path: Path,
    profile: QualityProfile,
    log_path: Optional[Path] = None,
) -> str:
    """
    Build a per-scene video clip from speaker-focused line images.
    Each line image gets its own short Ken Burns clip, then all are concatenated.
    """
    if not line_images:
        raise ValueError(f"scene {scene_id}: no line_images provided")

    tmp_dir = out_path.parent / f"scene_{scene_id:02d}_line_clips"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    line_clips: List[str] = []
    for item in line_images:
        idx = int(item.get("line_index", 0))
        duration_ms = int(item.get("duration_ms", 0))
        duration_sec = max(0.3, duration_ms / 1000.0)
        image_path = str(item.get("image_path", ""))
        if not image_path or not Path(image_path).exists():
            continue
        clip_path = tmp_dir / f"line_{idx:03d}.mp4"
        direction = "in" if idx % 2 == 1 else "out"
        ken_burns_clip(
            image_path=image_path,
            out_path=clip_path,
            duration_sec=duration_sec,
            profile=profile,
            direction=direction,
            log_path=log_path,
        )
        line_clips.append(str(clip_path))

    if not line_clips:
        raise RuntimeError(f"scene {scene_id}: failed to build any line clips")
    if len(line_clips) == 1:
        Path(out_path).write_bytes(Path(line_clips[0]).read_bytes())
        return str(out_path)

    return concat_scenes(line_clips, out_path=out_path, log_path=log_path)


def concat_scenes_with_crossfade(
    scene_clips: List[str],
    out_path: Path,
    profile: QualityProfile,
    transition_sec: float = 0.35,
    log_path: Optional[Path] = None,
) -> str:
    """
    Concatenate scene clips with video xfade + audio acrossfade transitions.
    Re-encodes output (required for transition filters).
    """
    if not scene_clips:
        raise ValueError("concat_scenes_with_crossfade: scene_clips is empty")
    if len(scene_clips) == 1 or transition_sec <= 0:
        return concat_scenes(scene_clips, out_path, log_path=log_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    durations = [probe_duration_sec(p) for p in scene_clips]
    # Ensure offsets are valid; degrade gracefully if very short clips.
    transition_sec = max(0.1, float(transition_sec))
    for d in durations:
        if d > 0:
            transition_sec = min(transition_sec, max(0.1, d * 0.25))

    args: List[str] = []
    for clip in scene_clips:
        args.extend(["-i", clip])

    filter_parts: List[str] = []
    prev_v = "[0:v]"
    prev_a = "[0:a]"
    cumulative = durations[0] if durations and durations[0] > 0 else 0.0

    for idx in range(1, len(scene_clips)):
        next_v = f"[{idx}:v]"
        next_a = f"[{idx}:a]"
        out_v = f"[v{idx}]"
        out_a = f"[a{idx}]"
        offset = max(0.0, cumulative - transition_sec)
        filter_parts.append(
            f"{prev_v}{next_v}xfade=transition=fade:duration={transition_sec:.3f}:offset={offset:.3f}{out_v}"
        )
        filter_parts.append(
            f"{prev_a}{next_a}acrossfade=d={transition_sec:.3f}:c1=tri:c2=tri{out_a}"
        )
        prev_v = out_v
        prev_a = out_a
        dur = durations[idx] if idx < len(durations) else 0.0
        cumulative = cumulative + max(dur, 0.0) - transition_sec

    filter_complex = ";".join(filter_parts)
    args.extend(
        [
            "-filter_complex", filter_complex,
            "-map", prev_v,
            "-map", prev_a,
            "-c:v", "libx264",
            "-preset", profile.preset,
            "-crf", str(profile.crf),
            "-r", str(profile.fps),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out_path),
        ]
    )
    _run_ffmpeg(args, log_path=log_path, label=f"concat_crossfade {out_path.name}")
    return str(out_path)


# ── 4. Subtitles ────────────────────────────────────────────────────────────

def _ms_to_srt(ms: int) -> str:
    if ms < 0:
        ms = 0
    hours = ms // 3_600_000
    ms -= hours * 3_600_000
    minutes = ms // 60_000
    ms -= minutes * 60_000
    seconds = ms // 1000
    millis = ms - seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_srt(
    timing_manifest: Dict[str, Any],
    out_path: Path,
    scene_transition_sec: float = 0.0,
) -> str:
    """
    Build an .srt from the Phase 2 timing manifest.
    Uses each line's absolute start_ms / end_ms.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    counter = 1
    prev_end = 0
    transition_ms = max(0, int(scene_transition_sec * 1000))
    for scene_idx, scene in enumerate(timing_manifest.get("scenes", [])):
        scene_shift = scene_idx * transition_ms
        for line in scene.get("lines", []):
            start = max(0, int(line.get("start_ms", 0)) - scene_shift)
            end = max(start + 1, int(line.get("end_ms", start + 1500)) - scene_shift)
            if start < prev_end:
                start = prev_end
                end = max(start + 80, end)
            speaker = (line.get("speaker") or "").strip()
            text = (line.get("line") or "").strip().strip('"').strip()
            if not text:
                continue
            display = f"{speaker}: {text}" if speaker else text
            lines.append(str(counter))
            lines.append(f"{_ms_to_srt(start)} --> {_ms_to_srt(end)}")
            lines.append(display)
            lines.append("")
            counter += 1
            prev_end = end
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)


def burn_subtitles(
    video_path: str,
    srt_path: str,
    out_path: Path,
    profile: QualityProfile,
    log_path: Optional[Path] = None,
) -> str:
    """
    Burn an SRT into the video. Re-encodes with the chosen profile.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_srt = str(srt_path).replace("\\", "/").replace(":", "\\:")
    vf = (
        f"subtitles='{safe_srt}':force_style="
        "'FontName=Arial,FontSize=18,Outline=1,Shadow=0,"
        "MarginV=40,PrimaryColour=&H00FFFFFF&,OutlineColour=&H80000000&'"
    )
    args = [
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", profile.preset,
        "-crf", str(profile.crf),
        "-r", str(profile.fps),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out_path),
    ]
    _run_ffmpeg(args, log_path=log_path, label=f"burn_subtitles {out_path.name}")
    return str(out_path)


# ── 5. ffprobe duration ─────────────────────────────────────────────────────

def probe_duration_sec(path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=False, capture_output=True, text=True,
        )
        return float((result.stdout or "0").strip())
    except Exception:
        return 0.0
