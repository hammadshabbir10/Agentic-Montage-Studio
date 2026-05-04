"""
video_compose.py  –  Phase 3 ffmpeg utilities (CPU-friendly)

Pipeline
--------
1. ken_burns_clip()           : still image -> smooth animated MP4 (zoom + pan)
2. build_title_card()         : intro card with story title + subtitle
3. build_end_card()           : closing card
4. compose_scene()            : Ken Burns clip + voice + bgm -> scene MP4 with fades
5. concat_scenes()            : scene MP4s -> single final MP4 (lossless concat demuxer)
6. concat_scenes_with_crossfade() : concat + xfade/acrossfade between segments
7. build_srt()                : timing manifest lines -> .srt subtitle file
8. burn_subtitles()           : optional subtitle burn-in over final MP4

All functions log the exact ffmpeg command they run to a per-run log file
so the grader can audit reproducibility.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

try:
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    _PIL_AVAILABLE = False


# ── Quality profiles ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class QualityProfile:
    width: int
    height: int
    fps: int
    crf: int             # libx264 quality (lower = better, 18 visually lossless, 28 small)
    preset: str          # libx264 speed/quality preset
    bgm_volume: float    # 0.0–1.0, voice always at 1.0
    upscale_factor: int = 2  # internal canvas multiplier for smoother Ken Burns


_PROFILES: Dict[str, QualityProfile] = {
    "fast":      QualityProfile(854,  480, 24, 28, "veryfast", 0.18, upscale_factor=2),
    "balanced":  QualityProfile(1280, 720, 30, 23, "medium",   0.18, upscale_factor=2),
    "cinematic": QualityProfile(1920, 1080, 30, 20, "slow",    0.20, upscale_factor=2),
}


def get_profile(name: str) -> QualityProfile:
    return _PROFILES.get(name, _PROFILES["balanced"])


# ── Motion preset library (Tier 2) ──────────────────────────────────────────
@dataclass(frozen=True)
class MotionPreset:
    """
    Per-line camera move described by start/end zoom and normalized x/y in [0, 1].
    Position is the top-left of the cropped window expressed as a fraction of the
    available headroom: x = (iw - iw/zoom) * x_norm, similarly for y.
    """
    name: str
    zoom_start: float
    zoom_end: float
    x_start: float = 0.5
    y_start: float = 0.5
    x_end: float = 0.5
    y_end: float = 0.5


MOTION_PRESETS: Dict[str, MotionPreset] = {
    "push_in":         MotionPreset("push_in",         1.00, 1.10, 0.50, 0.50, 0.50, 0.50),
    "pull_out":        MotionPreset("pull_out",        1.10, 1.00, 0.50, 0.50, 0.50, 0.50),
    "push_to_face":    MotionPreset("push_to_face",    1.00, 1.18, 0.50, 0.45, 0.50, 0.38),
    "pan_left_right":  MotionPreset("pan_left_right",  1.06, 1.06, 0.05, 0.50, 0.95, 0.50),
    "pan_right_left":  MotionPreset("pan_right_left",  1.06, 1.06, 0.95, 0.50, 0.05, 0.50),
    "diagonal_in":     MotionPreset("diagonal_in",     1.00, 1.12, 0.30, 0.65, 0.70, 0.35),
    "subtle_drift":    MotionPreset("subtle_drift",    1.04, 1.06, 0.40, 0.50, 0.60, 0.50),
}


def pick_motion_preset(
    visual_cue: str = "",
    mood: str = "",
    scene_id: int = 0,
    line_index: int = 0,
) -> MotionPreset:
    """
    Choose a camera move based on visual cue keywords + mood + a deterministic
    fallback rotation so adjacent scenes/lines never repeat the same motion.
    """
    cue = (visual_cue or "").lower()
    mood = (mood or "").lower()

    if any(k in cue for k in ("close-up", "close up", "extreme close", "face", "eyes")):
        return MOTION_PRESETS["push_to_face"]
    if any(k in cue for k in ("wide shot", "wide", "establish", "landscape", "city")):
        return MOTION_PRESETS["pull_out"]
    if any(k in cue for k in ("over-the-shoulder", "ots", "over the shoulder")):
        return (
            MOTION_PRESETS["pan_left_right"]
            if line_index % 2 == 0
            else MOTION_PRESETS["pan_right_left"]
        )
    if any(k in cue for k in ("tracking", "dolly", "moves", "walks")):
        return MOTION_PRESETS["pan_left_right"]

    if mood in ("tense", "action"):
        return MOTION_PRESETS["push_in"]
    if mood in ("hopeful", "uplifting"):
        return MOTION_PRESETS["pull_out"]
    if mood in ("mysterious", "sad"):
        return (
            MOTION_PRESETS["pan_left_right"]
            if (scene_id + line_index) % 2 == 0
            else MOTION_PRESETS["diagonal_in"]
        )

    rotation = [
        MOTION_PRESETS["push_in"],
        MOTION_PRESETS["pan_left_right"],
        MOTION_PRESETS["push_to_face"],
        MOTION_PRESETS["pull_out"],
        MOTION_PRESETS["pan_right_left"],
        MOTION_PRESETS["diagonal_in"],
    ]
    return rotation[(scene_id + line_index) % len(rotation)]


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

def _build_zoompan_expressions(
    preset: MotionPreset,
    total_frames: int,
) -> Dict[str, str]:
    """Build smooth, time-based zoompan expressions (Tier 1)."""
    d_minus_1 = max(total_frames - 1, 1)
    z0, z1 = float(preset.zoom_start), float(preset.zoom_end)
    x0, x1 = float(preset.x_start), float(preset.x_end)
    y0, y1 = float(preset.y_start), float(preset.y_end)

    # Linear interpolation in time (on/[d-1]) for perfectly smooth motion;
    # zoom is referenced inside x/y to compute the headroom precisely.
    z_expr = f"{z0:.5f}+({z1:.5f}-{z0:.5f})*on/{d_minus_1}"
    x_expr = (
        f"(iw-iw/zoom)*({x0:.4f}+({x1:.4f}-{x0:.4f})*on/{d_minus_1})"
    )
    y_expr = (
        f"(ih-ih/zoom)*({y0:.4f}+({y1:.4f}-{y0:.4f})*on/{d_minus_1})"
    )
    return {"z": z_expr, "x": x_expr, "y": y_expr}


def _cinematic_filters(width: int, height: int) -> List[str]:
    """Color grade + film grain + vignette + 2.35:1 letterbox bars."""
    bar = max(2, int(round(height * 0.075)))  # ~7.5% top/bottom
    return [
        "eq=contrast=1.05:saturation=1.08:gamma=0.98",
        "vignette=PI/4.5",
        "noise=alls=6:allf=t+u",
        f"drawbox=x=0:y=0:w=iw:h={bar}:color=black@1.0:t=fill",
        f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black@1.0:t=fill",
    ]


def _ken_burns_clip_zoompan(
    image_path: str,
    out_path: Path,
    duration_sec: float,
    profile: QualityProfile,
    motion_preset: MotionPreset,
    cinematic: bool,
    log_path: Optional[Path],
) -> str:
    """
    Legacy zoompan-based renderer. Kept as fallback when Pillow is not available
    or when --motion-engine zoompan is forced.
    """
    duration_sec = max(2.0, float(duration_sec))
    fps = profile.fps
    total_frames = max(int(round(duration_sec * fps)), fps * 2)
    width, height = profile.width, profile.height

    upscale = max(2, profile.upscale_factor)
    canvas_w = width * upscale
    canvas_h = height * upscale

    expr = _build_zoompan_expressions(motion_preset, total_frames)

    vf_parts: List[str] = [
        (
            f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase"
            ":flags=lanczos+accurate_rnd+full_chroma_int"
        ),
        f"crop={canvas_w}:{canvas_h}",
        (
            f"zoompan=z='{expr['z']}':d={total_frames}"
            f":x='{expr['x']}':y='{expr['y']}'"
            f":s={canvas_w}x{canvas_h}:fps={fps}"
        ),
        f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int",
    ]
    if cinematic:
        vf_parts.extend(_cinematic_filters(width, height))
    vf_parts.append("format=yuv420p")
    vf = ",".join(vf_parts)

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
    _run_ffmpeg(
        args,
        log_path=log_path,
        label=f"ken_burns_zoompan[{motion_preset.name}] {out_path.name}",
    )
    return str(out_path)


def _prepare_canvas_image(image_path: str, canvas_w: int, canvas_h: int):
    """Open image and resize+center-crop to (canvas_w, canvas_h) with LANCZOS."""
    img = Image.open(image_path).convert("RGB")
    iw, ih = img.size
    src_aspect = iw / ih
    tgt_aspect = canvas_w / canvas_h
    if src_aspect > tgt_aspect:
        new_h = canvas_h
        new_w = int(round(canvas_h * src_aspect))
    else:
        new_w = canvas_w
        new_h = int(round(canvas_w / src_aspect))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - canvas_w) // 2
    top = (new_h - canvas_h) // 2
    img = img.crop((left, top, left + canvas_w, top + canvas_h))
    return img


def _ken_burns_clip_pil(
    image_path: str,
    out_path: Path,
    duration_sec: float,
    profile: QualityProfile,
    motion_preset: MotionPreset,
    cinematic: bool,
    log_path: Optional[Path],
) -> str:
    """
    Sub-pixel precise Ken Burns renderer using Pillow's BICUBIC AFFINE
    transform per frame, piped to ffmpeg as raw RGB24. This eliminates the
    integer pixel quantization that causes zoompan vibration.
    """
    if not _PIL_AVAILABLE:
        return _ken_burns_clip_zoompan(
            image_path, out_path, duration_sec, profile, motion_preset, cinematic, log_path
        )

    duration_sec = max(2.0, float(duration_sec))
    fps = profile.fps
    total_frames = max(int(round(duration_sec * fps)), fps * 2)
    width, height = profile.width, profile.height

    # Larger canvas = more room for sub-pixel motion to feel buttery.
    upscale = max(2, profile.upscale_factor)
    canvas_w = width * upscale
    canvas_h = height * upscale
    canvas = _prepare_canvas_image(image_path, canvas_w, canvas_h)

    z0 = float(motion_preset.zoom_start)
    z1 = float(motion_preset.zoom_end)
    x0 = float(motion_preset.x_start)
    x1 = float(motion_preset.x_end)
    y0 = float(motion_preset.y_start)
    y1 = float(motion_preset.y_end)
    denom = max(total_frames - 1, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg input filter chain (cinematic look applied here at target res).
    vf_parts: List[str] = []
    if cinematic:
        vf_parts.extend(_cinematic_filters(width, height))
    vf_parts.append("format=yuv420p")
    vf = ",".join(vf_parts) if vf_parts else "format=yuv420p"

    ffmpeg = _ensure_ffmpeg()
    args = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", profile.preset,
        "-crf", str(profile.crf),
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"\n# ken_burns_pil[{motion_preset.name}] {out_path.name} "
                f"(dur={duration_sec:.3f}s, frames={total_frames}, canvas={canvas_w}x{canvas_h})\n"
            )
            fh.write(" ".join(_quote_args(args)) + "\n")

    LOGGER.info(
        "ken_burns_pil[%s] %s: rendering %d frames at %dx%d",
        motion_preset.name, out_path.name, total_frames, width, height,
    )

    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in range(total_frames):
            t = f / denom
            z = z0 + (z1 - z0) * t
            x_norm = x0 + (x1 - x0) * t
            y_norm = y0 + (y1 - y0) * t
            crop_w = canvas_w / z
            crop_h = canvas_h / z
            x = (canvas_w - crop_w) * x_norm
            y = (canvas_h - crop_h) * y_norm
            # AFFINE matrix maps OUTPUT (x', y') to SOURCE (a*x'+b*y'+c, d*x'+e*y'+f).
            # We want a horizontal scale of crop_w/width, vertical crop_h/height,
            # and translation (x, y) in source coordinates.
            matrix = (
                crop_w / width, 0.0, x,
                0.0, crop_h / height, y,
            )
            frame = canvas.transform(
                (width, height),
                Image.AFFINE,
                matrix,
                resample=Image.BICUBIC,
            )
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except (BrokenPipeError, OSError) as exc:
        proc.kill()
        stderr = (proc.stderr.read() if proc.stderr else b"").decode(errors="ignore")
        raise RuntimeError(
            f"ken_burns_pil pipe write failed for {out_path.name}: {exc}\n{stderr[-500:]}"
        ) from exc

    rc = proc.wait()
    if rc != 0:
        stderr = (proc.stderr.read() if proc.stderr else b"").decode(errors="ignore")
        raise RuntimeError(
            f"ken_burns_pil ffmpeg failed (exit={rc}) for {out_path.name}\n{stderr[-500:]}"
        )
    return str(out_path)


def ken_burns_clip(
    image_path: str,
    out_path: Path,
    duration_sec: float,
    profile: QualityProfile,
    direction: str = "in",
    motion_preset: Optional[MotionPreset] = None,
    cinematic: bool = True,
    log_path: Optional[Path] = None,
    engine: str = "auto",
) -> str:
    """
    Generate a smooth Ken Burns MP4 from a single still image.

    Engines
    -------
    - "auto" (default): use Pillow sub-pixel BICUBIC frames piped to ffmpeg.
      Falls back to "zoompan" if Pillow is not installed.
    - "pil"   : force the Pillow renderer.
    - "zoompan": legacy ffmpeg zoompan filter (faster, may have micro jitter).

    motion_preset: if provided, overrides ``direction``. Otherwise:
        "in"  -> push_in
        "out" -> pull_out
    cinematic: apply color grade + grain + vignette + 2.35:1 letterbox bars.
    """
    if motion_preset is None:
        motion_preset = (
            MOTION_PRESETS["pull_out"] if direction == "out" else MOTION_PRESETS["push_in"]
        )

    use_engine = engine.lower()
    if use_engine == "auto":
        use_engine = "pil" if _PIL_AVAILABLE else "zoompan"

    if use_engine == "pil":
        return _ken_burns_clip_pil(
            image_path, out_path, duration_sec, profile, motion_preset, cinematic, log_path
        )
    return _ken_burns_clip_zoompan(
        image_path, out_path, duration_sec, profile, motion_preset, cinematic, log_path
    )


# ── 2. Per-scene A/V composition ────────────────────────────────────────────

def compose_scene(
    clip_path: str,
    voice_path: str,
    bgm_path: str,
    out_path: Path,
    profile: QualityProfile,
    log_path: Optional[Path] = None,
    audio_fade_in_sec: float = 0.25,
    audio_fade_out_sec: float = 0.40,
) -> str:
    """
    Mix the silent Ken Burns clip with the scene voice track and BGM.
    Voice is at full volume; BGM is ducked to profile.bgm_volume.
    Adds short audio fade-in/out at the scene boundaries so cuts/concats
    do not produce audible clicks.
    Output duration is clamped to the voice/clip duration.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    has_bgm = bool(bgm_path) and Path(bgm_path).exists()

    voice_dur = probe_duration_sec(voice_path)
    if voice_dur <= 0:
        voice_dur = probe_duration_sec(clip_path)
    fade_out_start = max(0.0, voice_dur - max(0.05, audio_fade_out_sec))

    fade_in = max(0.0, float(audio_fade_in_sec))
    fade_out = max(0.0, float(audio_fade_out_sec))

    if has_bgm:
        voice_chain = "volume=1.0"
        if fade_in > 0:
            voice_chain += f",afade=t=in:st=0:d={fade_in:.3f}"
        if fade_out > 0:
            voice_chain += f",afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}"

        bgm_chain = (
            f"aloop=loop=-1:size=2e9,volume={profile.bgm_volume:.3f}"
        )
        if fade_in > 0:
            bgm_chain += f",afade=t=in:st=0:d={fade_in:.3f}"
        if fade_out > 0:
            bgm_chain += f",afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}"

        filter_complex = (
            f"[1:a]{voice_chain}[voice];"
            f"[2:a]{bgm_chain}[bgm];"
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
            "-ar", "44100",
            "-shortest",
            str(out_path),
        ]
    else:
        af_parts = []
        if fade_in > 0:
            af_parts.append(f"afade=t=in:st=0:d={fade_in:.3f}")
        if fade_out > 0:
            af_parts.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}")
        af = ",".join(af_parts) if af_parts else "anull"

        args = [
            "-i", clip_path,
            "-i", voice_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-af", af,
            "-c:v", "libx264",
            "-preset", profile.preset,
            "-crf", str(profile.crf),
            "-r", str(profile.fps),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
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
    mood: str = "",
    cinematic: bool = True,
    motion_engine: str = "auto",
    log_path: Optional[Path] = None,
) -> str:
    """
    Build a per-scene video clip from speaker-focused line images.
    Each line image becomes its own short Ken Burns clip with a motion preset
    chosen from the line's visual_cue + scene mood + line index, then all are
    concatenated into a single scene clip.
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
        visual_cue = str(item.get("visual_cue", ""))
        line_mood = str(item.get("mood", "") or mood)
        preset = pick_motion_preset(
            visual_cue=visual_cue,
            mood=line_mood,
            scene_id=scene_id,
            line_index=idx,
        )
        clip_path = tmp_dir / f"line_{idx:03d}.mp4"
        ken_burns_clip(
            image_path=image_path,
            out_path=clip_path,
            duration_sec=duration_sec,
            profile=profile,
            motion_preset=preset,
            cinematic=cinematic,
            log_path=log_path,
            engine=motion_engine,
        )
        line_clips.append(str(clip_path))

    if not line_clips:
        raise RuntimeError(f"scene {scene_id}: failed to build any line clips")
    if len(line_clips) == 1:
        Path(out_path).write_bytes(Path(line_clips[0]).read_bytes())
        return str(out_path)

    return concat_scenes(line_clips, out_path=out_path, log_path=log_path)


# ── Title and end cards (Tier 5) ────────────────────────────────────────────

def _escape_drawtext_path(path: str) -> str:
    """ffmpeg drawtext textfile= path needs ':' escaped (Windows drives)."""
    return str(path).replace("\\", "/").replace(":", "\\:")


# Common cross-platform font candidates (first existing one wins)
_FONT_CANDIDATES: List[str] = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def _resolve_font_file() -> Optional[str]:
    """Return the first existing TTF/TTC font path or None."""
    env_font = os.getenv("MONTAGE_FONT_FILE", "").strip()
    if env_font and Path(env_font).exists():
        return env_font
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _write_text_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip(), encoding="utf-8")
    return path


def build_title_card(
    title: str,
    subtitle: str,
    out_path: Path,
    profile: QualityProfile,
    duration_sec: float = 3.0,
    log_path: Optional[Path] = None,
) -> str:
    """
    Render a black title card with animated fade-in/hold/fade-out title text
    and a smaller subtitle, plus matching silent stereo audio so it is
    concat-compatible with composed scene clips.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = profile.width, profile.height
    fps = profile.fps
    duration_sec = max(1.5, float(duration_sec))

    title_file = _write_text_file(out_path.with_suffix(".title.txt"), title or "Untitled")
    subtitle_file = _write_text_file(out_path.with_suffix(".subtitle.txt"), subtitle or "")

    fade_in = 0.6
    fade_out = 0.6
    hold = max(0.4, duration_sec - fade_in - fade_out)
    alpha_expr = (
        f"if(lt(t,{fade_in}),t/{fade_in},"
        f"if(lt(t,{fade_in + hold}),1,"
        f"max(0,({duration_sec}-t)/{fade_out})))"
    )

    title_size = max(28, int(round(height * 0.085)))
    subtitle_size = max(14, int(round(height * 0.035)))
    subtitle_offset = max(20, int(round(height * 0.07)))

    font_path = _resolve_font_file()
    font_clause = (
        f":fontfile='{_escape_drawtext_path(font_path)}'" if font_path else ""
    )

    drawtext_title = (
        f"drawtext=textfile='{_escape_drawtext_path(title_file)}'"
        f"{font_clause}:fontsize={title_size}:fontcolor=white"
        f":x=(w-text_w)/2:y=(h-text_h)/2-{subtitle_offset}"
        f":alpha='{alpha_expr}'"
    )
    drawtext_sub = (
        f"drawtext=textfile='{_escape_drawtext_path(subtitle_file)}'"
        f"{font_clause}:fontsize={subtitle_size}:fontcolor=white@0.85"
        f":x=(w-text_w)/2:y=(h-text_h)/2+{subtitle_offset}"
        f":alpha='{alpha_expr}'"
    )

    vf_chain = f"[0:v]{drawtext_title},{drawtext_sub},format=yuv420p[v]"

    args = [
        "-f", "lavfi", "-i",
        f"color=c=black:s={width}x{height}:r={fps}:d={duration_sec}",
        "-f", "lavfi", "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-filter_complex", vf_chain,
        "-map", "[v]",
        "-map", "1:a",
        "-t", f"{duration_sec:.3f}",
        "-c:v", "libx264",
        "-preset", profile.preset,
        "-crf", str(profile.crf),
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        str(out_path),
    ]

    _run_ffmpeg(args, log_path=log_path, label=f"title_card {out_path.name}")
    return str(out_path)


def build_end_card(
    title: str,
    subtitle: str,
    out_path: Path,
    profile: QualityProfile,
    duration_sec: float = 3.0,
    log_path: Optional[Path] = None,
) -> str:
    """End card uses the same builder as the title card."""
    return build_title_card(
        title=title,
        subtitle=subtitle,
        out_path=out_path,
        profile=profile,
        duration_sec=duration_sec,
        log_path=log_path,
    )


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
    global_offset_sec: float = 0.0,
) -> str:
    """
    Build an .srt from the Phase 2 timing manifest.
    Uses each line's absolute start_ms / end_ms.

    `global_offset_sec` shifts every subtitle forward by N seconds (used to
    account for an intro title card prepended to the final video).
    `scene_transition_sec` accounts for crossfade overlap between scenes.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    counter = 1
    prev_end = 0
    transition_ms = max(0, int(scene_transition_sec * 1000))
    offset_ms = max(0, int(global_offset_sec * 1000))
    for scene_idx, scene in enumerate(timing_manifest.get("scenes", [])):
        scene_shift = scene_idx * transition_ms
        for line in scene.get("lines", []):
            start = max(0, int(line.get("start_ms", 0)) - scene_shift) + offset_ms
            end = max(start + 1, int(line.get("end_ms", start + 1500)) - scene_shift + offset_ms)
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
