"""
timing_manifest.py  –  Phase 2 Timing Utility
Builds a timing_manifest.json from audio results produced by voice_synthesizer.

Output schema
-------------
{
  "workflow_id": "...",
  "timestamp": "...",
  "scenes": [
    {
      "scene_id": 1,
      "audio_file": "data/phase2_runs/run01/audio/scene_01_run01.mp3",
      "bgm_file":   "data/bgm_library/tense_01.mp3",
      "mood":       "tense",
      "start_ms":   0,
      "end_ms":     8400,
      "duration_ms": 8400,
      "lines": [
        {
          "speaker":    "JACK",
          "voice":      "en-US-GuyNeural",
          "line":       "We need to get out...",
          "start_ms":   0,
          "end_ms":     2200,
          "duration_ms": 2200
        },
        ...
      ]
    },
    ...
  ]
}

Timing is estimated from byte offsets because MP3 is CBR-ish.
A better approach would be to use mutagen/pydub to read actual duration,
but we keep it dependency-light here.  The estimate is:
    duration_ms ≈ (byte_length / total_bytes) * scene_total_duration_ms
where scene_total_duration_ms is read from the MP3 file with a fallback
of 300 ms per character.
"""

import datetime
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from src.io.pydantic_schemas import validate_timing_manifest_payload


def _probe_duration_ms(path: str) -> int:
    """Read media duration with ffprobe; fallback to file-size estimate."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            duration_s = float((result.stdout or "0").strip() or "0")
            if duration_s > 0:
                return int(duration_s * 1000)
        except Exception:
            pass

    # Last resort fallback for environments without ffprobe.
    try:
        size = Path(path).stat().st_size
        return int(size / 16000 * 1000)
    except Exception:
        return 3000


def build(
    audio_results: List[Dict],
    music_results: List[Dict],
    run_tag: str,
    out_dir: str,
) -> str:
    """
    Parameters
    ----------
    audio_results : list of { scene_id, path, segments }  from voice_synthesizer
    music_results : list of { scene_id, mood, bgm_path }  from music_selector
    run_tag       : e.g. "run01"
    out_dir       : directory to write timing_manifest.json

    Returns
    -------
    Path to the written JSON file.
    """
    music_map = {m["scene_id"]: m for m in music_results}

    scenes_out = []
    cursor_ms = 0

    for result in sorted(audio_results, key=lambda r: r["scene_id"]):
        scene_id = result["scene_id"]
        audio_file = result.get("path", "")
        segments = result.get("segments", [])

        scene_duration_ms = _probe_duration_ms(audio_file)

        music_info = music_map.get(scene_id, {})

        lines_out = []
        line_cursor_ms = cursor_ms
        line_durations = []
        for seg in segments:
            seg_audio_path = seg.get("audio_file", "")
            seg_ms = _probe_duration_ms(seg_audio_path) if seg_audio_path else 0
            if seg_ms <= 0:
                seg_ms = 800
            line_durations.append(seg_ms)

        summed_line_ms = sum(line_durations)
        if summed_line_ms <= 0:
            summed_line_ms = 1
        scale = scene_duration_ms / summed_line_ms

        for idx, seg in enumerate(segments):
            seg_ms = max(200, int(line_durations[idx] * scale))
            lines_out.append({
                "speaker":     seg.get("speaker", ""),
                "voice":       seg.get("voice", ""),
                "line":        seg.get("line", ""),
                "visual_cue":  seg.get("visual_cue", ""),
                "start_ms":    line_cursor_ms,
                "end_ms":      line_cursor_ms + seg_ms,
                "duration_ms": seg_ms,
            })
            line_cursor_ms += seg_ms

        # Force final line end to match exact scene end for perfect subtitle sync.
        if lines_out:
            scene_end = cursor_ms + scene_duration_ms
            lines_out[-1]["end_ms"] = scene_end
            lines_out[-1]["duration_ms"] = max(
                200,
                scene_end - lines_out[-1]["start_ms"],
            )

        scenes_out.append({
            "scene_id":    scene_id,
            "audio_file":  audio_file,
            "bgm_file":    music_info.get("bgm_path", ""),
            "mood":        music_info.get("mood", "neutral"),
            "start_ms":    cursor_ms,
            "end_ms":      cursor_ms + scene_duration_ms,
            "duration_ms": scene_duration_ms,
            "lines":       lines_out,
        })

        cursor_ms += scene_duration_ms

    manifest = {
        "workflow_id": f"phase2_{run_tag}",
        "timestamp":   datetime.datetime.now().isoformat(),
        "run_tag":     run_tag,
        "total_duration_ms": cursor_ms,
        "scenes": scenes_out,
    }
    manifest = validate_timing_manifest_payload(manifest)

    out_path = Path(out_dir) / f"timing_manifest_{run_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(out_path)