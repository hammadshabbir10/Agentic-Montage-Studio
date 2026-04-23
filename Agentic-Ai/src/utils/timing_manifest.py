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
from pathlib import Path
from typing import Dict, List


def _estimate_mp3_duration_ms(path: str) -> int:
    """
    Best-effort MP3 duration from file size.
    Assumes ~128 kbps CBR → 16 000 bytes/second.
    Falls back to 3000 ms if file is missing.
    """
    try:
        size = Path(path).stat().st_size
        return int(size / 16000 * 1000)  # 128 kbps
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

        scene_duration_ms = _estimate_mp3_duration_ms(audio_file)
        total_bytes = sum(s.get("byte_length", 1) for s in segments) or 1

        music_info = music_map.get(scene_id, {})

        lines_out = []
        line_cursor_ms = cursor_ms
        for seg in segments:
            frac = seg.get("byte_length", 1) / total_bytes
            seg_ms = max(200, int(frac * scene_duration_ms))
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

    out_path = Path(out_dir) / f"timing_manifest_{run_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(out_path)