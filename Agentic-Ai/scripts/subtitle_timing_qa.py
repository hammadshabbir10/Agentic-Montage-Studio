from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


SRT_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")


def _srt_ts_to_ms(ts: str) -> int:
    m = SRT_TS_RE.fullmatch(ts.strip())
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {ts!r}")
    hh, mm, ss, ms = [int(x) for x in m.groups()]
    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms


def _parse_srt(path: Path) -> List[Tuple[int, int, str]]:
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    entries: List[Tuple[int, int, str]] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        times = lines[1]
        if "-->" not in times:
            continue
        a, b = [x.strip() for x in times.split("-->", 1)]
        start = _srt_ts_to_ms(a)
        end = _srt_ts_to_ms(b)
        payload = " ".join(lines[2:]).strip()
        entries.append((start, end, payload))
    return entries


def _probe_duration_ms(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return int(float((result.stdout or "0").strip() or "0") * 1000)
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Subtitle timing QA checker")
    parser.add_argument("--timing", required=True, help="Path to timing_manifest_runXX.json")
    parser.add_argument("--srt", required=True, help="Path to subtitles_runXX.srt")
    parser.add_argument("--video", default="", help="Optional final mp4 to verify duration envelope")
    parser.add_argument("--max-delta-ms", type=int, default=450, help="Max acceptable drift per line")
    parser.add_argument(
        "--transition-sec",
        type=float,
        default=0.0,
        help="Scene transition overlap in seconds (used to compensate expected subtitle times)",
    )
    parser.add_argument(
        "--title-offset-sec",
        type=float,
        default=0.0,
        help="Intro title-card duration prepended to the final video (shifts expected subtitle times)",
    )
    parser.add_argument(
        "--end-card-sec",
        type=float,
        default=0.0,
        help="Closing end-card duration appended to the final video (extends expected video length)",
    )
    args = parser.parse_args()

    timing_path = Path(args.timing)
    srt_path = Path(args.srt)
    if not timing_path.exists() or not srt_path.exists():
        raise SystemExit("timing or srt file does not exist")

    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    srt_entries = _parse_srt(srt_path)
    timing_lines = []
    transition_ms = max(0, int(args.transition_sec * 1000))
    title_offset_ms = max(0, int(args.title_offset_sec * 1000))
    end_card_ms = max(0, int(args.end_card_sec * 1000))
    for scene_idx, scene in enumerate(timing.get("scenes", [])):
        scene_shift = scene_idx * transition_ms
        for line in scene.get("lines", []):
            timing_lines.append(
                (
                    max(0, int(line.get("start_ms", 0)) - scene_shift) + title_offset_ms,
                    max(0, int(line.get("end_ms", 0)) - scene_shift) + title_offset_ms,
                    str(line.get("line", "")).strip().strip('"'),
                )
            )

    issues: List[str] = []
    if len(srt_entries) != len(timing_lines):
        issues.append(
            f"Line count mismatch: SRT={len(srt_entries)} timing_manifest={len(timing_lines)}"
        )

    pairs = min(len(srt_entries), len(timing_lines))
    for i in range(pairs):
        s_start, s_end, _ = srt_entries[i]
        t_start, t_end, _ = timing_lines[i]
        if s_end <= s_start:
            issues.append(f"SRT line {i+1}: non-positive duration")
        if i > 0 and s_start < srt_entries[i - 1][1]:
            issues.append(f"SRT line {i+1}: overlaps previous subtitle")
        if abs(s_start - t_start) > args.max_delta_ms:
            issues.append(
                f"SRT line {i+1}: start drift {abs(s_start - t_start)}ms > {args.max_delta_ms}ms"
            )
        if abs(s_end - t_end) > args.max_delta_ms:
            issues.append(
                f"SRT line {i+1}: end drift {abs(s_end - t_end)}ms > {args.max_delta_ms}ms"
            )

    timing_total = int(timing.get("total_duration_ms", 0))
    effective_timing_total = max(
        0,
        timing_total - max(0, len(timing.get("scenes", [])) - 1) * transition_ms,
    )
    expected_subtitle_end = effective_timing_total + title_offset_ms
    extra_card_transitions = (1 if title_offset_ms > 0 else 0) + (1 if end_card_ms > 0 else 0)
    expected_video_total = (
        expected_subtitle_end + end_card_ms - extra_card_transitions * transition_ms
    )

    if srt_entries and expected_subtitle_end > 0:
        last_end = srt_entries[-1][1]
        if abs(last_end - expected_subtitle_end) > args.max_delta_ms:
            issues.append(
                f"Final subtitle end drift {abs(last_end - expected_subtitle_end)}ms vs expected end {expected_subtitle_end}ms"
            )

    if args.video:
        video_ms = _probe_duration_ms(Path(args.video))
        if video_ms > 0 and expected_video_total > 0:
            if video_ms + args.max_delta_ms < expected_video_total:
                issues.append(
                    f"Video shorter than expected: video={video_ms}ms expected={expected_video_total}ms"
                )

    print("Subtitle QA summary")
    print(f"- timing lines: {len(timing_lines)}")
    print(f"- srt lines   : {len(srt_entries)}")
    print(f"- timing total: {timing_total} ms")
    print(f"- effective total (after transition): {effective_timing_total} ms")
    print(f"- title offset: {title_offset_ms} ms     end-card: {end_card_ms} ms")
    print(f"- expected subtitle end: {expected_subtitle_end} ms")
    print(f"- expected video total : {expected_video_total} ms")
    if args.video:
        print(f"- video total : {_probe_duration_ms(Path(args.video))} ms")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    print("\nPASS: subtitle timing is consistent with timing manifest/video.")


if __name__ == "__main__":
    main()
