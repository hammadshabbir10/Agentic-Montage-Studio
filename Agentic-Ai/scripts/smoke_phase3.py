"""
smoke_phase3.py  –  End-to-end Phase 3 smoke test.

Runs Phase 3 (assumes Phase 1 + 2 outputs already exist) and verifies that
every required artifact is produced. Exits non-zero on any missing file.

Usage:
    python scripts/smoke_phase3.py
    python scripts/smoke_phase3.py --backend pollinations --quality fast
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run_phase3(args_extra: list[str]) -> None:
    cmd = [sys.executable, "-m", "src.main_phase3", *args_extra]
    print(f"[smoke] running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(ROOT))
    if completed.returncode != 0:
        sys.exit(f"[smoke] FAIL: phase3 exited with {completed.returncode}")


def check_artifact(path: Path, label: str, min_size: int = 1) -> None:
    if not path.exists():
        sys.exit(f"[smoke] FAIL: missing {label}: {path}")
    if path.stat().st_size < min_size:
        sys.exit(f"[smoke] FAIL: {label} too small (<{min_size}B): {path}")
    print(f"[smoke] OK   {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="pollinations")
    parser.add_argument("--quality", default="fast")
    args = parser.parse_args()

    run_phase3(["--backend", args.backend, "--quality", args.quality])

    counter_path = ROOT / "data" / "phase3_run_counter.json"
    if not counter_path.exists():
        sys.exit("[smoke] FAIL: phase3_run_counter.json missing")

    last_run = int(json.loads(counter_path.read_text())["last_run"])
    if last_run < 1:
        # Phase 3 may have aligned to a Phase 2 run, check phase3_runs
        any_runs = sorted((ROOT / "data" / "phase3_runs").glob("run*"))
        if not any_runs:
            sys.exit("[smoke] FAIL: no phase3_runs/run* directory found")
        run_dir = any_runs[-1]
    else:
        run_dir = ROOT / "data" / "phase3_runs" / f"run{last_run:02d}"

    tag = run_dir.name
    check_artifact(run_dir / f"final_output_{tag}.mp4", "final video", min_size=10_000)
    check_artifact(run_dir / f"phase3_state_{tag}.json", "phase3 state")
    check_artifact(run_dir / f"phase3_outputs_{tag}.json", "phase3 outputs")
    check_artifact(run_dir / f"image_prompts_{tag}.json", "image prompts")
    check_artifact(run_dir / f"ffmpeg_commands_{tag}.log", "ffmpeg log")
    images = list((run_dir / "images").glob("*.png"))
    if not images:
        sys.exit("[smoke] FAIL: no images generated")
    print(f"[smoke] OK   {len(images)} scene image(s) generated")
    print("\n[smoke] PASS — Phase 3 produced all required artifacts.")


if __name__ == "__main__":
    main()
