from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from flask import Flask, Response, jsonify, request, send_file

from dotenv import load_dotenv

from src.mcp.tool_client import ToolClient
from src.mcp.tool_registry import ToolRegistry


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PHASE2_RUNS = DATA_DIR / "phase2_runs"
PHASE3_RUNS = DATA_DIR / "phase3_runs"
STATE_FILE = DATA_DIR / "phase4_web_state.json"

load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

app = Flask(__name__)

RUN_EVENTS: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}
RUN_STATUS: Dict[str, Dict[str, Any]] = {}
RUN_APPROVAL_EVENTS: Dict[str, threading.Event] = {}


def _safe_read_json(path: Path, fallback: Dict[str, Any] | List[Any] | None = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback if fallback is not None else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _emit(run_id: str, event: str, payload: Dict[str, Any]) -> None:
    item = {"event": event, "payload": payload, "ts": time.time()}
    RUN_EVENTS[run_id].put(item)
    if run_id in RUN_STATUS:
        if event == "phase_status":
            RUN_STATUS[run_id].setdefault("phases", {})[payload.get("phase", "unknown")] = payload
        elif event == "pipeline_status":
            RUN_STATUS[run_id]["status"] = payload.get("status", RUN_STATUS[run_id].get("status"))
            if payload.get("latest_video"):
                RUN_STATUS[run_id]["latest_video"] = payload["latest_video"]


def _pause_for_approval(run_id: str) -> bool:
    approval = threading.Event()
    RUN_APPROVAL_EVENTS[run_id] = approval
    _emit(run_id, "pipeline_status", {"status": "awaiting_approval", "message": "Phase 1 complete. Approve to continue to Phase 2 and 3."})
    approved = approval.wait(timeout=60 * 30)
    if not approved:
        _emit(run_id, "pipeline_status", {"status": "failed", "error": "Approval timeout waiting for human review."})
    return approved


def _latest_video_path() -> str:
    candidates = sorted(PHASE3_RUNS.glob("run*/final_output_run*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else ""


def _latest_phase2_timing() -> str:
    candidates = sorted(PHASE2_RUNS.glob("run*/timing_manifest_run*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else ""


def _last_run_tag(path_str: str) -> str:
    m = re.search(r"(run\d{2})", path_str or "")
    return m.group(1) if m else ""


def _ffmpeg_path_prefixes() -> List[str]:
    """
    Phase 3 uses shutil.which('ffmpeg'). GUI / IDE launches often have a smaller PATH
    than an interactive shell, so prepend dirs where ffmpeg is commonly installed.
    Optional: set FFMPEG_DIR (folder containing ffmpeg.exe) or FFMPEG_BIN.
    """
    dirs: List[str] = []
    seen: set[str] = set()

    def add(p: str) -> None:
        if not p:
            return
        expanded = os.path.expandvars(os.path.expanduser(p))
        if expanded and expanded not in seen:
            seen.add(expanded)
            dirs.append(expanded)

    for key in ("FFMPEG_BIN", "FFMPEG_DIR"):
        add(os.environ.get(key, ""))

    exe_hint = os.environ.get("IMAGEIO_FFMPEG_EXE", "").strip()
    if exe_hint:
        add(str(Path(exe_hint).resolve().parent))

    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    la = os.environ.get("LOCALAPPDATA", "")

    if os.name == "nt":
        add(r"C:\ffmpeg\bin")
        add(os.path.join(pf, "ffmpeg", "bin"))
        add(os.path.join(pf86, "ffmpeg", "bin"))
        add(os.path.join(pf, "Gyan", "ffmpeg", "bin"))
        if la:
            add(os.path.join(la, "Microsoft", "WinGet", "Links"))
            winget_base = os.path.join(la, "Microsoft", "WinGet", "Packages")
            if os.path.isdir(winget_base):
                try:
                    root = Path(winget_base)
                    for ffmpeg_exe in root.rglob("ffmpeg.exe"):
                        add(str(ffmpeg_exe.parent))
                        break
                except OSError:
                    pass
    else:
        add("/opt/homebrew/bin")
        add("/usr/local/bin")

    return dirs


def _phase_subprocess_env() -> Dict[str, str]:
    """
    Windows consoles often default to cp1252; Phase 2/3 log Unicode (e.g. arrows).
    Force UTF-8 for child stdio without touching pipeline code in Phases 1–3.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    path = env.get("PATH", "")
    extra = [d for d in _ffmpeg_path_prefixes() if d and Path(d).is_dir()]
    if extra:
        env["PATH"] = os.pathsep.join(extra + ([path] if path else []))
    return env


def _ffmpeg_probe_for_sse(run_id: str) -> None:
    """Tell the UI exactly which ffmpeg the Phase 4 child env resolves (debug aid)."""
    probe_env = _phase_subprocess_env()
    ffmpeg_path = shutil.which("ffmpeg", path=probe_env.get("PATH", ""))
    if ffmpeg_path:
        _emit(
            run_id,
            "phase_log",
            {"phase": "phase3", "line": f"[Phase 4] ffmpeg resolved -> {ffmpeg_path}"},
        )
    else:
        _emit(
            run_id,
            "phase_log",
            {
                "phase": "phase3",
                "line": (
                    "[Phase 4] WARNING: ffmpeg not found in augmented PATH after WinGet/GitHub installs. "
                    "Set FFMPEG_DIR to the folder containing ffmpeg.exe or restart Flask after installing FFmpeg."
                ),
            },
        )


def _run_command(run_id: str, phase_name: str, cmd: List[str]) -> int:
    if phase_name == "phase3":
        _ffmpeg_probe_for_sse(run_id)
    _emit(run_id, "phase_status", {"phase": phase_name, "status": "running", "command": " ".join(cmd)})
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=_phase_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        _emit(run_id, "phase_log", {"phase": phase_name, "line": line.rstrip()})
    code = process.wait()
    final = "completed" if code == 0 else "failed"
    _emit(run_id, "phase_status", {"phase": phase_name, "status": final, "exit_code": code})
    return code


def _phase1_cmd(prompt: str, scenes: int) -> List[str]:
    return [
        sys.executable,
        "-m",
        "src.main",
        "--mode",
        "auto",
        "--prompt",
        prompt,
        "--scenes",
        str(scenes),
        "--auto-approve",
    ]


def _phase2_cmd() -> List[str]:
    return [sys.executable, "-m", "src.main_phase2"]


def _phase3_cmd(quality: str, backend: str, subtitles: bool, scene_id: int | None) -> List[str]:
    cmd = [sys.executable, "-m", "src.main_phase3", "--quality", quality, "--backend", backend]
    if subtitles:
        cmd.append("--enable-subtitles")
    if scene_id is not None:
        cmd.extend(["--scene-id", str(scene_id)])
    return cmd


def _pipeline_worker(run_id: str, payload: Dict[str, Any]) -> None:
    try:
        prompt = payload["prompt"].strip()
        scenes = int(payload.get("scenes", 3))
        quality = payload.get("quality", "balanced")
        backend = payload.get("backend", "auto")
        subtitles = bool(payload.get("enable_subtitles", True))
        rerun_phase = payload.get("rerun_phase")
        scene_id = payload.get("scene_id")
        if scene_id is not None:
            scene_id = int(scene_id)

        _emit(run_id, "pipeline_status", {"status": "started", "run_id": run_id})
        sequence = []
        if rerun_phase:
            sequence = [rerun_phase]
        else:
            sequence = ["phase1", "phase2", "phase3"]

        for index, phase in enumerate(sequence):
            if phase == "phase1":
                code = _run_command(run_id, "phase1", _phase1_cmd(prompt, scenes))
                if code == 0 and payload.get("human_in_loop") and (not rerun_phase or rerun_phase == "phase1"):
                    approved = _pause_for_approval(run_id)
                    if not approved:
                        RUN_STATUS[run_id]["status"] = "failed"
                        return
            elif phase == "phase2":
                code = _run_command(run_id, "phase2", _phase2_cmd())
            elif phase == "phase3":
                code = _run_command(run_id, "phase3", _phase3_cmd(quality, backend, subtitles, scene_id))
            else:
                _emit(run_id, "pipeline_status", {"status": "failed", "error": f"Unknown phase: {phase}"})
                RUN_STATUS[run_id]["status"] = "failed"
                return

            if code != 0:
                RUN_STATUS[run_id]["status"] = "failed"
                _emit(run_id, "pipeline_status", {"status": "failed", "phase": phase, "exit_code": code})
                return

        latest_video = _latest_video_path()
        latest_timing = _latest_phase2_timing()
        run_tag = _last_run_tag(latest_video) or _last_run_tag(latest_timing)

        RUN_STATUS[run_id]["status"] = "completed"
        RUN_STATUS[run_id]["latest_video"] = latest_video
        RUN_STATUS[run_id]["run_tag"] = run_tag

        _write_json(
            STATE_FILE,
            {
                "last_run_id": run_id,
                "latest_video": latest_video,
                "run_tag": run_tag,
                "updated_at_epoch": time.time(),
            },
        )
        _emit(run_id, "pipeline_status", {"status": "completed", "run_tag": run_tag, "latest_video": latest_video})
    except Exception as exc:
        RUN_STATUS[run_id]["status"] = "failed"
        _emit(run_id, "pipeline_status", {"status": "failed", "error": str(exc)})


@app.get("/")
def index() -> Response:
    latest = _safe_read_json(STATE_FILE, fallback={})
    latest_video = latest.get("latest_video", "") or _latest_video_path()
    latest_video_url = f"/video?path={quote(latest_video)}" if latest_video else ''
    latest_download_url = f"/download?path={quote(latest_video)}" if latest_video else '#'
    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Montage Studio</title>
  <style>
    :root {{
      --bg: #030305;
      --surface: rgba(255,255,255,0.06);
      --surface2: rgba(255,255,255,0.08);
      --text: #f8f8f3;
      --muted: #bfc5d1;
      --accent: #ffe600;
      --accent2: #00ffc8;
      --accent3: #ff5f7d;
      --ok: #22c55e;
      --bad: #ef4444;
      --warn: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, 'Segoe UI', Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top left, rgba(255,230,0,0.08), transparent 15%),
                  radial-gradient(circle at 80% 10%, rgba(0,255,200,0.08), transparent 18%),
                  linear-gradient(180deg, #060607 0%, #0b0b10 55%, #050507 100%);
      min-height: 100vh;
    }}
    .container {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 24px 20px 40px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: clamp(3.4rem, 5vw, 5.5rem);
      line-height: 0.9;
      letter-spacing: -0.05em;
      background: linear-gradient(90deg, #ffe600, #00ffc8 35%, #ff5f7d 70%);
      -webkit-background-clip: text;
      color: transparent;
      text-shadow: 0 0 30px rgba(255,230,0,0.18);
    }}
    h2 {{ margin: 0 0 14px; font-size: 1.35rem; color: #f3f4f8; }}
    p.subtitle {{ color: #d6d8e0; margin-top: 0; max-width: 680px; line-height: 1.65; font-size: 1.05rem; }}
    .nav {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 24px; }}
    .brand {{ font-weight: 800; letter-spacing: 0.2em; text-transform: uppercase; color: var(--accent); }}
    .nav-links {{ display: flex; gap: 18px; flex-wrap: wrap; }}
    .nav-links a {{ color: var(--muted); text-decoration: none; font-size: 0.95rem; transition: color 0.2s ease; }}
    .nav-links a:hover {{ color: var(--accent2); }}
    .hero {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 28px; align-items: center; margin-bottom: 32px; }}
    .hero-tag {{ display: inline-flex; text-transform: uppercase; letter-spacing: 0.35em; font-size: 0.82rem; color: var(--accent); margin-bottom: 18px; opacity: 0.95; }}
    .hero-subtitle {{ max-width: 650px; margin-bottom: 24px; color: #d6d8e0; font-size: 1.05rem; }}
    .hero-links button {{ min-width: 180px; }}
    .hero-links .secondary {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18); color: var(--text); }}
    .hero-visual {{ position: relative; padding: 24px; background: radial-gradient(closest-side, rgba(255,255,255,0.12), transparent 62%); border: 1px solid rgba(255,255,255,0.16); border-radius: 32px; box-shadow: 0 34px 110px rgba(0,0,0,0.32); animation: float 8s ease-in-out infinite; overflow: hidden; }}
    .hero-visual::before {{ content: ''; position: absolute; inset: 18px; background: linear-gradient(135deg, rgba(255,230,0,0.18), transparent 45%, rgba(0,255,200,0.12)); border-radius: 26px; pointer-events: none; mix-blend-mode: screen; }}
    .hero-visual::after {{ content: ''; position: absolute; width: 220px; height: 220px; border-radius: 50%; background: rgba(255,230,0,0.16); top: -40px; right: -40px; filter: blur(32px); pointer-events: none; }}
    .hero-card {{
      position: relative;
      z-index: 1;
      min-height: 320px;
      border-radius: 28px;
      background: rgba(4,4,8,0.98);
      padding: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text);
      font-size: 1rem;
      line-height: 1.65;
      text-align: center;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
      overflow: hidden;
    }}
    .hero-card::before {{
      content: '';
      position: absolute;
      left: -20%;
      top: -30%;
      width: 160%;
      height: 180px;
      background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0));
      transform: rotate(-18deg);
      filter: blur(10px);
      opacity: 0.45;
      pointer-events: none;
    }}
    .hero-card::after {{
      content: '';
      position: absolute;
      right: 24px;
      bottom: 24px;
      width: 80px;
      height: 80px;
      border-radius: 20px;
      background: rgba(255,95,125,0.14);
      box-shadow: 0 0 40px rgba(255,95,125,0.18);
      pointer-events: none;
    }}
    .hero-card strong {{ display: block; font-size: 1.25rem; margin-bottom: 12px; color: #ffffff; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
    .primary-text {{ color: #ffffff; font-weight: 800; }}
    .card {{
      background: rgba(255,255,255,0.05);
      backdrop-filter: blur(18px);
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 28px;
      padding: 26px;
      box-shadow: 0 30px 90px rgba(0,0,0,0.22);
      position: relative;
      overflow: hidden;
      transition: transform 0.3s ease, border-color 0.3s ease;
    }}
    .card:hover {{ transform: translateY(-4px); border-color: rgba(255,245,0,0.3); }}
    .card::before {{
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at top left, rgba(255,255,0,0.08), transparent 22%),
                  radial-gradient(circle at bottom right, rgba(0,255,200,0.08), transparent 24%);
      pointer-events: none;
      opacity: 0.7;
    }}
    .phase-panel {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 14px 0;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .phase-panel:last-child {{ border-bottom: none; }}
    label {{ display: block; margin: 10px 0 6px; color: var(--muted); }}
    input, select, textarea {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.15);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      padding: 14px;
      font-size: 14px;
      transition: border-color 0.2s ease, background 0.2s ease;
    }}
    input:focus, select:focus, textarea:focus {{
      outline: none;
      border-color: rgba(255,230,0,0.8);
      background: rgba(255,255,255,0.08);
    }}
    textarea {{ min-height: 110px; resize: vertical; }}
    .row {{ display: flex; gap: 12px; align-items: center; margin-top: 14px; flex-wrap: wrap; }}
    button {{
      border: none;
      border-radius: 999px;
      padding: 14px 24px;
      color: #050507;
      cursor: pointer;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      box-shadow: 0 18px 32px rgba(255,230,0,0.15);
      transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }}
    button:hover {{ transform: translateY(-1px); filter: brightness(1.05); box-shadow: 0 24px 38px rgba(255,230,0,0.24); }}
    button.secondary {{
      background: linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.08));
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.2);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }}
    button.secondary:hover {{ transform: translateY(-1px); box-shadow: 0 20px 30px rgba(255,255,255,0.12); }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(255,255,255,0.08);
      margin: 4px 6px 0 0;
      font-size: 12px;
      letter-spacing: 0.02em;
    }}
    .chip code {{ color: var(--accent2); }}
    #log {{
      background: rgba(2,4,8,0.95);
      color: #dce6ff;
      border-radius: 18px;
      min-height: 260px;
      max-height: 400px;
      overflow: auto;
      padding: 20px;
      font-family: Consolas, monospace;
      font-size: 13px;
      white-space: pre-wrap;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .status-ok {{ color: var(--ok); }}
    .status-bad {{ color: var(--bad); }}
    .status-warn {{ color: var(--warn); }}
    .status-info {{ color: var(--accent2); }}
    video {{ width: 100%; border-radius: 18px; background: #000; height: auto; box-shadow: 0 18px 54px rgba(0,0,0,0.45); }}
    .glow {{ animation: glowPulse 3s infinite alternate; }}
    @keyframes glowPulse {{
      from {{ box-shadow: 0 0 18px rgba(255,230,0,0.14); }}
      to {{ box-shadow: 0 0 32px rgba(0,255,200,0.22); }}
    }}
    @keyframes float {{
      0%, 100% {{ transform: translateY(0px); }}
      50% {{ transform: translateY(-14px); }}
    }}
    @media (max-width: 980px) {{ .hero, .grid {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 680px) {{ .nav {{ flex-direction: column; align-items: start; }} .hero {{ gap: 20px; }} }}
  </style>
</head>
<body>
  <div class="container">
    <header class="nav">
      <div class="brand">Montage Studio</div>
      <div class="nav-links">
        <a href="#pipeline">Pipeline</a>
        <a href="#tools">Tools</a>
        <a href="#preview">Preview</a>
      </div>
      <button class="secondary" onclick="startPipeline()">Run Demo</button>
    </header>

    <section class="hero">
      <div>
        <div class="hero-tag">Video Editing AI</div>
        <h1>Montage Studio</h1>
        <p class="subtitle hero-subtitle">Turn a single creative prompt into a polished cinematic montage — story, voice, music, and video all orchestrated in real time.</p>
        <div class="row hero-links">
          <button onclick="startPipeline()">Run Full Pipeline</button>
          <button class="secondary" onclick="refreshLatest()">Refresh Latest Output</button>
        </div>
      </div>
      <div class="hero-visual">
        <div class="hero-card">
          <strong>Real-time montage creation</strong>
          Live workflow dashboard with phase controls, tool calls, timelined scenes, and instant output preview for cinematic iteration.
        </div>
      </div>
    </section>

    <div class="grid" id="pipeline">
      <div class="card">
        <h2>Pipeline Controls</h2>
        <label>Prompt</label>
        <textarea id="prompt" placeholder="A neon cyberpunk detective solving a mystery in 3 scenes"></textarea>
        <div class="row">
          <div style="flex:1">
            <label>Scenes</label>
            <input id="scenes" type="number" min="1" max="10" value="3" />
          </div>
          <div style="flex:1">
            <label>Quality</label>
            <select id="quality">
              <option value="fast">fast</option>
              <option value="balanced" selected>balanced</option>
              <option value="cinematic">cinematic</option>
            </select>
          </div>
          <div style="flex:1">
            <label>Image Backend</label>
            <select id="backend">
              <option value="auto" selected>auto</option>
              <option value="hf">hf</option>
              <option value="pollinations">pollinations</option>
            </select>
          </div>
        </div>
        <div class="row">
          <label><input id="subs" type="checkbox" checked /> Enable subtitles</label>
          <label><input id="humanApproval" type="checkbox" /> Human approval after Phase 1</label>
        </div>
        <div class="row">
          <button onclick="startPipeline()">Run Full Pipeline (P1→P3)</button>
          <button class="secondary" onclick="rerun('phase1')">Regenerate Story/Script (Phase 1)</button>
          <button class="secondary" onclick="rerun('phase2')">Regenerate Voice/Audio (Phase 2)</button>
          <button class="secondary" onclick="rerun('phase3')">Recompose Video (Phase 3)</button>
        </div>
        <div class="row">
          <input id="sceneId" type="number" min="1" placeholder="Scene id for partial Phase 3 rerun" />
          <button class="secondary" onclick="rerunScene()">Rerun Single Scene (Phase 3)</button>
        </div>
        <div class="row" style="margin-top:14px; justify-content:space-between;">
          <button class="secondary" id="approveButton" onclick="approveRun()" disabled>Approve Phase 1</button>
          <div id="currentTool" class="chip">Tool: idle</div>
        </div>
      </div>

      <div class="card">
        <h3>Phase Status</h3>
        <div class="phase-panel">
          <div><strong>Phase 1</strong></div>
          <div id="phase1State" class="chip">waiting</div>
          <div id="phase1Command" class="chip" style="background:rgba(255,255,255,0.05);font-size:11px;">command pending</div>
        </div>
        <div class="phase-panel">
          <div><strong>Phase 2</strong></div>
          <div id="phase2State" class="chip">waiting</div>
          <div id="phase2Command" class="chip" style="background:rgba(255,255,255,0.05);font-size:11px;">command pending</div>
        </div>
        <div class="phase-panel">
          <div><strong>Phase 3</strong></div>
          <div id="phase3State" class="chip">waiting</div>
          <div id="phase3Command" class="chip" style="background:rgba(255,255,255,0.05);font-size:11px;">command pending</div>
        </div>
        <div class="phase-panel" style="margin-top:10px;">
          <div><strong>Pipeline</strong></div>
          <div id="pipelineState" class="chip status-warn">idle</div>
          <div id="runTag" class="chip" style="background:rgba(255,255,255,0.05);font-size:11px;">latest run tag</div>
        </div>
      </div>
    </div>

    <div class="grid" style="margin-top:16px; gap:14px;">
      <div class="card">
        <h3>Tool Calling (MCP-style)</h3>
        <p style="color:var(--muted); margin-top:0">Loaded dynamically from <code>data/mcp_registry.json</code>.</p>
        <div id="tools"></div>
        <div class="row">
          <input id="capability" placeholder="capability (e.g. generate_scene_image_fallback)" />
        </div>
        <textarea id="toolPayload" placeholder='{{"prompt":"sunset city skyline","output_dir":"data/image_assets","filename":"demo_city"}}'></textarea>
        <div class="row">
          <button class="secondary" onclick="callTool()">Call Tool</button>
        </div>
        <div id="toolResult" style="font-family:Consolas,monospace; font-size:12px; white-space:pre-wrap;"></div>
      </div>

      <div class="card">
        <h3>Phase Outputs</h3>
        <div class="row" style="flex-direction:column; align-items:flex-start; gap:6px;">
          <span class="chip" id="outputRunTag">Latest run: {latest.get('run_tag','-')}</span>
          <span class="chip">Latest video path:</span>
          <code id="latestVideoPath" style="display:block; word-break:break-all; margin-top:6px; color:var(--muted);">{latest_video}</code>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Live Progress</h3>
      <div id="log"></div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>Final Video</h3>
      <video id="preview" controls src="{latest_video_url}"></video>
      <div class="row">
        <button class="secondary" onclick="refreshLatest()">Refresh Latest Output</button>
        <a id="downloadLink" href="{latest_download_url}" style="color:#c4b5fd">Download MP4</a>
      </div>
    </div>
  </div>

  <script>
    let activeRunId = null;
    let currentEventSource = null;

    async function loadTools() {{
      const resp = await fetch('/api/tools');
      const data = await resp.json();
      const el = document.getElementById('tools');
      el.innerHTML = (data.tools || []).map(t => `<span class="chip">${{t.capability}} -> ${{t.type}}</span>`).join('');
    }}

    function addLog(msg, cls='') {{
      const log = document.getElementById('log');
      const line = document.createElement('div');
      line.className = cls;
      line.textContent = msg;
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
    }}

    function setPhaseStatus(phase, status, command) {{
      document.getElementById(`phase${{phase.slice(-1)}}State`).textContent = status;
      document.getElementById(`phase${{phase.slice(-1)}}State`).className = `chip ${{status === 'completed' ? 'status-ok' : status === 'failed' ? 'status-bad' : 'status-warn'}}`;
      if (command) {{
        document.getElementById(`phase${{phase.slice(-1)}}Command`).textContent = command;
      }}
    }}

    function setPipelineStatus(status, message, runTag) {{
      const badge = document.getElementById('pipelineState');
      badge.textContent = status;
      badge.className = `chip ${{status === 'completed' ? 'status-ok' : status === 'failed' ? 'status-bad' : 'status-warn'}}`;
      if (runTag) {{
        document.getElementById('runTag').textContent = `run: ${{runTag}}`;
        document.getElementById('outputRunTag').textContent = `Latest run: ${{runTag}}`;
      }}
      if (message) addLog(`[pipeline] ${{message}}`, 'status-info');
    }}

    function currentPayload(extra={{}}) {{
      return {{
        prompt: document.getElementById('prompt').value,
        scenes: Number(document.getElementById('scenes').value || 3),
        quality: document.getElementById('quality').value,
        backend: document.getElementById('backend').value,
        enable_subtitles: document.getElementById('subs').checked,
        human_in_loop: document.getElementById('humanApproval').checked,
        ...extra
      }};
    }}

    async function startPipeline() {{
      const res = await fetch('/api/pipeline/start', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(currentPayload())
      }});
      const data = await res.json();
      if (data.run_id) connectSSE(data.run_id);
    }}

    async function rerun(phase) {{
      const res = await fetch('/api/pipeline/start', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(currentPayload({{ rerun_phase: phase }}))
      }});
      const data = await res.json();
      if (data.run_id) connectSSE(data.run_id);
    }}

    async function rerunScene() {{
      const sid = Number(document.getElementById('sceneId').value);
      if (!sid) return alert('Enter a valid scene id');
      const res = await fetch('/api/pipeline/start', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(currentPayload({{ rerun_phase: 'phase3', scene_id: sid }}))
      }});
      const data = await res.json();
      if (data.run_id) connectSSE(data.run_id);
    }}

    function approveRun() {{
      if (!activeRunId) return alert('Start a run before approving Phase 1');
      fetch(`/api/pipeline/approve/${{encodeURIComponent(activeRunId)}}`, {{ method: 'POST' }})
        .then(r => r.json())
        .then(data => {{
          if (data.ok) {{
            addLog(`[approval] Phase 1 approved for run ${{activeRunId}}`, 'status-ok');
            document.getElementById('approveButton').disabled = true;
            document.getElementById('currentTool').textContent = 'Tool: human approval granted';
          }} else {{
            addLog(`[approval] ${{data.error || 'Approval failed'}}`, 'status-bad');
          }}
        }});
    }}

    function resetPhaseCards() {{
      ['1', '2', '3'].forEach(num => {{
        document.getElementById(`phase${{num}}State`).textContent = 'waiting';
        document.getElementById(`phase${{num}}State`).className = 'chip status-warn';
        document.getElementById(`phase${{num}}Command`).textContent = 'command pending';
      }});
      const pipelineState = document.getElementById('pipelineState');
      pipelineState.textContent = 'starting';
      pipelineState.className = 'chip status-warn';
      document.getElementById('approveButton').disabled = true;
      document.getElementById('currentTool').textContent = 'Tool: idle';
    }}

    function connectSSE(runId) {{
      if (currentEventSource) {{
        currentEventSource.close();
      }}
      activeRunId = runId;
      resetPhaseCards();
      addLog(`\n--- Listening to run ${{runId}} ---`, 'status-warn');
      currentEventSource = new EventSource(`/api/pipeline/stream/${{runId}}`);
      currentEventSource.addEventListener('phase_log', (e) => {{
        const d = JSON.parse(e.data);
        addLog(`[${{d.phase}}] ${{d.line}}`, '');
      }});
      currentEventSource.addEventListener('phase_status', (e) => {{
        const d = JSON.parse(e.data);
        setPhaseStatus(d.phase, d.status, d.command || 'command pending');
        addLog(`[${{d.phase}}] status=${{d.status}}`, d.status === 'completed' ? 'status-ok' : (d.status === 'failed' ? 'status-bad' : 'status-warn'));
      }});
      currentEventSource.addEventListener('pipeline_status', (e) => {{
        const d = JSON.parse(e.data);
        setPipelineStatus(d.status, d.message || d.error || '', d.run_tag);
        if (d.status === 'awaiting_approval') {{
          document.getElementById('approveButton').disabled = false;
          document.getElementById('currentTool').textContent = 'Tool: awaiting human approval';
        }} else if (d.status === 'approved') {{
          document.getElementById('currentTool').textContent = 'Tool: human approved';
        }} else if (d.status === 'completed') {{
          refreshLatest();
          currentEventSource.close();
          document.getElementById('currentTool').textContent = 'Tool: completed';
        }} else if (d.status === 'failed') {{
          currentEventSource.close();
          document.getElementById('currentTool').textContent = 'Tool: failed';
        }}
      }});
    }}
    async function callTool() {{
      try {{
        const capability = document.getElementById('capability').value.trim();
        const payload = JSON.parse(document.getElementById('toolPayload').value || '{{}}');
        document.getElementById('currentTool').textContent = `Tool: calling ${{capability || 'unknown'}}`;
        const res = await fetch('/api/tools/call', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ capability, payload }})
        }});
        const data = await res.json();
        document.getElementById('toolResult').textContent = JSON.stringify(data, null, 2);
        document.getElementById('currentTool').textContent = data.ok ? `Tool: ${{capability}} returned` : 'Tool: error';
      }} catch (err) {{
        document.getElementById('toolResult').textContent = String(err);
        document.getElementById('currentTool').textContent = 'Tool: failed';
      }}
    }}

    async function refreshLatest() {{
      const res = await fetch('/api/pipeline/latest');
      const data = await res.json();
      if (data.latest_video) {{
        const videoPath = `/video?path=${{encodeURIComponent(data.latest_video)}}&t=${{Date.now()}}`;
        const preview = document.getElementById('preview');
        preview.src = videoPath;
        preview.load();
        document.getElementById('downloadLink').href = `/download?path=${{encodeURIComponent(data.latest_video)}}`;
        document.getElementById('outputRunTag').textContent = `Latest run: ${{data.run_tag || '-'}}`;
        document.getElementById('latestVideoPath').textContent = data.latest_video;
      }}
    }}

    loadTools();
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.post("/api/pipeline/start")
def start_pipeline() -> Response:
    payload = request.get_json(silent=True) or {}
    run_id = str(uuid.uuid4())[:8]
    RUN_EVENTS[run_id] = queue.Queue()
    RUN_STATUS[run_id] = {"status": "queued", "payload": payload, "phases": {}}
    threading.Thread(target=_pipeline_worker, args=(run_id, payload), daemon=True).start()
    return jsonify({"ok": True, "run_id": run_id})


@app.post("/api/pipeline/approve/<run_id>")
def approve_pipeline(run_id: str) -> Response:
    if run_id not in RUN_APPROVAL_EVENTS:
        return jsonify({"ok": False, "error": "Approval not required or run not found."}), 404
    RUN_APPROVAL_EVENTS[run_id].set()
    _emit(run_id, "pipeline_status", {"status": "approved", "message": "Human approval received. Continuing pipeline."})
    return jsonify({"ok": True, "run_id": run_id})


@app.get("/api/pipeline/status/<run_id>")
def get_pipeline_status(run_id: str) -> Response:
    if run_id not in RUN_STATUS:
        return jsonify({"ok": False, "error": "Run not found"}), 404
    return jsonify({"ok": True, "run_id": run_id, "status": RUN_STATUS[run_id]})


@app.get("/api/pipeline/stream/<run_id>")
def stream_pipeline(run_id: str) -> Response:
    if run_id not in RUN_EVENTS:
        return Response("event: pipeline_status\ndata: {\"status\":\"failed\",\"error\":\"run not found\"}\n\n", mimetype="text/event-stream")

    def gen() -> Any:
        q = RUN_EVENTS[run_id]
        while True:
            try:
                item = q.get(timeout=20)
                event = item["event"]
                payload = json.dumps(item["payload"])
                yield f"event: {event}\ndata: {payload}\n\n"
                if event == "pipeline_status" and item["payload"].get("status") in {"completed", "failed"}:
                    break
            except queue.Empty:
                yield "event: heartbeat\ndata: {}\n\n"

    return Response(gen(), mimetype="text/event-stream")


@app.get("/api/pipeline/latest")
def latest() -> Response:
    latest = _safe_read_json(STATE_FILE, fallback={})
    latest_video = latest.get("latest_video", "") or _latest_video_path()
    return jsonify({"latest_video": latest_video, "run_tag": _last_run_tag(latest_video)})


@app.get("/api/tools")
def list_tools() -> Response:
    registry = ToolRegistry()
    return jsonify({"tools": registry.list_tools()})


@app.post("/api/tools/call")
def call_tool() -> Response:
    body = request.get_json(silent=True) or {}
    capability = (body.get("capability") or "").strip()
    payload = body.get("payload") or {}
    if not capability:
        return jsonify({"ok": False, "error": "capability is required"}), 400
    try:
        registry = ToolRegistry()
        client = ToolClient(registry, image_dir="data/image_assets")
        result = client.invoke_by_capability(capability, payload)
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/video")
def stream_video() -> Response:
    path = request.args.get("path", "")
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (ROOT / path).resolve()
    if not file_path.exists():
        return jsonify({"ok": False, "error": "video not found"}), 404
    response = send_file(file_path, mimetype="video/mp4")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/download")
def download_video() -> Response:
    path = request.args.get("path", "")
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (ROOT / path).resolve()
    if not file_path.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_file(file_path, mimetype="video/mp4", as_attachment=True, download_name=file_path.name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
