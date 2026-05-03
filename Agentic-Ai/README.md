# Agentic Montage Studio

End-to-end AI pipeline that turns a single natural-language prompt into a polished short animated video — story, dialogue, character voices, scene visuals, and a final composited MP4 — orchestrated by LangGraph agents.

This repository implements **Phase 1 (Story & Script)**, **Phase 2 (Audio)** and **Phase 3 (Video Composition)** of the assignment. Phase 4 (web UI) and Phase 5 (edit/undo agent) are owned by other team members.

---

## Pipeline Overview

```
prompt
  │
  ▼
Phase 1: Story / Script / Characters     (LangGraph + Groq LLM)
  │   story_manifest_auto.json
  │   scene_manifest_auto.json
  │   character_db_auto.json
  ▼
Phase 2: Audio + BGM + Timing            (LangGraph + edge-tts + Freesound)
  │   audio/scene_NN_runXX.mp3
  │   bgm/<mood>_sceneNN_freesound.mp3
  │   timing_manifest_runXX.json
  ▼
Phase 3: Video Composition               (LangGraph + HF / Pollinations + ffmpeg)
      images/scene_NN_*.png
      clips/scene_NN_kb.mp4
      composed/scene_NN_composed.mp4
      final_output_runXX.mp4
      subtitles_runXX.srt   (optional)
```

---

## Prerequisites

- Python 3.11+
- ffmpeg + ffprobe on PATH (`ffmpeg -version` should print a version)
- A free Hugging Face token (for Phase 3 image generation, optional — Pollinations works without a key)
- (Optional) Groq, Freesound, ElevenLabs API keys for higher quality Phase 1 + 2

### Setup

```powershell
cd Agentic-Ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# So that Unicode banners in logs do not crash on Windows cp1252
$env:PYTHONIOENCODING = "utf-8"
```

### Required `.env` (project root)

```
GROQ_API_KEY=<your_key>
FREESOUND_API_KEY=<your_key>
ELEVENLABS_API_KEY=<your_key>
HF_TOKEN=<your_huggingface_token>
HF_IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
```

`HF_TOKEN` is only needed if you want the Hugging Face image backend; without it, Phase 3 falls back to Pollinations automatically.

---

## How to Run (End-to-End)

```powershell
# 1. Reset previous runs (optional)
python reset_data.py

# 2. Phase 1 — generate story, script and characters
python -m src.main --mode auto `
                   --prompt "A sci-fi mystery about time travel with 3 scenes" `
                   --scenes 3 `
                   --auto-approve

# 3. Phase 2 — generate audio, BGM and timing manifest
python -m src.main_phase2

# 4. Phase 3 — generate visuals and composite the final MP4 (with crossfades)
python -m src.main_phase3 --quality balanced --transition-sec 0.35 --enable-subtitles
```

The final MP4 will be at `data/phase3_runs/runXX/final_output_runXX.mp4` (or `..._subbed.mp4` when subtitles are burned in).

---

## Phase 3 — Detailed Usage

### CLI flags

```
python -m src.main_phase3 [flags]

  --manifest        Path to scene_manifest JSON         (default: data/scene_manifest_auto.json)
  --characters      Path to character_db JSON           (default: data/character_db_auto.json)
  --timing          Path to timing_manifest JSON        (default: latest under data/phase2_runs)
  --phase2-run      Specific Phase 2 run tag, e.g. run01

  --backend         auto | hf | pollinations             (default: auto = HF then Pollinations)
  --quality         fast | balanced | cinematic          (default: balanced)
  --seed            Optional integer seed for deterministic image generation
  --scene-id        Partial rerun: regenerate only this scene id

  --enable-subtitles  Burn dialogue subtitles into the final MP4
  --transition-sec    Crossfade duration in seconds (default: 0.35, 0 disables fades)
  --scene-image-only  Disable speaker-focused line images and use one still per scene
```

### Quality profiles

| Profile     | Resolution  | FPS | x264 CRF | x264 preset | Image res |
|-------------|-------------|-----|----------|-------------|-----------|
| `fast`      | 854 × 480   | 24  | 28       | veryfast    | 768 × 768 |
| `balanced`  | 1280 × 720  | 30  | 23       | medium      | 1024 × 1024|
| `cinematic` | 1920 × 1080 | 30  | 20       | slow        | 1280 × 720|

### What you get per Phase 3 run

```
data/phase3_runs/runXX/
├── images/scene_NN_<fingerprint>.png        # one cinematic still per scene
├── clips/scene_NN_kb.mp4                    # silent Ken Burns animation per scene
├── composed/scene_NN_composed.mp4           # scene clip + voice + BGM
├── final_output_runXX.mp4                   # all scenes concatenated
├── final_output_runXX_subbed.mp4            # only if --enable-subtitles
├── subtitles_runXX.srt                      # only if --enable-subtitles
├── image_prompts_runXX.json                 # exact prompt + backend used per scene
├── phase3_state_runXX.json                  # workflow status snapshot
├── phase3_outputs_runXX.json                # full result payload (paths + options)
└── ffmpeg_commands_runXX.log                # every ffmpeg invocation, for audit
```

The Phase 3 run tag is **aligned to the Phase 2 run** by default (e.g. Phase 2 `run01` → Phase 3 `run01`). Pass `--phase2-run runXX` to lock it explicitly.

### Demo-friendly tricks

- `--seed 42 --backend pollinations` → deterministic, key-free demo run.
- Default mode is speaker-focused (one image per spoken line); use `--scene-image-only` for old behavior.
- `--scene-id 3` → re-render only scene 3 after editing; previous scenes are reused.
- Cached images: identical prompt + size + seed reuses the on-disk image, saving API calls.
- `image_prompts_runXX.json` shows exactly which backend produced each frame (`hf:<model>`, `pollinations`, or `cache`).

---

## Architecture

```
src/
├── main.py                         # Phase 1 entrypoint
├── main_phase2.py                  # Phase 2 entrypoint
├── main_phase3.py                  # Phase 3 entrypoint  (new)
├── run_manager.py                  # Phase 2 + Phase 3 run / counter management
├── workflows/
│   ├── langgraph_flow.py           # Phase 1 LangGraph DAG
│   ├── langgraph_phase2.py         # Phase 2 LangGraph DAG
│   └── langgraph_phase3.py         # Phase 3 LangGraph DAG  (new)
├── agents/
│   ├── scriptwriter.py / story_generator.py / character_designer.py …
│   ├── voice_synthesizer.py
│   ├── music_selector.py
│   ├── scene_visualizer.py         # Phase 3 image generation (HF + Pollinations) (new)
│   └── video_generator.py          # Thin MCP-style wrapper over scene_visualizer
├── io/
│   ├── json_schema.py
│   ├── script_ingest.py
│   └── phase3_contracts.py         # Phase 3 input validation + ScenePlan model (new)
├── utils/
│   ├── timing_manifest.py
│   └── video_compose.py            # ffmpeg helpers: Ken Burns, compose, concat, SRT (new)
├── mcp/
│   ├── tool_registry.py
│   └── tool_client.py              # +hf_image, +pollinations_image (new)
└── memory/vector_store.py
```

### Phase 3 LangGraph DAG

```
scene_parser  → image_gen  → motion (Ken Burns) → compose (scene A/V)
              → mux (final concat) → subtitles (optional) → memory_commit → END
```

Every node updates a shared `Phase3State` so the workflow is fully observable and a partial rerun (`--scene-id N`) skips work safely.

---

## Tests

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m unittest tests.test_phase3 -v
```

15 unit tests cover:

- Input contract validation (positive + negative cases)
- Scene plan cross-join from `scene_manifest` + `timing_manifest`
- Image prompt construction (location, mood, character continuity, style anchor)
- Image backend fallback (HF fail → Pollinations success)
- Backend selection: explicit `pollinations` skips HF
- Image cache hit avoids any backend call
- SRT line formatting + ffmpeg quality profile defaults

End-to-end smoke test (runs the real Phase 3 pipeline against existing Phase 2 outputs):

```powershell
python scripts\smoke_phase3.py --backend pollinations --quality fast
```

Subtitle timing QA (checks timing manifest vs SRT vs final video envelope):

```powershell
python scripts\subtitle_timing_qa.py `
  --timing data\phase2_runs\run03\timing_manifest_run03.json `
  --srt data\phase3_runs\run03\subtitles_run03.srt `
  --video data\phase3_runs\run03\final_output_run03_subbed.mp4
```

---

## Phase 3 Rubric Mapping

| Rubric criterion (PDF)                                | Where it is satisfied                                                                 |
|-------------------------------------------------------|---------------------------------------------------------------------------------------|
| Per-scene image generation from script context        | `src/agents/scene_visualizer.py` (`build_scene_prompt` + `generate_scene_image`)      |
| Free + no-GPU image backend                           | Hugging Face Inference API (primary), Pollinations (fallback) — both free            |
| Light animation (zoom/pan/Ken Burns) via FFmpeg       | `src/utils/video_compose.py::ken_burns_clip`                                          |
| A/V sync using timing manifest                         | `compose_node` reads `audio_file` + `bgm_file` + `duration_ms` from timing manifest   |
| Subtitle overlay (optional)                           | `--enable-subtitles` ⇒ `build_srt` + `burn_subtitles`                                |
| Compositing scenes with transitions into final MP4    | `concat_scenes_with_crossfade` using ffmpeg `xfade` + `acrossfade`                    |
| Final MP4 export                                      | `data/phase3_runs/runXX/final_output_runXX.mp4`                                       |
| Phase-level unit tests                                | `tests/test_phase3.py` (15 tests, including fallback + cache + SRT correctness)       |
| Reproducible run artifacts                            | `phase3_state_*.json`, `phase3_outputs_*.json`, `image_prompts_*.json`, ffmpeg log    |
| Modular, independently testable phase                 | Phase 3 has its own LangGraph workflow, entrypoint, and contracts                     |

---

## Troubleshooting

| Symptom                                            | Fix                                                                                        |
|----------------------------------------------------|--------------------------------------------------------------------------------------------|
| `UnicodeEncodeError ... charmap` on Windows        | Run `\$env:PYTHONIOENCODING = "utf-8"` before any Python command                            |
| `ffmpeg executable not found on PATH`              | Install ffmpeg (e.g. `winget install Gyan.FFmpeg`) and reopen the shell                    |
| HF returns 503 / model loading                     | Phase 3 retries with the estimated wait time, then falls back to Pollinations               |
| Pollinations rate-limit                            | Re-run with `--seed N` to hit cache, or `--backend hf`                                     |
| `ModuleNotFoundError: No module named 'src'`       | Run from project root using `python -m src.main_phase3` (not `python src/main_phase3.py`)  |
| Need a fresh state                                 | `python reset_data.py` clears all run history                                              |
