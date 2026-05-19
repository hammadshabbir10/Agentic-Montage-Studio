# 🎬 CineAgent — AI-Powered Animated Video Generation System

> *From a single prompt to a polished short film — fully autonomous, end-to-end, agent-orchestrated.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Pipeline-purple?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203-orange?style=flat-square)
![EdgeTTS](https://img.shields.io/badge/Edge--TTS-Voice%20Synthesis-green?style=flat-square)
![Freesound](https://img.shields.io/badge/Freesound-BGM-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Phase Status](#-phase-status)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [JSON Schema Design](#-json-schema-design)
- [Setup & Installation](#-setup--installation)
- [Running the Pipeline](#-running-the-pipeline)
- [Sample Outputs](#-sample-outputs)
- [Team](#-team)

---

## 🌟 Project Overview

**CineAgent** is a multi-phase, agentic AI pipeline that accepts a single natural-language prompt and autonomously produces a complete short animated video — including story, dialogue, character voices, background music, and a fully synchronized timing manifest — with zero manual creative intervention.

Built for the **Agentic AI Course Project 2026** at FAST-NUCES Islamabad, the system is not a simple API wrapper. Each phase is a distinct AI-powered module with well-defined JSON contracts, orchestrated by LangGraph agents.

**Example Prompt →**
```
"A CIA operative in 1960s divided Berlin must extract a Soviet defector
 before the KGB closes in, uncovering a double agent within his own team"
```

**Pipeline Output →**
- `story_manifest.json` — structured narrative with 4-act breakdown
- `scene_manifest.json` — scene-by-scene script with dialogue & visual cues
- `character_db.json` — character profiles with personality, appearance, voice style
- `audio/` — per-line MP3 files with unique voices per character
- `bgm/` — mood-detected background music per scene (Freesound / MusicGen / stub)
- `timing_manifest.json` — millisecond-accurate A/V sync map

---

## 🏗 System Architecture

```
User Prompt
     │
     ▼
┌─────────────────────────────────────────────┐
│  Phase 1 — Story, Script & Character Design │
│  LangGraph → Groq LLaMA 3                   │
│  story_manifest + scene_manifest + char_db  │
└──────────────────────┬──────────────────────┘
                       │  structured JSON
                       ▼
┌─────────────────────────────────────────────┐
│  Phase 2 — Audio Generation & Integration   │
│  Edge-TTS (per-character voices)            │
│  Freesound API / MusicGen (BGM)             │
│  timing_manifest.json                       │
└──────────────────────┬──────────────────────┘
                       │  audio + sync map
                       ▼
┌─────────────────────────────────────────────┐
│  Phase 3 — Video Generation & Composition   │  
│  Stable Diffusion / DALL-E visuals          │
│  FFmpeg / MoviePy → final_output.mp4        │
└──────────────────────┬──────────────────────┘
                       │  rendered MP4
                       ▼
┌─────────────────────────────────────────────┐
│  Phase 4 — Web Interface & Orchestration    │  
│  FastAPI backend + React frontend           │
│  Real-time progress via WebSocket/SSE       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  Phase 5 — Intelligent Edit & Undo Agent    │  ✅ Complete
│  LangGraph edit intent classifier           │
│  State versioning + full undo/revert        │
└─────────────────────────────────────────────┘
```

---

## ✅ Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Story, Script & Character Design | ✅ Complete |
| **Phase 2** | Audio Generation & Integration | ✅ Complete |
| **Phase 3** | Video Generation & Composition | ✅ Complete |
| **Phase 4** | Web Interface & Orchestration | ✅ Complete |
| **Phase 5** | Intelligent Edit & Undo Agent | ✅ Complete |

---

## 🛠 Technology Stack

| Layer | Primary | Local / Budget Alt |
|---|---|---|
| LLM / Agents | Groq API (LLaMA 3 70B) + LangGraph | Ollama + LLaMA 3 / Mistral |
| TTS | Microsoft Edge-TTS (free, neural) | ElevenLabs, Coqui TTS |
| BGM | Freesound API + MusicGen (Meta) | Royalty-free library |
| Image Gen | Stable Diffusion / DALL-E 3 | ComfyUI (local) |
| Video Comp. | MoviePy + FFmpeg | FFmpeg only |
| Backend | FastAPI + Uvicorn | Django |
| Frontend | React + Vite | Next.js |
| State Store | File-based JSON snapshots | SQLite (LangGraph SqliteSaver) |
| Vector Memory | ChromaDB | In-memory |

---

## 📁 Project Structure

```
CineAgent/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── story_generator.py       # Phase 1 — story arc generation
│   │   ├── scriptwriter.py          # Phase 1 — scene-by-scene script
│   │   ├── character_designer.py    # Phase 1 — character roster
│   │   ├── validator.py             # Phase 1 — script validation
│   │   ├── hitl.py                  # Phase 1 — human-in-the-loop approval
│   │   ├── voice_synthesizer.py     # Phase 2 — TTS per dialogue line
│   │   ├── music_selector.py        # Phase 2 — BGM selection / generation
│   │   └── image_synthesizer.py     # Phase 3 — visual generation (WIP)
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── json_schema.py           # build_scene_manifest / build_story_manifest
│   │   └── script_ingest.py         # raw screenplay → structured JSON
│   │
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── langgraph_flow.py        # Phase 1 LangGraph pipeline
│   │   └── langgraph_phase2.py      # Phase 2 LangGraph pipeline
│   │
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── tool_client.py           # unified MCP tool dispatcher
│   │   └── tool_registry.py         # tool capability registry
│   │
│   ├── memory/
│   │   └── vector_store.py          # ChromaDB memory store
│   │
│   ├── utils/
│   │   ├── logging.py
│   │   ├── timing_manifest.py       # Phase 2 — A/V sync manifest builder
│   │   └── gender_detector.py       # character gender detection utility
│   │
│   ├── main.py                      # Phase 1 entry point
│   ├── main_phase2.py               # Phase 2 entry point
│   └── main_phase3.py               # Phase 3 entry point
│
├── src/agents/
│   ├── edit_intent_classifier.py    # Phase 5 — LLM+rule edit intent classifier
│   └── edit_executor.py             # Phase 5 — edit dispatch & execution
│
├── src/workflows/
│   ├── langgraph_flow.py            # Phase 1 LangGraph pipeline
│   ├── langgraph_phase2.py          # Phase 2 LangGraph pipeline
│   ├── langgraph_phase3.py          # Phase 3 LangGraph pipeline
│   └── langgraph_phase5.py          # Phase 5 LangGraph pipeline (edit & undo)
│
├── src/utils/
│   ├── image_filters.py             # Phase 5 — OpenCV filter library
│   ├── video_compose.py             # Phase 3 — FFmpeg composition
│   └── timing_manifest.py           # Phase 2 — A/V sync manifest
│
├── src/
│   └── state_versioning.py          # Phase 5 — StateManager (snapshot/revert/history)
│
├── data/
│   ├── scene_manifest_auto.json     # Phase 1 output — scenes & dialogue
│   ├── scene_manifest_manual.json
│   ├── story_manifest_auto.json     # Phase 1 output — story structure
│   ├── character_db_auto.json       # Phase 1 output — character profiles
│   ├── character_db_manual.json
│   ├── last_script_auto.txt         # raw generated screenplay text
│   ├── bgm_library/                 # optional local BGM files (mood_XX.mp3)
│   ├── image_assets/                # character reference images
│   ├── memory/                      # ChromaDB vector store
│   ├── phase2_runs/
│   │   └── run{NN}/
│   │       ├── audio/
│   │       │   ├── scene{N}/        # individual line MP3s
│   │       │   │   ├── JACK_line001.mp3
│   │       │   │   └── LARISA_line002.mp3
│   │       │   └── scene_01_run{NN}.mp3   # concatenated scene audio
│   │       ├── bgm/
│   │       │   └── tense_scene01_freesound.mp3
│   │       ├── timing_manifest_run{NN}.json
│   │       ├── task_graph_run{NN}.json
│   │       └── phase2_outputs_run{NN}.json
│   └── task_graph_logs/
│       └── run{NN}_task_graph.json
│
├── tests/
│   ├── test_phase1.py
│   └── test_phase2.py
│
├── .env                             # API keys (not committed)
├── .env.example                     # template for .env
├── requirements.txt
├── mcp_registry.json                # tool capability → type mapping
└── README.md
```

---

## 📐 JSON Schema Design

All phases communicate through a shared JSON state. Below are the core schemas.

### `story_manifest.json`
```json
{
  "workflow_id": "workflow_20260423_193225",
  "timestamp": "2026-04-23T19:32:25.093513",
  "story": {
    "title": "Berlin Divide",
    "logline": "A CIA operative in 1960s Berlin must extract a Soviet defector before the KGB closes in, while uncovering a double agent within his own team.",
    "genre": "Cold War Thriller",
    "tone": "tense and gritty",
    "setting": "Divided Berlin",
    "time_period": "1960s",
    "themes": ["loyalty", "betrayal", "survival"],
    "acts": [
      { "act": 1, "label": "Introduction",    "description": "We meet CIA operative JACK HARRIS navigating 1960s Berlin..." },
      { "act": 2, "label": "Rising Action",   "description": "Harris suspects a mole within his own team..." },
      { "act": 3, "label": "Climax",          "description": "Harris confronts the double agent with the KGB closing in..." },
      { "act": 4, "label": "Resolution",      "description": "Harris completes the extraction, haunted by the cost..." }
    ],
    "protagonist": "Jack Harris, a seasoned CIA operative",
    "antagonist": "The KGB and an unknown mole inside the CIA",
    "world": "In the midst of the Cold War, Berlin is a city divided where the slightest misstep means capture or death."
  }
}
```

### `scene_manifest.json`
```json
{
  "workflow_id": "workflow_20260423_193227",
  "timestamp": "2026-04-23T19:32:27.592659",
  "total_scenes": 4,
  "scenes": [
    {
      "scene_id": 1,
      "location": "BERLIN CIA SAFE HOUSE",
      "duration": 10,
      "characters": ["JACK", "LARISA", "RYAN"],
      "dialogue": [
        {
          "speaker": "JACK",
          "line": "We need to get out of here, now. The KGB will be looking for you.",
          "visual_cue": "Medium shot of JACK, expression intense and focused."
        },
        {
          "speaker": "LARISA",
          "line": "What about my family? They'll be in danger if I'm caught.",
          "visual_cue": "Close-up of LARISA, a worried expression on her face."
        }
      ]
    }
  ]
}
```

### `character_db.json`
```json
{
  "workflow_id": "workflow_20260423_193240",
  "timestamp": "2026-04-23T19:32:40.801933",
  "total_characters": 4,
  "characters": [
    {
      "name": "JACK",
      "personality": "Driven and fiercely loyal, growing paranoia clouds his judgment.",
      "appearance": "Ruggedly handsome, late 30s, scar above left eyebrow, classic suit and tie.",
      "role": "protagonist",
      "style_reference": "Cold War thriller, dark and gritty, high-contrast shadows.",
      "first_appearance": 1,
      "dialogue_samples": [
        {
          "line": "I don't trust anyone right now. Not even you, Ryan.",
          "visual_cue": "Wide shot of VICTORIA, her eyes scanning the surroundings."
        }
      ]
    }
  ]
}
```

### `timing_manifest.json` (Phase 2 output)
```json
{
  "workflow_id": "phase2_run06",
  "timestamp": "2026-04-23T19:35:10.000000",
  "run_tag": "run06",
  "total_duration_ms": 40000,
  "total_duration_sec": 40.0,
  "scenes": [
    {
      "scene_id": 1,
      "audio_file": "data/phase2_runs/run06/audio/scene_01_run06.mp3",
      "bgm_file":   "data/phase2_runs/run06/bgm/tense_scene01_freesound.mp3",
      "mood":       "tense",
      "bgm_source": "freesound",
      "start_ms":   0,
      "end_ms":     10000,
      "duration_ms": 10000,
      "lines": [
        {
          "speaker":     "JACK",
          "voice":       "en-US-GuyNeural",
          "line":        "We need to get out of here, now.",
          "visual_cue":  "Medium shot of JACK, expression intense and focused.",
          "audio_file":  "data/phase2_runs/run06/audio/scene1/JACK_line001.mp3",
          "start_ms":    0,
          "end_ms":      2200,
          "duration_ms": 2200
        }
      ]
    }
  ]
}
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- `ffmpeg` installed and on PATH ([download](https://ffmpeg.org/download.html))
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/CineAgent.git
cd CineAgent
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
```bash
cp .env.example .env
```
Edit `.env` and fill in your keys:
```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Phase 2 BGM (optional — falls back to silence stub)
FREESOUND_API_KEY=your_freesound_api_key_here
ENABLE_MUSICGEN=false

# Optional image generation
STABILITY_API_KEY=your_stability_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
```

> **Get free API keys:**
> - Groq: https://console.groq.com (free tier, fast LLaMA 3)
> - Freesound: https://freesound.org/apiv2/apply/ (free)

---

## 🚀 Running the Pipeline

### Phase 1 — Story, Script & Character Generation

**Auto mode** (LLM generates everything from your prompt):
```bash
python -m src.main \
  --mode auto \
  --scenes 4 \
  --auto-approve \
  --prompt "A CIA operative in 1960s divided Berlin must extract a Soviet defector before the KGB closes in, uncovering a double agent within his own team"
```

**Manual mode** (provide your own screenplay):
```bash
python -m src.main \
  --mode manual \
  --script-path data/sample_script.txt \
  --auto-approve
```

**Phase 1 outputs saved to `data/`:**
```
data/story_manifest_auto.json
data/scene_manifest_auto.json
data/character_db_auto.json
data/last_script_auto.txt
```

---

### Phase 2 — Audio Generation & Integration

Runs automatically after Phase 1, or can be run independently:
```bash
python -m src.main_phase2 \
  --manifest data/scene_manifest_auto.json \
  --characters data/character_db_auto.json
```

**Phase 2 outputs saved to `data/phase2_runs/run{NN}/`:**
```
data/phase2_runs/run06/
├── audio/
│   ├── scene1/           ← individual per-line MP3 files
│   ├── scene_01_run06.mp3  ← full concatenated scene audio
│   └── ...
├── bgm/
│   ├── tense_scene01_freesound.mp3
│   └── hopeful_scene02_freesound.mp3
└── timing_manifest_run06.json
```

---

### BGM Library (Optional Local Files)

To use your own background music instead of Freesound, place files in `data/bgm_library/` following this naming convention:

```
data/bgm_library/
├── tense_01.mp3
├── tense_02.mp3
├── hopeful_01.mp3
├── mysterious_01.mp3
├── action_01.mp3
└── sad_01.mp3
```

The pipeline will automatically use local files before trying Freesound.

---

### Phase 3 — Video Generation & Composition

Run Phase 3 once Phase 2 has produced a timing manifest and audio assets:
```bash
cd Agentic-Ai
python -m src.main_phase3 \
  --quality balanced \
  --backend auto \
  --enable-subtitles
```

If you need to target a specific Phase 2 run manifest:
```bash
python -m src.main_phase3 \
  --quality balanced \
  --backend auto \
  --timing data/phase2_runs/run19/timing_manifest_run19.json
```

Phase 3 outputs are written into `data/phase3_runs/runXX/`:
```
data/phase3_runs/runXX/
  images/
  clips/
  composed/
  final_output_runXX.mp4
  final_output_runXX_subbed.mp4
  phase3_outputs_runXX.json
```

> Note: Phase 3 uses image generation backends with `auto` fallback from HF → Pollinations. If Pollinations returns `429`, the backend is rate-limiting requests; retry later or use `--backend hf` with a valid HF key.

---

### Phase 4 — Web App & Pipeline Orchestration

Start the Phase 4 web interface from the `Agentic-Ai` folder:
```bash
cd Agentic-Ai
python phase4_flask_app.py
```

Then open the app in your browser:
```
http://localhost:5050
```

The web app provides:
- full pipeline control from prompt to final video
- live SSE progress logs for Phase 1–3
- human approval gating and phase rerun controls
- MCP-style tool calling for dynamic capabilities
- refresh and download of the latest generated video

If the app cannot import `src`, make sure you run it from the `Agentic-Ai` directory.

---

### Phase 5 — Intelligent Edit & Undo Agent

Phase 5 is integrated into the Phase 4 web app. After running a pipeline, use the **Edit Agent** panel to submit natural-language edit commands:

```
Examples:
  "Make scene 1 darker"
  "Apply sepia filter to scene 2"
  "Speed up scene 3"
  "Change voice tone to whispered"
  "Remove the subtitles"
  "Regenerate the script"
```

The **Version History** panel shows all snapshots. Click **Revert** on any version to restore that state and its assets.

You can also run Phase 5 programmatically:
```python
from src.workflows.langgraph_phase5 import run_edit

result = run_edit("Apply sepia filter to scene 1")
print(result)  # {"status": "completed", "intent": ..., "result_summary": ...}
```

Run Phase 5 tests:
```bash
cd Agentic-Ai
python -m pytest tests/test_phase5.py -v
```

---

## 🎧 Sample Outputs

Phase 1 and Phase 2 outputs from the Berlin spy thriller prompt are included in the `data/` directory:

| File | Description |
|------|-------------|
| `data/story_manifest_auto.json` | 4-act story structure for "Berlin Divide" |
| `data/scene_manifest_auto.json` | 4 scenes with dialogue and visual cues |
| `data/character_db_auto.json` | JACK, LARISA, RYAN, VICTORIA, IVAN profiles |
| `data/last_script_auto.txt` | Raw generated screenplay text |
| `data/phase2_runs/run06/` | Audio files + timing manifest from latest run |

---

## 👥 Team

| Member | Role | Phases |
|--------|------|--------|
| Member 1 | Story & Script Agent | Phase 1 + Phase 2 |
| Member 2 | Video Composition | Phase 3 |
| Member 3 | Full-Stack Web App | Phase 4 |
| Member 4 | Edit Agent & Versioning | Phase 5 |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ for Agentic AI Course 2026 — FAST-NUCES Islamabad</p>