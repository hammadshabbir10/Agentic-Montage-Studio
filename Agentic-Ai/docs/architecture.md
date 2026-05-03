# System Architecture

The Agentic Montage Studio pipeline is split into modular, independently runnable phases. Each phase is a LangGraph DAG that consumes JSON contracts produced by the previous phase and emits its own contracts for the next.

## Phase 1 — Story, Script & Characters

LangGraph supervisor-worker workflow.

- **mode_selector**: manual vs auto generation
- **story_node**: generates the story manifest (logline, themes, acts, protagonist)
- **scriptwriter / validator**: produces the screenplay text and parsed scene manifest
- **hitl**: human-in-the-loop checkpoint (or `--auto-approve`)
- **character_node**: extracts character roster with personality and appearance
- **image_node**: optional placeholder character images (SVG)
- **memory_commit**: persists artifacts to the vector store

Outputs (root `data/`):

- `story_manifest_<mode>.json`
- `scene_manifest_<mode>.json`
- `character_db_<mode>.json`
- `last_script_<mode>.txt`

## Phase 2 — Audio Generation & Integration

LangGraph DAG: `scene_parser → voice_synth → music_select → timing_build → memory_commit`.

- Voice consistency via a global voice map (same character → same voice across all scenes)
- BGM via Freesound (with local library and silence fallbacks)
- `timing_manifest_runXX.json` with absolute `start_ms` / `end_ms` per line and per scene

Outputs (`data/phase2_runs/runXX/`):

- `audio/scene_NN_runXX.mp3` and `audio/sceneN/<speaker>_lineNNN.mp3`
- `bgm/<mood>_sceneNN_freesound.mp3`
- `timing_manifest_runXX.json`

## Phase 3 — Video Generation & Composition

LangGraph DAG: `scene_parser → image_gen → motion → compose → mux → subtitles? → memory_commit`.

- **scene_parser** validates Phase 1 + 2 contracts and builds `ScenePlan` objects
- **image_gen** generates one cinematic still per scene (HF Inference API primary, Pollinations fallback) with prompt continuity across scenes (location + mood + character appearance + global style anchor) and an on-disk image cache keyed by prompt + size + seed
- **motion** runs ffmpeg `zoompan` to produce a Ken Burns animated MP4 sized to the timing manifest's duration
- **compose** mixes the silent clip with the per-scene voice track and BGM (BGM ducked relative to voice)
- **mux** concatenates per-scene composed clips losslessly via the concat demuxer
- **subtitles** (optional) builds an SRT from the timing manifest's lines and burns it into the final MP4
- **memory_commit** persists Phase 3 outputs to the vector store

Outputs (`data/phase3_runs/runXX/`):

- `images/`, `clips/`, `composed/`
- `final_output_runXX.mp4` (and `..._subbed.mp4` if subtitles enabled)
- `image_prompts_runXX.json`, `phase3_state_runXX.json`, `phase3_outputs_runXX.json`
- `ffmpeg_commands_runXX.log` (every ffmpeg invocation, for audit)

## MCP Tool Discovery

Tools are loaded dynamically from `data/mcp_registry.json` at runtime. Phase 3 added two new tool types:

- `hf_image` — Hugging Face Inference API
- `pollinations_image` — Pollinations (no API key)

This keeps backends pluggable without touching the agent code.

## Data Flow Across Phases

```
prompt
  → Phase 1 → story_manifest + scene_manifest + character_db
  → Phase 2 → timing_manifest (+ audio + bgm)
  → Phase 3 → final_output.mp4 (+ images + clips + composed scenes)
```
