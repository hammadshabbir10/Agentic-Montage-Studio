# Agent Definitions

## Phase 1

### Scriptwriter Agent
- Segments scenes, writes dialogue, injects visual cues
- Uses MCP capability: `generate_script_segment`

### Script Validator Agent
- Verifies scene headings and dialogue labels

### Human-in-the-Loop Agent
- Confirms alignment with user intent before continuing

### Character Designer Agent
- Extracts characters and maintains consistent identity metadata

### Image Synthesizer Agent
- Generates placeholder character visuals (SVG)

## Phase 2

### Voice Synthesizer Agent
- Per-line TTS via edge-tts with a global voice map for character consistency

### Music Selector Agent
- Detects scene mood (keyword-based) and fetches royalty-free BGM from Freesound

### Timing Manifest Builder
- Produces per-line `start_ms` / `end_ms` and a scene-level timeline

## Phase 3

### Scene Visualizer Agent  (`src/agents/scene_visualizer.py`)
- Builds a continuity-preserving prompt from location + mood + visual cue + character appearance + a film-wide style anchor
- Uses MCP capabilities `generate_scene_image` (HF) and `generate_scene_image_fallback` (Pollinations)
- Caches images on disk by `sha1(prompt + WxH + seed)`

### Ken Burns Motion Agent  (`src/utils/video_compose.py`)
- Renders a slow zoom (in or out) using ffmpeg `zoompan` filter
- Clip duration is read from the timing manifest

### Scene Compositor Agent  (`src/utils/video_compose.py`)
- Mixes the silent clip with the scene voice track and BGM
- BGM ducked relative to voice; voice always at full volume

### Final Mux Agent  (`src/utils/video_compose.py`)
- Concatenates per-scene composed MP4s losslessly via the concat demuxer

### Subtitle Agent  (`src/utils/video_compose.py`)
- Generates an SRT from the timing manifest's per-line `start_ms` / `end_ms`
- Optionally burns the SRT into the final MP4 with a configurable style
