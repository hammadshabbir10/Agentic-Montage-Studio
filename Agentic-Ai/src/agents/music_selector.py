"""
music_selector.py  –  Phase 2 BGM Selector

Priority chain for each scene:
  1. Local BGM library   data/bgm_library/mood_XX.mp3
  2. Freesound API       real royalty-free MP3 downloaded by mood
  3. MusicGen (Meta)     local generation (only if ENABLE_MUSICGEN=true)
  4. Silence stub        always works, last resort

Setup
-----
pip install requests
pip install audiocraft torch torchaudio   # only needed if ENABLE_MUSICGEN=true

.env
----
FREESOUND_API_KEY=your_key_here    # free at https://freesound.org/apiv2/apply/
ENABLE_MUSICGEN=false
"""

import os
import wave
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

# ── Mood configuration ────────────────────────────────────────────────────────
_MOOD_CONFIG: Dict[str, Dict] = {
    "tense": {
        # Multiple queries tried in order until one returns results
        "queries":  [
            "tense cinematic background music",
            "thriller suspense music no vocals",
            "dark cinematic underscore",
            "tension music orchestral",
        ],
        "duration": 30,
    },
    "action": {
        "queries":  [
            "action cinematic background music",
            "intense chase music no vocals",
            "epic action orchestral",
        ],
        "duration": 30,
    },
    "mysterious": {
        "queries":  [
            "mysterious ambient background music",
            "dark ambient cinematic no vocals",
            "suspense mystery music",
        ],
        "duration": 30,
    },
    "sad": {
        "queries":  [
            "sad emotional cinematic piano",
            "melancholic background music no vocals",
            "emotional orchestra underscore",
        ],
        "duration": 25,
    },
    "hopeful": {
        "queries":  [
            "hopeful uplifting cinematic background music orchestral",
            "inspiring orchestral music no vocals",
            "uplifting cinematic underscore",
        ],
        "duration": 25,
    },
    "neutral": {
        "queries":  [
            "ambient background music cinematic",
            "neutral underscore music no vocals",
        ],
        "duration": 20,
    },
}

# ── Mood detection ────────────────────────────────────────────────────────────
_MOOD_KEYWORDS: Dict[str, List[str]] = {
    "tense":      ["kgb", "danger", "spy", "threat", "caught", "closing", "escape",
                   "safe house", "headquarters", "interrogat", "agent", "double",
                   "mole", "compromised", "surveillance", "extract"],
    "action":     ["fight", "chase", "explosion", "shoot", "battle", "attack",
                   "move now", "go go", "run", "fire", "weapon"],
    "mysterious": ["secret", "unknown", "shadow", "dark", "hidden", "watch",
                   "observe", "plan", "double agent", "signal", "transmission"],
    "sad":        ["family", "scared", "afraid", "worried", "cry", "death",
                   "loss", "miss", "goodbye", "sorry", "alone"],
    "hopeful":    ["promise", "together", "believe", "trust", "better", "safe",
                   "free", "future", "hope", "finally"],
    "neutral":    [],
}


def _detect_mood(task: Dict) -> str:
    text_blob = " ".join([
        task.get("location", ""),
        " ".join(d.get("line", "")       for d in task.get("dialogue", [])),
        " ".join(d.get("visual_cue", "") for d in task.get("dialogue", [])),
    ]).lower()

    scores: Dict[str, int] = {mood: 0 for mood in _MOOD_KEYWORDS}
    for mood, keywords in _MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in text_blob:
                scores[mood] += 1

    best = max(
        (m for m in scores if m != "neutral"),
        key=lambda m: scores[m],
        default="tense",
    )
    return best if scores[best] > 0 else "tense"


# ── Source 1: Local library ───────────────────────────────────────────────────
def _find_local_bgm(mood: str, bgm_library: Path) -> Optional[str]:
    if not bgm_library.exists():
        return None
    for suffix in (".mp3", ".wav", ".ogg"):
        matches = sorted(bgm_library.glob(f"{mood}_*{suffix}"))
        if matches:
            return str(matches[0])
    return None


# ── Source 2: Freesound API ───────────────────────────────────────────────────
def _download_freesound(mood: str, scene_id: int, out_dir: Path) -> Optional[str]:
    """
    Search Freesound and download the best matching preview MP3.

    Freesound auth:
      - Search endpoint: pass token as query param  (?token=KEY)
      - Download/preview: pass OAuth2 token in Authorization header
        BUT preview URLs are public CDN links — no auth needed to download them.
    """
    api_key = os.getenv("FREESOUND_API_KEY", "").strip()
    if not api_key:
        LOGGER.warning("FREESOUND_API_KEY not set — skipping Freesound")
        print("[BGM] FREESOUND_API_KEY not set — skipping Freesound download")
        return None

    config  = _MOOD_CONFIG.get(mood, _MOOD_CONFIG["neutral"])
    queries = config.get("queries", [config.get("query", mood)])

    search_url = "https://freesound.org/apiv2/search/text/"

    try:
        for query in queries:
            LOGGER.info("Freesound search: mood=%s query=%r", mood, query)
            print(f"[BGM] Freesound search: mood={mood!r} query={query!r}")

            params = {
                "query":     query,
                "token":     api_key,
                "fields":    "id,name,previews,duration,license",
                "filter":    "duration:[15 TO 120]",
                "page_size": 5,
                "sort":      "rating_desc",
            }

            resp = requests.get(search_url, params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("results", [])

            if not results:
                LOGGER.warning("Freesound: no results for query=%r, trying next…", query)
                print(f"[BGM] Freesound: no results for {query!r}, trying next query…")
                continue

            for result in results:
                previews    = result.get("previews", {})
                preview_url = (
                    previews.get("preview-hq-mp3")
                    or previews.get("preview-lq-mp3")
                )
                if not preview_url:
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{mood}_scene{scene_id:02d}_freesound.mp3"

                LOGGER.info("Freesound download: %s → %s", preview_url, out_path)
                print(f"[BGM] Downloading from Freesound: {preview_url}")

                dl = requests.get(preview_url, timeout=30)
                dl.raise_for_status()
                out_path.write_bytes(dl.content)

                print(
                    f"[BGM] Freesound OK: '{result.get('name','')}' "
                    f"({result.get('duration', 0):.1f}s) → {out_path}"
                )
                return str(out_path)

        LOGGER.warning("Freesound: no downloadable preview found for mood=%s", mood)
        print(f"[BGM] Freesound: exhausted all queries for mood={mood!r}, falling back")
        return None

    except requests.exceptions.HTTPError as exc:
        # Print the response body so we can see the exact Freesound error
        body = exc.response.text if exc.response is not None else "no body"
        LOGGER.warning("Freesound HTTP error: %s | body: %s", exc, body)
        print(f"[BGM] Freesound HTTP error: {exc}\n      Response: {body[:300]}")
        return None
    except Exception as exc:
        LOGGER.warning("Freesound download failed: %s", exc)
        print(f"[BGM] Freesound failed: {exc}")
        return None


# ── Source 3: MusicGen ────────────────────────────────────────────────────────
def _generate_musicgen(mood: str, scene_id: int, out_dir: Path) -> Optional[str]:
    if os.getenv("ENABLE_MUSICGEN", "false").lower() != "true":
        return None
    try:
        from audiocraft.data.audio import audio_write
        from audiocraft.models import MusicGen
    except ImportError:
        LOGGER.warning("MusicGen not installed – skipping")
        print("[BGM] MusicGen not installed. Run: pip install audiocraft torch torchaudio")
        return None

    config   = _MOOD_CONFIG.get(mood, _MOOD_CONFIG["neutral"])
    prompt   = config["query"]
    duration = config["duration"]
    LOGGER.info("MusicGen generating: mood=%s duration=%ds", mood, duration)
    print(f"[BGM] MusicGen generating {duration}s of {mood!r} music…")

    try:
        model = MusicGen.get_pretrained("facebook/musicgen-small")
        model.set_generation_params(duration=duration)
        wav = model.generate([prompt])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_stem = out_dir / f"{mood}_scene{scene_id:02d}_musicgen"
        audio_write(str(out_stem), wav[0].cpu(), model.sample_rate, strategy="loudness")
        out_path = Path(str(out_stem) + ".wav")
        print(f"[BGM] MusicGen saved: {out_path}")
        return str(out_path)
    except Exception as exc:
        LOGGER.warning("MusicGen failed: %s", exc)
        print(f"[BGM] MusicGen failed: {exc}")
        return None


# ── Source 4: Silence stub ────────────────────────────────────────────────────
def _write_silence_stub(scene_id: int, mood: str, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bgm_scene_{scene_id:02d}_{mood}_stub.wav"
    if not path.exists():
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 22050)   # 1 second silence
    print(f"[BGM] Scene {scene_id} → STUB (no BGM available): {path}")
    return str(path)


# ── Public entry point ────────────────────────────────────────────────────────
def run(
    task: Dict,
    bgm_library_dir: str = "data/bgm_library",
    audio_dir: str = "data/audio",
    bgm_dir: Optional[str] = None,
) -> Dict:
    """
    Parameters
    ----------
    task            : scene dict  { scene_id, location, dialogue, ... }
    bgm_library_dir : path to local royalty-free BGM library folder
    audio_dir       : fallback audio root (used when bgm_dir is None)
    bgm_dir         : run-specific BGM output folder (preferred)

    Returns
    -------
    { scene_id, mood, bgm_path, bgm_source }
    bgm_source is one of: "local" | "freesound" | "musicgen" | "stub"
    """
    scene_id = task.get("scene_id", 0)
    mood     = _detect_mood(task)
    bgm_lib  = Path(bgm_library_dir)
    bgm_out  = Path(bgm_dir) if bgm_dir else Path(audio_dir) / "bgm"

    LOGGER.info("Scene %s detected mood: %s", scene_id, mood)
    print(f"[BGM] Scene {scene_id} mood detected: {mood!r}")

    # 1. Local library
    bgm_path = _find_local_bgm(mood, bgm_lib)
    if bgm_path:
        print(f"[BGM] Scene {scene_id} → LOCAL: {bgm_path}")
        return {"scene_id": scene_id, "mood": mood,
                "bgm_path": bgm_path, "bgm_source": "local"}

    # 2. Freesound API
    bgm_path = _download_freesound(mood, scene_id, bgm_out)
    if bgm_path:
        print(f"[BGM] Scene {scene_id} → FREESOUND: {bgm_path}")
        return {"scene_id": scene_id, "mood": mood,
                "bgm_path": bgm_path, "bgm_source": "freesound"}

    # 3. MusicGen
    bgm_path = _generate_musicgen(mood, scene_id, bgm_out)
    if bgm_path:
        print(f"[BGM] Scene {scene_id} → MUSICGEN: {bgm_path}")
        return {"scene_id": scene_id, "mood": mood,
                "bgm_path": bgm_path, "bgm_source": "musicgen"}

    # 4. Silence stub
    bgm_path = _write_silence_stub(scene_id, mood, bgm_out)
    return {"scene_id": scene_id, "mood": mood,
            "bgm_path": bgm_path, "bgm_source": "stub"}