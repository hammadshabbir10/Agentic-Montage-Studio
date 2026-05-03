"""
voice_synthesizer.py  –  Phase 2 Audio Agent

Key fix: voice assignment is now GLOBAL across all scenes.
The same character always gets the same voice regardless of which scene
they appear in.  Call build_global_voice_map(manifest) once before
processing any scenes, then pass the map into run().

Rules
-----
* Speaker name ends in 'A' (first word, uppercase) → female voice pool
* Otherwise                                         → male  voice pool
* Each UNIQUE character gets a DIFFERENT voice (round-robin within pool)
* Segments are written as individual MP3s inside a per-scene sub-folder
  AND concatenated into one scene-level MP3 for the timing manifest.
"""

import asyncio
import re
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import edge_tts

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

# ── Voice pools ───────────────────────────────────────────────────────────────
_MALE_VOICES: List[str] = [
    "en-US-GuyNeural",        # deep, authoritative
    "en-US-AndrewNeural",     # calm, measured
    "en-US-BrianNeural",      # warm, natural
    "en-GB-RyanNeural",       # British, refined
    "en-AU-WilliamNeural",    # Australian, firm
    "en-US-ChristopherNeural",# clear, confident
    "en-US-EricNeural",       # friendly, direct
]

_FEMALE_VOICES: List[str] = [
    "en-US-JennyNeural",      # warm, conversational
    "en-US-AriaNeural",       # expressive, young
    "en-US-MichelleNeural",   # clear, professional
    "en-GB-SoniaNeural",      # British, elegant
    "en-AU-NatashaNeural",    # Australian, bright
    "en-US-EmmaNeural",       # natural, friendly
    "en-US-AvaNeural",        # smooth, articulate
]

# ── Unicode normalisation ─────────────────────────────────────────────────────
_UNICODE_MAP = {
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\u201c": '"', "\u201d": '"',   # curly double quotes
    "\u2032": "'", "\u2033": '"',   # prime / double-prime
    "\u0060": "'", "\u00b4": "'",   # grave / acute
    "\u2013": "-", "\u2014": "-",   # en-dash / em-dash
    "\u2026": "...",                # ellipsis
}

def _normalize_text(text: str) -> str:
    for bad, good in _UNICODE_MAP.items():
        text = text.replace(bad, good)
    return text

# ── Known name gender override lists ─────────────────────────────────────────
# Names that break the "ends in A = female" rule. Extend as needed.
_KNOWN_FEMALE_NAMES = {
    "RACHEL", "SARAH", "JENNIFER", "ELIZABETH", "HELEN", "CLAIRE",
    "ISABEL", "CAROL", "CHERYL", "ABIGAIL", "MARGARET", "RUTH",
    "JUDITH", "DEBORAH", "JANET", "DIANE", "KATHLEEN", "JOYCE",
    "ALICE", "MARIE", "ANNE", "ROSE", "GRACE", "FAITH", "HOPE",
    "JOY", "DAWN", "BROOKE", "PAIGE", "LEIGH", "SIMONE", "EVE",
    "CLAIRE", "IRENE", "YVONNE", "COLETTE", "BRIGITTE", "INGRID",
}

_KNOWN_MALE_NAMES = {
    # Male names that accidentally end in A
    "JOSHUA", "EZRA", "NOAH", "LUCA", "NICOLA", "ANDREA", "COSTA",
    "SILVA", "VILLA", "GARCIA", "BARCA", "ZETA", "STRATA",
}


# ── Gender detection ──────────────────────────────────────────────────────────
def _is_female(speaker: str) -> bool:
    """
    Priority:
    1. Known-female list  (catches RACHEL, SARAH, etc.)
    2. Known-male list    (catches NOAH, LUCA, etc.)
    3. Canonical name ends with A  (project naming rule)

    Both the canonical (core) name AND the first word of the raw name
    are checked against the lists, so "RACHEL JENKINS" → female even
    though canonical is JENKINS.
    """
    canon      = _extract_core_name(speaker)
    first_word = speaker.strip().upper().split()[0] if speaker.strip() else ""

    if canon in _KNOWN_FEMALE_NAMES or first_word in _KNOWN_FEMALE_NAMES:
        return True
    if canon in _KNOWN_MALE_NAMES or first_word in _KNOWN_MALE_NAMES:
        return False

    return canon.endswith("A")


# ── Name canonicalization ─────────────────────────────────────────────────────
def _extract_core_name(raw: str) -> str:
    """
    Strip role prefixes/suffixes so variants of the same character map to
    one canonical key.

    Examples
    --------
    "CIA OPERATIVE JACKSON"   → "JACKSON"
    "JACKSON'S TEAM LEAD, RYAN" → "RYAN"
    "SOVIET DEFECTOR, LARISA" → "LARISA"
    "KGB AGENT, IVAN"         → "IVAN"
    "JACKSON"                 → "JACKSON"
    "LARISA"                  → "LARISA"

    Rules (applied in order):
    1. Uppercase and strip.
    2. If the name contains a comma, take the LAST comma-separated token
       (e.g. "SOVIET DEFECTOR, LARISA" → "LARISA").
    3. Take the LAST whitespace-separated word — that's almost always the
       personal name rather than a role descriptor.
    4. Strip possessive 's  (e.g. "JACKSON'S" → "JACKSON").
    """
    name = raw.strip().upper()

    # Step 2: comma → take last token
    if "," in name:
        name = name.split(",")[-1].strip()

    # Step 3: take last word
    words = name.split()
    name  = words[-1] if words else name

    # Step 4: strip possessive
    name = re.sub(r"'S$", "", name)

    return name


# ── Global voice map builder ──────────────────────────────────────────────────
def build_global_voice_map(manifest: Dict) -> Dict[str, str]:
    """
    Walk ALL scenes in the manifest, canonicalize every speaker name,
    and assign ONE distinct voice per unique character.

    Returns TWO parallel dicts bundled together:
      {
        "by_canonical": { "JACKSON": "en-US-GuyNeural", ... },
        "by_raw":       { "CIA OPERATIVE JACKSON": "en-US-GuyNeural",
                          "JACKSON": "en-US-GuyNeural", ... }
      }

    Use "by_raw" for lookups during synthesis (exact match first),
    falling back to the canonical key if the exact raw name isn't found.

    Public helper  voice_for(raw_name, voice_map)  handles this automatically.
    """
    # 1. Collect all raw names in first-appearance order
    raw_names_ordered: List[str] = []
    for scene in manifest.get("scenes", []):
        for d in scene.get("dialogue", []):
            raw = d.get("speaker", "").strip()
            if raw and raw not in raw_names_ordered:
                raw_names_ordered.append(raw)

    # 2. Map raw → canonical, group raw names that share a canonical name
    #    canonical → [raw1, raw2, ...]  (first raw = canonical representative)
    canonical_to_raws: Dict[str, List[str]] = {}
    raw_to_canonical:  Dict[str, str]       = {}

    for raw in raw_names_ordered:
        canon = _extract_core_name(raw)
        raw_to_canonical[raw] = canon
        if canon not in canonical_to_raws:
            canonical_to_raws[canon] = []
        canonical_to_raws[canon].append(raw)

    # 3. Assign one voice per canonical character
    #    Use the FIRST raw variant to decide gender
    by_canonical: Dict[str, str] = {}
    male_idx   = 0
    female_idx = 0

    for canon, raws in canonical_to_raws.items():
        gender_ref = raws[0]          # use first appearance to decide gender
        if _is_female(gender_ref):
            voice = _FEMALE_VOICES[female_idx % len(_FEMALE_VOICES)]
            female_idx += 1
        else:
            voice = _MALE_VOICES[male_idx % len(_MALE_VOICES)]
            male_idx += 1
        by_canonical[canon] = voice

    # 4. Build raw → voice lookup (all variants of same character → same voice)
    by_raw: Dict[str, str] = {
        raw: by_canonical[raw_to_canonical[raw]]
        for raw in raw_names_ordered
    }

    # Print assignment table
    print("[Voice] Character voice assignments (canonical -> voice):")
    for canon, voice in by_canonical.items():
        gender = "F" if _is_female(canonical_to_raws[canon][0]) else "M"
        aliases = canonical_to_raws[canon]
        alias_str = ", ".join(aliases) if len(aliases) > 1 else aliases[0]
        print(f"  [{gender}] {canon:20s} -> {voice:30s}  (raw: {alias_str})")

    return {"by_canonical": by_canonical, "by_raw": by_raw}


def voice_for(raw_name: str, voice_map: Dict) -> str:
    """
    Resolve a raw speaker name to its assigned voice.
    Works with both the old flat dict and the new nested dict from
    build_global_voice_map().
    """
    # New format: nested dict with by_raw / by_canonical
    if "by_raw" in voice_map:
        by_raw       = voice_map["by_raw"]
        by_canonical = voice_map["by_canonical"]
        if raw_name in by_raw:
            return by_raw[raw_name]
        # Fallback: try canonical lookup
        canon = _extract_core_name(raw_name)
        if canon in by_canonical:
            return by_canonical[canon]
        # Last resort: gender-based default
        return _FEMALE_VOICES[0] if _is_female(raw_name) else _MALE_VOICES[0]

    # Old format: flat { name: voice }
    if raw_name in voice_map:
        return voice_map[raw_name]
    canon = _extract_core_name(raw_name)
    return voice_map.get(canon, _MALE_VOICES[0])

# ── Clean a dialogue line ─────────────────────────────────────────────────────
def _clean_line(line: str) -> str:
    line = re.sub(r"^\([^)]*\)\s*", "", line).strip()   # strip (stage directions)
    line = line.strip('"').strip("'").strip()            # strip outer quotes
    line = _normalize_text(line)                         # fix Unicode
    return line

# ── Async TTS helpers ─────────────────────────────────────────────────────────
async def _synth_segment(
    text: str,
    voice: str,
    out_path: Path,
    retries: int = 5,
) -> None:
    for attempt in range(retries):
        try:
            comm = edge_tts.Communicate(text, voice=voice)
            await comm.save(str(out_path))
            return
        except Exception as exc:
            LOGGER.warning(
                "TTS attempt %d/%d failed (voice=%s): %s",
                attempt + 1, retries, voice, exc,
            )
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt + 1)


async def _synthesize_scene_async(
    dialogue: List[Dict],
    voice_map: Dict[str, str],
    scene_folder: Path,     # per-scene sub-folder for individual line files
    concat_path: Path,      # scene-level concatenated MP3
    scene_id: int,
) -> Tuple[str, List[Dict]]:
    """
    Synthesise every line, write individual MP3s to scene_folder,
    then concatenate into concat_path.

    Returns (concat_path_str, segments_meta).
    """
    clean_lines: List[Tuple] = []
    for idx, entry in enumerate(dialogue or []):
        speaker  = entry.get("speaker", "").strip()
        raw_line = entry.get("line", "")
        line     = _clean_line(raw_line)
        if not line:
            continue
        voice     = voice_for(speaker, voice_map)
        # Individual line file inside the scene sub-folder
        safe_speaker = re.sub(r"[^a-zA-Z0-9_]+", "_", speaker)
        line_path = scene_folder / f"{safe_speaker}_line{idx + 1:03d}.mp3"
        clean_lines.append((speaker, line, voice, line_path, entry.get("visual_cue", "")))

    # ── Silence fallback ──────────────────────────────────────────────────────
    if not clean_lines:
        silence = concat_path.with_suffix(".wav")
        with wave.open(str(silence), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 22050)
        return str(silence), []

    # ── Synthesise all lines concurrently ─────────────────────────────────────
    print(f"[Phase 2] voice_synth: scene {scene_id} - synthesising "
          f"{len(clean_lines)} line(s) concurrently...")

    await asyncio.gather(*[
        _synth_segment(line, voice, path)
        for (_, line, voice, path, _) in clean_lines
    ])

    # ── Concatenate into one scene MP3 ────────────────────────────────────────
    segments_meta: List[Dict] = []
    combined = bytearray()
    for (speaker, line, voice, path, visual_cue) in clean_lines:
        data = path.read_bytes()
        segments_meta.append({
            "speaker":     speaker,
            "line":        line,
            "voice":       voice,
            "visual_cue":  visual_cue,
            "audio_file":  str(path),        # individual line path
            "byte_offset": len(combined),
            "byte_length": len(data),
        })
        combined.extend(data)

    concat_path.write_bytes(combined)
    return str(concat_path), segments_meta


# ── Public entry point ────────────────────────────────────────────────────────
def run(
    task: Dict,
    tool_client,
    audio_dir: str,
    run_tag: Optional[str] = None,
    global_voice_map: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Parameters
    ----------
    task              : scene dict { scene_id, dialogue, location, ... }
    tool_client       : passed through (unused for edge-tts)
    audio_dir         : root audio directory for this run
    run_tag           : e.g. "run05"
    global_voice_map  : pre-built map from build_global_voice_map().
                        If None, a local map is built from this scene only
                        (use only for single-scene testing).

    Returns
    -------
    { scene_id, path, segments }
    """
    scene_id = task.get("scene_id", 0)
    dialogue = task.get("dialogue", [])

    # Use global map if provided, else fall back to scene-local assignment
    if global_voice_map is not None:
        voice_map = global_voice_map
    else:
        voice_map = _build_local_voice_map(dialogue)

    speakers = {d.get("speaker", "") for d in dialogue if d.get("speaker")}
    LOGGER.info("Scene %s speakers: %s", scene_id, speakers)

    # Directory layout:  audio_dir/scene{N}/
    scene_folder = Path(audio_dir) / f"scene{scene_id}"
    scene_folder.mkdir(parents=True, exist_ok=True)

    suffix      = f"_{run_tag}" if run_tag else ""
    concat_path = Path(audio_dir) / f"scene_{scene_id:02d}{suffix}.mp3"

    path, segments = asyncio.run(
        _synthesize_scene_async(
            dialogue, voice_map, scene_folder, concat_path, scene_id
        )
    )

    LOGGER.info("Voice synth complete: scene %s to %s", scene_id, path)
    print(f"[Phase 2] voice_synth: scene {scene_id} to {path} ({len(segments)} segment(s))")

    return {"scene_id": scene_id, "path": path, "segments": segments}


# ── Internal helpers ──────────────────────────────────────────────────────────
def _build_local_voice_map(dialogue: List[Dict]) -> Dict:
    """Fallback: build voice map from a single scene. Returns nested format."""
    fake_manifest = {"scenes": [{"dialogue": dialogue}]}
    return build_global_voice_map(fake_manifest)