"""
scene_visualizer.py  –  Phase 3 Scene Image Generation Agent

Generates one cinematic still per scene, used by the Ken Burns animator.

Backends (no GPU required, no paid APIs)
----------------------------------------
1. Hugging Face Inference API   (primary)
   - Reads HF_TOKEN and HF_IMAGE_MODEL from environment
   - Default model: black-forest-labs/FLUX.1-schnell
2. Pollinations.ai              (fallback, no key required)
   - URL pattern: https://image.pollinations.ai/prompt/{prompt}

Prompt continuity
-----------------
For each scene we combine:
  - Scene location
  - Scene mood
  - Top visual cue
  - Character appearance descriptors (from Phase 1 character_db)
  - A consistent global "style anchor" (so all scenes look like the same film)
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.io.phase3_contracts import ScenePlan
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


# ── Defaults & configuration ────────────────────────────────────────────────
_DEFAULT_HF_MODEL = "black-forest-labs/FLUX.1-schnell"
_HF_TIMEOUT_SEC = 90
_POLLINATIONS_TIMEOUT_SEC = 120
_HF_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{model}"
_POLLINATIONS_URL_TEMPLATE = "https://image.pollinations.ai/prompt/{prompt}"

# Resolution presets per quality profile (width, height)
_QUALITY_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "fast":      (960,  540),
    "balanced":  (1280, 720),
    "cinematic": (1920, 1080),
}

# A film-wide style anchor — ensures every scene shares an aesthetic
_GLOBAL_STYLE_ANCHOR = (
    "cinematic still, high detail, dramatic lighting, depth of field, "
    "film grain, 35mm photography, color graded"
)


# ── Prompt builder ──────────────────────────────────────────────────────────

def build_scene_prompt(
    plan: ScenePlan,
    character_db: Dict[str, Any],
    style_anchor: Optional[str] = None,
) -> str:
    """
    Build a scene image prompt that emphasises continuity:
      - location
      - mood
      - first visual cue
      - character appearance (only the speakers in this scene)
      - shared style anchor
    """
    style = style_anchor or _GLOBAL_STYLE_ANCHOR

    char_lookup = {
        c.get("name", "").upper(): c
        for c in character_db.get("characters", [])
    }

    char_descriptors: List[str] = []
    for speaker in plan.speakers[:3]:                # at most 3 to keep prompt tight
        char = char_lookup.get(speaker.upper())
        if not char:
            continue
        appearance = char.get("appearance", "").strip().rstrip(".")
        if appearance:
            char_descriptors.append(f"{speaker.title()}: {appearance}")

    visual_cue = plan.visual_cues[0] if plan.visual_cues else ""

    parts: List[str] = []
    if plan.location:
        parts.append(f"Location: {plan.location.title()}")
    if plan.mood and plan.mood != "neutral":
        parts.append(f"Mood: {plan.mood}")
    if visual_cue:
        parts.append(f"Shot: {visual_cue}")
    if char_descriptors:
        parts.append("Characters — " + "; ".join(char_descriptors))
    parts.append(style)

    return " | ".join(parts)


def build_line_prompt(
    plan: ScenePlan,
    line: Dict[str, Any],
    character_db: Dict[str, Any],
    portrait_bank: Optional[Dict[str, Dict[str, Any]]] = None,
    style_anchor: Optional[str] = None,
) -> str:
    """
    Build a speaker-focused prompt:
      - single speaking character only
      - scene background/location
      - per-line visual cue + spoken text context
    """
    style = style_anchor or _GLOBAL_STYLE_ANCHOR
    speaker = str(line.get("speaker", "")).strip()
    spoken_line = str(line.get("line", "")).strip().strip('"')
    cue = str(line.get("visual_cue", "")).strip()

    char_lookup = {
        c.get("name", "").upper(): c
        for c in character_db.get("characters", [])
    }
    char = char_lookup.get(speaker.upper(), {})
    appearance = str(char.get("appearance", "")).strip()
    personality = str(char.get("personality", "")).strip()

    parts: List[str] = []
    anchor_token = speaker.upper().replace(" ", "_")
    anchor = (portrait_bank or {}).get(speaker.upper(), {})
    anchor_traits = str(anchor.get("anchor_traits", "")).strip()

    parts.append(
        f"Single-character cinematic shot of {speaker}. "
        "Only one person visible, no second character in frame."
    )
    parts.append(
        f"Identity anchor token: {anchor_token}. Keep this exact face identity consistent."
    )
    if plan.location:
        parts.append(f"Background/setting: {plan.location.title()}")
    if plan.mood and plan.mood != "neutral":
        parts.append(f"Mood: {plan.mood}")
    if appearance:
        parts.append(f"Appearance: {appearance}")
    if personality:
        parts.append(f"Personality expression: {personality}")
    if anchor_traits:
        parts.append(f"Identity traits: {anchor_traits}")
    if cue:
        parts.append(f"Camera direction: {cue}")
    if spoken_line:
        parts.append(f"Dialogue context: {spoken_line}")
    parts.append(style)
    return " | ".join(parts)


# ── Prompt cache key ────────────────────────────────────────────────────────

def prompt_fingerprint(prompt: str, width: int, height: int, seed: Optional[int]) -> str:
    """Stable fingerprint so identical prompts reuse cached images."""
    blob = f"{prompt}|{width}x{height}|seed={seed if seed is not None else 'none'}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _speaker_seed(base_seed: Optional[int], speaker: str) -> int:
    base = 0 if base_seed is None else int(base_seed)
    return (base + (_stable_int(speaker.upper()) % 100000)) % 2_147_483_647


# ── Backend: Hugging Face ───────────────────────────────────────────────────

def _generate_via_hf(
    prompt: str,
    width: int,
    height: int,
    out_path: Path,
    seed: Optional[int] = None,
    retries: int = 2,
) -> Tuple[bool, str]:
    api_key = os.getenv("HF_TOKEN", "").strip()
    model = os.getenv("HF_IMAGE_MODEL", _DEFAULT_HF_MODEL).strip() or _DEFAULT_HF_MODEL

    if not api_key:
        return False, "HF_TOKEN not set"

    url = _HF_URL_TEMPLATE.format(model=model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/png",
        "Content-Type": "application/json",
    }
    parameters: Dict[str, Any] = {"width": width, "height": height}
    if seed is not None:
        parameters["seed"] = int(seed)

    payload = {"inputs": prompt, "parameters": parameters}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=_HF_TIMEOUT_SEC)
        except requests.RequestException as exc:
            LOGGER.warning("HF request error attempt %d: %s", attempt, exc)
            time.sleep(2 * attempt)
            continue

        if resp.status_code == 200 and resp.content:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(resp.content)
            return True, f"hf:{model}"

        # Model is loading; HF returns 503 with estimated_time
        if resp.status_code == 503:
            try:
                wait = float(resp.json().get("estimated_time", 8))
            except Exception:
                wait = 8.0
            wait = min(max(wait, 4.0), 25.0)
            LOGGER.warning("HF model loading, sleeping %.1fs (attempt %d)", wait, attempt)
            time.sleep(wait)
            continue

        LOGGER.warning(
            "HF non-200 status=%s body=%s",
            resp.status_code,
            resp.text[:200] if resp.text else "<empty>",
        )
        # Fall through and retry
        time.sleep(2 * attempt)

    return False, f"hf:{model} failed after {retries} attempt(s)"


# ── Backend: Pollinations ───────────────────────────────────────────────────

def _generate_via_pollinations(
    prompt: str,
    width: int,
    height: int,
    out_path: Path,
    seed: Optional[int] = None,
) -> Tuple[bool, str]:
    encoded = urllib.parse.quote(prompt, safe="")
    url = _POLLINATIONS_URL_TEMPLATE.format(prompt=encoded)
    params: Dict[str, Any] = {"width": width, "height": height, "nologo": "true"}
    if seed is not None:
        params["seed"] = int(seed)

    try:
        resp = requests.get(url, params=params, timeout=_POLLINATIONS_TIMEOUT_SEC)
    except requests.RequestException as exc:
        return False, f"pollinations error: {exc}"

    if resp.status_code == 200 and resp.content:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        return True, "pollinations"

    return False, f"pollinations status={resp.status_code}"


# ── Public API ──────────────────────────────────────────────────────────────

def resolve_resolution(quality: str) -> Tuple[int, int]:
    return _QUALITY_RESOLUTIONS.get(quality, _QUALITY_RESOLUTIONS["balanced"])


def _generate_with_backend_order(
    prompt: str,
    width: int,
    height: int,
    out_path: Path,
    backend: str,
    seed: Optional[int],
) -> Tuple[bool, str]:
    if backend == "hf":
        return _generate_via_hf(prompt, width, height, out_path, seed=seed)
    if backend == "pollinations":
        return _generate_via_pollinations(prompt, width, height, out_path, seed=seed)
    # auto
    ok, info = _generate_via_hf(prompt, width, height, out_path, seed=seed)
    if ok:
        return ok, info
    return _generate_via_pollinations(prompt, width, height, out_path, seed=seed)


def generate_scene_image(
    plan: ScenePlan,
    character_db: Dict[str, Any],
    images_dir: Path,
    backend: str = "auto",
    quality: str = "balanced",
    seed: Optional[int] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Generate one image for the given ScenePlan and return a result dict.

    Parameters
    ----------
    backend : "hf" | "pollinations" | "auto"
              "auto" means HF first, fall back to Pollinations on failure.
    quality : "fast" | "balanced" | "cinematic"
    """
    width, height = resolve_resolution(quality)
    prompt = build_scene_prompt(plan, character_db)
    plan.image_prompt = prompt

    fp = prompt_fingerprint(prompt, width, height, seed)
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / f"scene_{plan.scene_id:02d}_{fp}.png"

    # Cache hit
    if use_cache and out_path.exists() and out_path.stat().st_size > 0:
        LOGGER.info("scene %s: image cache hit -> %s", plan.scene_id, out_path)
        plan.image_path = str(out_path)
        plan.image_backend = "cache"
        return {
            "scene_id": plan.scene_id,
            "image_path": str(out_path),
            "backend": "cache",
            "prompt": prompt,
            "width": width,
            "height": height,
        }

    ok, info = _generate_with_backend_order(
        prompt=prompt,
        width=width,
        height=height,
        out_path=out_path,
        backend=backend,
        seed=seed,
    )
    if ok:
        plan.image_path = str(out_path)
        plan.image_backend = info
        return {
            "scene_id": plan.scene_id,
            "image_path": str(out_path),
            "backend": info,
            "prompt": prompt,
            "width": width,
            "height": height,
        }

    raise RuntimeError(
        f"scene {plan.scene_id}: all image backends failed. last_error={info}"
    )


def generate_all_scene_images(
    plans: List[ScenePlan],
    character_db: Dict[str, Any],
    images_dir: Path,
    backend: str = "auto",
    quality: str = "balanced",
    seed: Optional[int] = None,
    only_scene_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for plan in plans:
        if only_scene_id is not None and plan.scene_id != only_scene_id:
            continue
        result = generate_scene_image(
            plan,
            character_db,
            images_dir=images_dir,
            backend=backend,
            quality=quality,
            seed=seed,
        )
        results.append(result)
    return results


def generate_scene_line_images(
    plan: ScenePlan,
    character_db: Dict[str, Any],
    images_dir: Path,
    backend: str = "auto",
    quality: str = "balanced",
    seed: Optional[int] = None,
    use_cache: bool = True,
    portrait_bank: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate one image per dialogue line so the active speaker can be shown
    during their own line segment in the final video.
    """
    width, height = resolve_resolution(quality)
    results: List[Dict[str, Any]] = []
    images_dir.mkdir(parents=True, exist_ok=True)

    for idx, line in enumerate(plan.lines, start=1):
        prompt = build_line_prompt(
            plan=plan,
            line=line,
            character_db=character_db,
            portrait_bank=portrait_bank,
        )
        line_seed = _speaker_seed(seed, str(line.get("speaker", "")))
        fp = prompt_fingerprint(prompt, width, height, line_seed)
        out_path = images_dir / f"scene_{plan.scene_id:02d}_line_{idx:03d}_{fp}.png"

        if use_cache and out_path.exists() and out_path.stat().st_size > 0:
            results.append(
                {
                    "scene_id": plan.scene_id,
                    "line_index": idx,
                    "speaker": str(line.get("speaker", "")),
                    "start_ms": int(line.get("start_ms", 0)),
                    "end_ms": int(line.get("end_ms", 0)),
                    "duration_ms": int(line.get("duration_ms", 0)),
                    "image_path": str(out_path),
                    "backend": "cache",
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                }
            )
            continue

        ok, backend_info = _generate_with_backend_order(
            prompt=prompt,
            width=width,
            height=height,
            out_path=out_path,
            backend=backend,
            seed=line_seed,
        )

        if not ok:
            raise RuntimeError(
                f"scene {plan.scene_id} line {idx}: image generation failed: {backend_info}"
            )

        results.append(
            {
                "scene_id": plan.scene_id,
                "line_index": idx,
                "speaker": str(line.get("speaker", "")),
                "start_ms": int(line.get("start_ms", 0)),
                "end_ms": int(line.get("end_ms", 0)),
                "duration_ms": int(line.get("duration_ms", 0)),
                "image_path": str(out_path),
                "backend": backend_info,
                "prompt": prompt,
                "width": width,
                "height": height,
            }
        )
    return results


def generate_character_portrait_bank(
    character_db: Dict[str, Any],
    portraits_dir: Path,
    backend: str = "auto",
    quality: str = "balanced",
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate one canonical portrait per character to improve identity consistency.
    """
    width, height = resolve_resolution(quality)
    portraits_dir.mkdir(parents=True, exist_ok=True)
    bank: Dict[str, Dict[str, Any]] = {}

    for character in character_db.get("characters", []):
        name = str(character.get("name", "")).strip().upper()
        if not name:
            continue
        appearance = str(character.get("appearance", "")).strip()
        personality = str(character.get("personality", "")).strip()
        anchor_traits = "; ".join(
            part for part in [appearance, personality] if part
        ).strip()
        prompt = (
            f"Single-person character portrait of {name}. "
            f"Identity anchor token: {name.replace(' ', '_')}. "
            f"Only one person in frame. "
            f"Appearance: {appearance}. Personality expression: {personality}. "
            "Neutral background plate, cinematic lighting, high detail, 35mm style."
        )
        char_seed = _speaker_seed(seed, name)
        fp = prompt_fingerprint(prompt, width, height, char_seed)
        out_path = portraits_dir / f"{name.replace(' ', '_')}_{fp}.png"
        if not out_path.exists() or out_path.stat().st_size == 0:
            ok, info = _generate_with_backend_order(
                prompt=prompt,
                width=width,
                height=height,
                out_path=out_path,
                backend=backend,
                seed=char_seed,
            )
            if not ok:
                raise RuntimeError(f"portrait bank failed for {name}: {info}")
            backend_info = info
        else:
            backend_info = "cache"
        bank[name] = {
            "name": name,
            "portrait_path": str(out_path),
            "anchor_traits": anchor_traits,
            "backend": backend_info,
            "seed": char_seed,
            "width": width,
            "height": height,
        }
    return bank
