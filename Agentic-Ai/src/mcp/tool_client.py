"""
tool_client.py  –  Unified MCP Tool Client (Phase 1 + Phase 2)
Handles:
  - Groq LLM          (script generation, story, character design)
  - Image stubs       (SVG placeholders)
  - Stability AI      (real character portraits)
  - Edge TTS          (per-line audio synthesis)
  - ElevenLabs TTS    (cloud TTS alternative)
  - Music commit      (BGM stub writing)
  - Memory commit     (vector store persistence)
Video generation tools have been removed.
"""

import asyncio
import base64
import json
import os
import re
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import edge_tts
import requests
from groq import Groq

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class ToolClient:
    def __init__(
        self,
        registry,
        memory_store: Optional[Any] = None,
        image_dir: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.memory_store = memory_store
        self.image_dir = Path(image_dir) if image_dir else None

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def invoke_by_capability(
        self, capability: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        tool = self.registry.find_by_capability(capability)
        return self._invoke_tool(tool, payload)

    def _invoke_tool(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        tool_type = tool.get("type")

        if tool_type == "groq_llm":
            return self._invoke_groq(tool, payload)
        if tool_type == "image_stub":
            return self._invoke_image_stub(tool, payload)
        if tool_type == "stability_image":
            return self._invoke_stability_image(tool, payload)
        if tool_type == "edge_tts":
            return self._invoke_edge_tts(tool, payload)
        if tool_type == "elevenlabs_tts":
            return self._invoke_elevenlabs_tts(tool, payload)
        if tool_type == "voice_stub":
            return self._invoke_voice_stub(payload)
        if tool_type == "task_graph_stub":
            return self._invoke_task_graph_stub(payload)
        if tool_type == "memory_commit":
            return self._invoke_memory_commit(payload)
        if tool_type == "local_stub":
            return {"status": "ok", "detail": "stub", "payload": payload}

        raise ValueError(f"Unsupported tool type: {tool_type!r}")

    # ── Groq LLM ──────────────────────────────────────────────────────────────

    def _invoke_groq(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = os.getenv("GROQ_API_KEY")
        prompt = payload.get("prompt", "")

        if not api_key:
            LOGGER.warning("GROQ_API_KEY not set – returning stub response")
            return {
                "text": (
                    "INT. ROOM - DAY\n"
                    "A single scene stub.\n"
                    'CHARACTER: "Placeholder dialogue."\n'
                ),
                "model": "stub",
            }

        client = Groq(api_key=api_key)
        config = tool.get("config", {})
        model = config.get("model", "llama3-70b-8192")
        temperature = config.get("temperature", 0.7)

        system_msg = (
            "You are a screenplay writer. Return a multi-scene script with strict formatting:\n"
            "1) Each scene starts with a heading like 'INT. LOCATION - TIME' or 'EXT. LOCATION - TIME'.\n"
            "2) Actions are plain sentences on their own lines.\n"
            "3) Dialogue lines use uppercase speaker names: NAME: \"dialogue text\". "
            "DO NOT include parentheticals like (excited) in dialogue lines.\n"
            "4) Include visual cues using lines like 'VISUAL CUE: Close-up of CHARACTER, "
            "description of shot.'\n"
            "Do not return JSON. Return only the script text."
        )

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        return {"text": text, "model": model}

    # ── Image stub (SVG placeholder) ──────────────────────────────────────────

    def _invoke_image_stub(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        name = payload.get("name", "character")
        description = payload.get("description", "")

        output_dir = (
            Path(payload["output_dir"]) if payload.get("output_dir") else self.image_dir
        )
        if not output_dir:
            raise ValueError("output_dir or image_dir is required for image_stub")

        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_") or "character"
        file_path = output_dir / f"{safe_name}.svg"
        file_path.write_text(
            self._build_svg_placeholder(name, description), encoding="utf-8"
        )
        return {"path": str(file_path)}

    def _build_svg_placeholder(self, name: str, description: str) -> str:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">'
            '<rect width="100%" height="100%" fill="#f0f0f0"/>'
            f'<text x="32" y="64" font-size="24" fill="#333">{name[:32]}</text>'
            f'<text x="32" y="110" font-size="14" fill="#666">{description[:64]}</text>'
            "</svg>"
        )

    # ── Stability AI (real portraits) ─────────────────────────────────────────

    def _invoke_stability_image(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = os.getenv("STABILITY_API_KEY")
        if not api_key:
            raise ValueError("STABILITY_API_KEY is required for stability_image")
        if not self.image_dir:
            raise ValueError("image_dir is required for stability_image")

        name = payload.get("name", "character")
        description = payload.get("description", "")
        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_") or "character"
        file_path = self.image_dir / f"{safe_name}.png"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        config = tool.get("config", {})
        engine_id = config.get("engine_id", "stable-diffusion-v1-6")
        width  = int(config.get("width", 512))
        height = int(config.get("height", 512))
        steps  = int(config.get("steps", 30))
        cfg    = float(config.get("cfg_scale", 7))

        prompt = f"portrait of {name}, {description}, high detail, cinematic lighting"
        url = f"https://api.stability.ai/v1/generation/{engine_id}/text-to-image"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "text_prompts": [{"text": prompt, "weight": 1}],
                "cfg_scale": cfg,
                "height": height,
                "width":  width,
                "samples": 1,
                "steps":  steps,
            },
            timeout=120,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Stability API error {response.status_code}: {response.text}"
            )
        artifacts = response.json().get("artifacts", [])
        if not artifacts:
            raise ValueError("Stability API returned no artifacts")
        image_b64 = artifacts[0].get("base64")
        if not image_b64:
            raise ValueError("Stability API artifact missing base64 image")
        file_path.write_bytes(base64.b64decode(image_b64))
        return {"path": str(file_path)}

    # ── Edge TTS (single-line synthesis) ─────────────────────────────────────

    def _invoke_edge_tts(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        config = tool.get("config", {})
        voice = config.get("voice", "en-US-JennyNeural")
        text = payload.get("text", "")
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/audio"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = payload.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        file_path = output_dir / f"scene_{int(scene_id):02d}{suffix}.mp3"

        async def _synth() -> None:
            comm = edge_tts.Communicate(text, voice=voice)
            await comm.save(str(file_path))

        asyncio.run(_synth())
        return {"path": str(file_path)}

    # ── ElevenLabs TTS ────────────────────────────────────────────────────────

    def _invoke_elevenlabs_tts(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for elevenlabs_tts")

        config   = tool.get("config", {})
        voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        model_id = config.get("model_id", "eleven_multilingual_v2")
        text     = payload.get("text", "")
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/audio"))
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"scene_{int(scene_id):02d}.mp3"

        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": model_id, "output_format": "mp3_44100_128"},
            timeout=120,
        )
        if response.status_code != 200:
            raise ValueError(
                f"ElevenLabs error {response.status_code}: {response.text}"
            )
        file_path.write_bytes(response.content)
        return {"path": str(file_path)}

    # ── Voice stub (silence WAV) ──────────────────────────────────────────────

    def _invoke_voice_stub(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scene_id   = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/audio"))
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"scene_{int(scene_id):02d}.wav"
        sample_rate = 22050
        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"\x00\x00" * sample_rate)
        return {"path": str(file_path)}

    # ── Task graph stub ───────────────────────────────────────────────────────

    def _invoke_task_graph_stub(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        manifest = payload.get("manifest", {})
        tasks = [
            {
                "scene_id":   scene.get("scene_id", i + 1),
                "location":   scene.get("location", ""),
                "actions":    scene.get("actions", []),
                "dialogue":   scene.get("dialogue", []),
                "characters": scene.get("characters", []),
            }
            for i, scene in enumerate(manifest.get("scenes", []))
        ]
        return {"tasks": tasks}

    # ── Memory commit ─────────────────────────────────────────────────────────

    def _invoke_memory_commit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.memory_store:
            return {"status": "skipped", "detail": "memory_store not configured"}
        self.memory_store.add(payload, metadata={"source": "mcp"})
        return {"status": "ok"}