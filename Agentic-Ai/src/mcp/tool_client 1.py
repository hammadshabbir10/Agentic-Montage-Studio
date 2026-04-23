import asyncio
import base64
import json
import os
import re
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

from groq import Groq
import edge_tts
import requests

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

    def invoke_by_capability(self, capability: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.registry.find_by_capability(capability)
        return self._invoke_tool(tool, payload)

    def _invoke_tool(self, tool: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        tool_type = tool.get("type")
        if tool_type == "groq_llm":
            return self._invoke_groq(tool, payload)
        if tool_type == "image_stub":
            return self._invoke_image_stub(tool, payload)
        if tool_type == "stability_image":
            return self._invoke_stability_image(tool, payload)
        if tool_type == "task_graph_stub":
            return self._invoke_task_graph_stub(payload)
        if tool_type == "voice_stub":
            return self._invoke_voice_stub(payload)
        if tool_type == "elevenlabs_tts":
            return self._invoke_elevenlabs_tts(tool, payload)
        if tool_type == "edge_tts":
            return self._invoke_edge_tts(tool, payload)
        if tool_type == "identity_stub":
            return self._invoke_identity_stub(payload)
        if tool_type == "face_swap_stub":
            return self._invoke_face_swap_stub(payload)
        if tool_type == "lip_sync_stub":
            return self._invoke_lip_sync_stub(payload)
        if tool_type == "moviepy_lip_sync":
            return self._invoke_moviepy_lip_sync(payload)
        if tool_type == "fal_pika":
            return self._invoke_fal_pika(tool, payload)
        if tool_type == "pexels_video":
            return self._invoke_pexels_video(tool, payload)
        if tool_type == "hf_video":
            return self._invoke_hf_video(tool, payload)
        if tool_type == "replicate_video":
            return self._invoke_replicate_video(tool, payload)
        if tool_type == "heygen_face_swap":
            return self._invoke_heygen_face_swap(tool, payload)
        if tool_type == "heygen_lip_sync":
            return self._invoke_heygen_lip_sync(tool, payload)
        if tool_type == "fal_lip_sync":
            return self._invoke_fal_lip_sync(tool, payload)
        if tool_type == "memory_commit":
            return self._invoke_memory_commit(payload)
        if tool_type == "local_stub":
            return {"status": "ok", "detail": "stub", "payload": payload}
        raise ValueError(f"Unsupported tool type: {tool_type}")

    def _invoke_groq(self, tool: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("GROQ_API_KEY")
        prompt = payload.get("prompt", "")
        if not api_key:
            LOGGER.warning("GROQ_API_KEY not set, returning stub response")
            return {
                "text": "INT. ROOM - DAY\nA single scene stub.\nCHARACTER: Placeholder dialogue.\n",
                "model": "stub",
            }

        client = Groq(api_key=api_key)
        model = tool.get("config", {}).get("model", "llama3-70b-8192")
        temperature = tool.get("config", {}).get("temperature", 0.7)
        system_msg = (
            "You are a screenplay writer. Return a multi-scene script with strict formatting:\n"
            "1) Each scene starts with a heading like 'INT. LOCATION - TIME' or 'EXT. LOCATION - TIME'.\n"
            "2) Actions are plain sentences on their own lines.\n"
            "3) Dialogue lines use uppercase speaker names: NAME: dialogue text. DO NOT include parentheticals like (excited) or (to herself) in the dialogue lines.\n"
            "4) Include visual cues using lines like 'VISUAL: [keywords for stock footage search, e.g. woman walking in rain, cinematic lighting]' before every dialogue line.\n"
            "Do not return JSON. Return only the script text."
        )
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        return {"text": text, "model": model}

    def _invoke_image_stub(self, tool: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        name = payload.get("name", "character")
        description = payload.get("description", "")
        # Use output_dir from payload if present, else fallback to self.image_dir
        output_dir = Path(payload.get("output_dir")) if payload.get("output_dir") else self.image_dir
        if not output_dir:
            raise ValueError("output_dir or image_dir is required for image_stub")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_") or "character"
        file_path = output_dir / f"{safe_name}.svg"
        svg = self._build_svg_placeholder(name, description)
        file_path.write_text(svg, encoding="utf-8")
        return {"path": str(file_path)}

    def _build_svg_placeholder(self, name: str, description: str) -> str:
        title = name[:32]
        detail = description[:64]
        return (
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"512\" height=\"512\">"
            "<rect width=\"100%\" height=\"100%\" fill=\"#f0f0f0\"/>"
            f"<text x=\"32\" y=\"64\" font-size=\"24\" fill=\"#333\">{title}</text>"
            f"<text x=\"32\" y=\"110\" font-size=\"14\" fill=\"#666\">{detail}</text>"
            "</svg>"
        )

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
        width = int(config.get("width", 512))
        height = int(config.get("height", 512))
        steps = int(config.get("steps", 30))
        cfg_scale = float(config.get("cfg_scale", 7))

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
                "cfg_scale": cfg_scale,
                "height": height,
                "width": width,
                "samples": 1,
                "steps": steps,
            },
            timeout=120,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Stability API error {response.status_code}: {response.text}"
            )

        data = response.json()
        artifacts = data.get("artifacts", [])
        if not artifacts:
            raise ValueError("Stability API returned no artifacts")
        image_b64 = artifacts[0].get("base64")
        if not image_b64:
            raise ValueError("Stability API artifact missing base64 image")

        image_bytes = base64.b64decode(image_b64)
        file_path.write_bytes(image_bytes)
        return {"path": str(file_path)}

    def _invoke_task_graph_stub(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        manifest = payload.get("manifest", {})
        tasks = []
        for scene in manifest.get("scenes", []):
            tasks.append(
                {
                    "scene_id": scene.get("scene_id", len(tasks) + 1),
                    "location": scene.get("location", ""),
                    "actions": scene.get("actions", []),
                    "dialogue": scene.get("dialogue", []),
                    "characters": scene.get("characters", []),
                }
            )
        return {"tasks": tasks}

    def _invoke_voice_stub(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/audio"))
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"scene_{int(scene_id):02d}.wav"

        sample_rate = 22050
        duration_seconds = 1
        frames = sample_rate * duration_seconds
        with wave.open(str(file_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b"\x00\x00" * frames)

        return {"path": str(file_path)}

    def _invoke_identity_stub(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok", "detail": "identity validated", "payload": payload}

    def _invoke_face_swap_stub(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok", "detail": "faces swapped", "payload": payload}

    def _invoke_lip_sync_stub(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/raw_scenes"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = payload.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        file_path = output_dir / f"scene_{int(scene_id):02d}{suffix}.mp4"
        file_path.write_bytes(b"STUB_MP4")
        return {"path": str(file_path)}

    def _invoke_moviepy_lip_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Real lip-sync: merge audio waveform onto video using moviepy.

        Achieves temporal alignment by:
        1. Loading the source video clip
        2. Loading the synthesised audio track
        3. Setting audio on the video (frame-by-frame alignment)
        4. Adjusting durations so speech and visuals stay in sync
        5. Writing a final composited MP4
        """
        from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

        scene_id = payload.get("scene_id", 0)
        video_path = payload.get("video_path", "")
        audio_path = payload.get("audio_path", "")
        output_dir = Path(payload.get("output_dir", "data/raw_scenes"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = payload.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        file_path = output_dir / f"scene_{int(scene_id):02d}{suffix}.mp4"

        if not video_path or not Path(video_path).exists():
            LOGGER.warning("Lip sync: video not found at %s, writing stub", video_path)
            file_path.write_bytes(b"STUB_MP4")
            return {"path": str(file_path)}

        LOGGER.info("Lip sync start: video=%s audio=%s", video_path, audio_path)

        video_clip = VideoFileClip(video_path)
        # Standardize resolution to 720p for faster 'chain' concatenation later
        if video_clip.w != 1280 or video_clip.h != 720:
            LOGGER.info("Resizing video from %dx%d to 1280x720", video_clip.w, video_clip.h)
            video_clip = video_clip.resized(width=1280, height=720)

        if audio_path and Path(audio_path).exists():
            audio_clip = AudioFileClip(audio_path)

            # ── Temporal alignment: match durations ──
            # If audio is longer than video, loop/extend video to match
            # If video is longer than audio, trim video to audio length
            vid_dur = video_clip.duration
            aud_dur = audio_clip.duration
            LOGGER.info(
                "Lip sync durations: video=%.2fs audio=%.2fs", vid_dur, aud_dur
            )

            if aud_dur > vid_dur and vid_dur > 0:
                # Loop the video to cover the full audio duration
                n_loops = int(aud_dur / vid_dur) + 1
                from moviepy import concatenate_videoclips
                video_clip = concatenate_videoclips([video_clip] * n_loops)
                video_clip = video_clip.subclipped(0, aud_dur)
            elif vid_dur > aud_dur and aud_dur > 0:
                # Trim video to match audio length
                video_clip = video_clip.subclipped(0, aud_dur)

            # ── Subtitle Rendering ──
            dialogue = payload.get("dialogue", [])
            lines_info = []
            
            if dialogue:
                import re
                
                # clean up lines and get char counts
                total_chars = 0
                for d in dialogue:
                    line = d.get("line", "")
                    # strip parentheticals
                    line = re.sub(r"\(.*?\)", "", line).strip()
                    chars = len(line)
                    total_chars += chars
                    lines_info.append({"text": line, "speaker": d.get("speaker", ""), "chars": chars})
                    
                # allocate start and end times proportionally
                current_time = 0.0
                for info in lines_info:
                    if total_chars > 0:
                        duration = aud_dur * (info["chars"] / total_chars)
                    else:
                        duration = aud_dur / len(lines_info)
                        
                    info["start"] = current_time
                    info["end"] = current_time + duration
                    current_time += duration
                    
            def render_subtitles(get_frame, t):
                frame = get_frame(t)
                if not lines_info:
                    return frame
                
                # find active line
                active_line = None
                for info in lines_info:
                    if info["start"] <= t <= info["end"]:
                        active_line = info
                        break
                        
                if active_line and active_line["text"]:
                    import cv2
                    import numpy as np
                    
                    # copy frame to avoid mutating original
                    frame = np.copy(frame)
                    
                    text = f"{active_line['speaker']}: {active_line['text']}"
                    
                    # Add text to bottom using cv2
                    font = cv2.FONT_HERSHEY_COMPLEX
                    font_scale = 1.4
                    thickness = 3
                    color = (255, 255, 255) # White
                    bg_color = (0, 0, 0)
                    
                    h, w = frame.shape[:2]
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Position
                    x = int((w - text_width) / 2)
                    # If text is too wide, we could wrap it, but for simplicity we'll just scale it down
                    if text_width > w - 40:
                        font_scale = font_scale * ((w - 40) / text_width)
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, max(1, int(thickness)))
                        x = int((w - text_width) / 2)

                    y = int(h - 60) # Lift it up a bit more
                    
                    # Draw background rect for readability with some padding
                    pad = 15
                    # We add alpha blending for a semi-transparent background if we want, 
                    # but drawing a solid rect is simpler and highly readable.
                    cv2.rectangle(frame, (x - pad, y - text_height - pad), (x + text_width + pad, y + baseline + pad), bg_color, -1)
                    
                    # Draw text
                    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                    
                return frame

            # Apply transform
            if lines_info:
                video_clip = video_clip.transform(render_subtitles)

            # ── Set the audio track on the video ──
            video_clip = video_clip.with_audio(audio_clip)
        else:
            LOGGER.warning("Lip sync: no audio at %s, outputting video only", audio_path)

        # ── Write the synchronised output ──
        video_clip.write_videofile(
            str(file_path),
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        video_clip.close()
        LOGGER.info("Lip sync complete: %s", file_path)
        return {"path": str(file_path)}

    def _invoke_elevenlabs_tts(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is required for elevenlabs_tts")

        config = tool.get("config", {})
        voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        model_id = config.get("model_id", "eleven_multilingual_v2")
        text = payload.get("text", "")
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/audio"))
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"scene_{int(scene_id):02d}.wav"

        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": model_id, "output_format": "wav"},
            timeout=120,
        )
        if response.status_code != 200:
            raise ValueError(
                f"ElevenLabs error {response.status_code}: {response.text}"
            )
        file_path.write_bytes(response.content)
        return {"path": str(file_path)}

    def _invoke_fal_pika(self, tool: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        import fal_client

        # Get API key – fal_client expects FAL_KEY, but you use FAL_API_KEY
        api_key = os.getenv("FAL_API_KEY") or os.getenv("FAL_KEY")
        if not api_key:
            raise ValueError("FAL_API_KEY or FAL_KEY is required for fal_pika")

        # Set the key for fal_client (global)
        fal_client.api_key = api_key

        config = tool.get("config", {})
        endpoint = config.get("endpoint", "fal-ai/pika/v2.2/text-to-video")
        # Remove leading slash if present (fal_client expects no leading slash)
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]

        # Extract generation parameters from payload and config
        prompt = payload.get("prompt", "")
        resolution = config.get("resolution", "720p")
        aspect_ratio = payload.get("aspect_ratio", config.get("aspect_ratio", "16:9"))
        duration = payload.get("duration", config.get("duration", 5))
        negative_prompt = payload.get("negative_prompt", config.get("negative_prompt", "ugly, bad, terrible"))

        # Optional: seed
        seed = payload.get("seed")

        # Build arguments
        arguments = {
            "prompt": prompt,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "negative_prompt": negative_prompt,
        }
        if seed is not None:
            arguments["seed"] = seed

        # Submit and wait for result (synchronous)
        try:
            result = fal_client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
            )
        except Exception as e:
            raise ValueError(f"fal_client error: {e}")

        # Extract video URL from result
        video_url = None
        if isinstance(result, dict):
            video_url = result.get("video", {}).get("url") or result.get("video_url")
        if not video_url:
            raise ValueError("fal_client response missing video URL")

        # Download video
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/raw_scenes"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = payload.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        file_path = output_dir / f"scene_{int(scene_id):02d}{suffix}.mp4"

        video_resp = requests.get(video_url, timeout=300)
        if video_resp.status_code != 200:
            raise ValueError(f"Video download error {video_resp.status_code}: {video_resp.text}")
        file_path.write_bytes(video_resp.content)

        return {"path": str(file_path)}

    def _invoke_pexels_video(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = os.getenv("PEXELS_API_KEY") or os.getenv("Pexels_API_KEY")
        if not api_key:
            raise ValueError("PEXELS_API_KEY is required for pexels_video")

        config = tool.get("config", {})
        per_page = int(config.get("per_page", 1))
        orientation = config.get("orientation", "landscape")
        size = config.get("size", "medium")

        query = payload.get("prompt", "")
        if not query:
            query = "cinematic scene"

        LOGGER.info("Pexels search start: %s", query)
        response = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": api_key},
            params={
                "query": query,
                "per_page": per_page,
                "orientation": orientation,
                "size": size,
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Pexels error {response.status_code}: {response.text}"
            )

        data = response.json()
        videos = data.get("videos", [])
        if not videos:
            raise ValueError("Pexels returned no videos for query")

        video_files = videos[0].get("video_files", [])
        if not video_files:
            raise ValueError("Pexels video missing video_files")

        def _score(item: Dict[str, Any]) -> int:
            width = item.get("width") or 0
            height = item.get("height") or 0
            return int(width) * int(height)

        best = max(video_files, key=_score)
        video_url = best.get("link")
        if not video_url:
            raise ValueError("Pexels video file missing link")

        LOGGER.info("Pexels download start: %s", video_url)

        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/raw_scenes"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = payload.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        file_path = output_dir / f"scene_{int(scene_id):02d}{suffix}.mp4"

        for attempt in range(1, 4):
            try:
                with requests.get(video_url, stream=True, timeout=300) as video_resp:
                    if video_resp.status_code != 200:
                        raise ValueError(
                            f"Pexels download error {video_resp.status_code}: {video_resp.text}"
                        )
                    with open(file_path, "wb") as handle:
                        for chunk in video_resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                LOGGER.info("Pexels download complete: %s", file_path)
                return {"path": str(file_path)}
            except Exception as exc:
                LOGGER.warning("Pexels download failed (attempt %d): %s", attempt, exc)
                if attempt == 3:
                    raise
                time.sleep(2)

    def _invoke_heygen_face_swap(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = os.getenv("HEYGEN_API_KEY")
        if not api_key:
            raise ValueError("HEYGEN_API_KEY is required for heygen_face_swap")

        config = tool.get("config", {})
        base_url = config.get("base_url", "https://api.heygen.com").rstrip("/")
        video_path = payload.get("video_path", "")
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/raw_scenes"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = payload.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        file_path = output_dir / f"scene_{int(scene_id):02d}{suffix}_faceswapped.mp4"

        characters = payload.get("characters", [])
        image_path = None
        if characters and self.image_dir:
            safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", characters[0]).strip("_")
            potential_path = self.image_dir / f"{safe_name}.png"
            if potential_path.exists():
                image_path = potential_path
            else:
                # Try exact name without safe_name conversion
                potential_path = self.image_dir / f"{characters[0]}.png"
                if potential_path.exists():
                    image_path = potential_path

        files = {}
        with open(video_path, "rb") as handle:
            files["video"] = handle
            if image_path:
                with open(image_path, "rb") as img_handle:
                    files["image"] = img_handle
                    response = requests.post(
                        f"{base_url}/v1/face_swap",
                        headers={"X-Api-Key": api_key},
                        files=files,
                        timeout=300,
                    )
            else:
                response = requests.post(
                    f"{base_url}/v1/face_swap",
                    headers={"X-Api-Key": api_key},
                    files=files,
                    timeout=300,
                )

        if response.status_code not in (200, 201, 202):
            raise ValueError(f"HeyGen error {response.status_code}: {response.text}")
        data = response.json()
        video_url = data.get("video_url") or data.get("url")
        if not video_url:
            return {"path": video_path}

        video_resp = requests.get(video_url, timeout=300)
        if video_resp.status_code >= 400:
            raise ValueError(
                f"HeyGen download error {video_resp.status_code}: {video_resp.text}"
            )
        file_path.write_bytes(video_resp.content)
        return {"path": str(file_path)}

    def _invoke_heygen_lip_sync(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = os.getenv("HEYGEN_API_KEY")
        if not api_key:
            raise ValueError("HEYGEN_API_KEY is required for heygen_lip_sync")

        config = tool.get("config", {})
        base_url = config.get("base_url", "https://api.heygen.com").rstrip("/")
        video_path = payload.get("video_path", "")
        audio_path = payload.get("audio_path", "")
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/raw_scenes"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = payload.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        file_path = output_dir / f"scene_{int(scene_id):02d}{suffix}.mp4"

        with open(video_path, "rb") as video_handle, open(audio_path, "rb") as audio_handle:
            files = {"video": video_handle, "audio": audio_handle}
            response = requests.post(
                f"{base_url}/v1/lip_sync",
                headers={"X-Api-Key": api_key},
                files=files,
                timeout=300,
            )
        if response.status_code not in (200, 201, 202):
            raise ValueError(f"HeyGen error {response.status_code}: {response.text}")
        data = response.json()
        video_url = data.get("video_url") or data.get("url")
        if not video_url:
            return {"path": video_path}

        video_resp = requests.get(video_url, timeout=300)
        if video_resp.status_code >= 400:
            raise ValueError(
                f"HeyGen download error {video_resp.status_code}: {video_resp.text}"
            )
        file_path.write_bytes(video_resp.content)
        return {"path": str(file_path)}

    def _invoke_fal_lip_sync(self, tool: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        import fal_client
        api_key = os.getenv("FAL_API_KEY") or os.getenv("FAL_KEY")
        if not api_key:
            raise ValueError("FAL_API_KEY or FAL_KEY is required for fal_lip_sync")
        
        # fal-client specifically looks for FAL_KEY
        os.environ["FAL_KEY"] = api_key
        fal_client.api_key = api_key

        config = tool.get("config", {})
        endpoint = config.get("endpoint", "fal-ai/sync-lipsync/v2")
        
        video_path = payload.get("video_path")
        audio_path = payload.get("audio_path")
        
        # Upload files if they are local paths
        video_url = fal_client.upload_file(video_path) if os.path.exists(video_path) else video_path
        audio_url = fal_client.upload_file(audio_path) if os.path.exists(audio_path) else audio_path
        
        arguments = {
            "video_url": video_url,
            "audio_url": audio_url,
        }
        
        # Merge other config params
        if "params" in config:
            arguments.update(config["params"])
        
        try:
            result = fal_client.subscribe(endpoint, arguments=arguments, with_logs=True)
        except Exception as e:
            raise ValueError(f"fal_client error: {e}")

        video_url = None
        if isinstance(result, dict):
            video_url = result.get("video", {}).get("url") or result.get("url")
            
        if not video_url:
            raise ValueError(f"fal_client response missing video URL: {result}")

        # Download result
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/raw_scenes"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = payload.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        file_path = output_dir / f"scene_{int(scene_id):02d}{suffix}_fal.mp4"

        resp = requests.get(video_url, timeout=300)
        resp.raise_for_status()
        file_path.write_bytes(resp.content)
        
        return {"path": str(file_path)}

    def _invoke_hf_video(self, tool: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_TOKEN is required for hf_video")

        config = tool.get("config", {})
        model = config.get("model", "cerspense/zeroscope_v2_576w")
        prompt = payload.get("prompt", "")
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/raw_scenes"))
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"scene_{int(scene_id):02d}.mp4"

        base_url = config.get("base_url", "https://router.huggingface.co/hf-inference")
        url = f"{base_url.rstrip('/')}/models/{model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "video/mp4",
        }
        for _ in range(12):
            response = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=300)
            if response.status_code == 200:
                file_path.write_bytes(response.content)
                return {"path": str(file_path)}
            if response.status_code == 503:
                time.sleep(5)
                continue
            if response.status_code == 404 and "api-inference.huggingface.co" not in base_url:
                fallback_url = f"https://api-inference.huggingface.co/models/{model}"
                response = requests.post(
                    fallback_url,
                    headers=headers,
                    json={"inputs": prompt},
                    timeout=300,
                )
                if response.status_code == 200:
                    file_path.write_bytes(response.content)
                    return {"path": str(file_path)}
                if response.status_code == 503:
                    time.sleep(5)
                    continue
                raise ValueError(
                    f"HF error {response.status_code}: {response.text}"
                )
            raise ValueError(
                f"HF error {response.status_code}: {response.text}"
            )

        raise ValueError("HF video generation timed out")

    def _invoke_replicate_video(
        self, tool: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = os.getenv("REPLICATE_API_TOKEN")
        if not api_key:
            raise ValueError("REPLICATE_API_TOKEN is required for replicate_video")

        config = tool.get("config", {})
        model = config.get("model")
        if not model:
            model = payload.get("model")
        version = config.get("version")
        if not model or "REPLICATE_" in model:
            raise ValueError("Replicate model must be set in data/mcp_registry.json")

        if not version or "REPLICATE_" in version:
            info = requests.get(
                f"https://api.replicate.com/v1/models/{model}",
                headers={"Authorization": f"Token {api_key}"},
                timeout=60,
            )
            if info.status_code >= 400:
                raise ValueError(
                    f"Replicate model lookup error {info.status_code}: {info.text}"
                )
            version = info.json().get("latest_version", {}).get("id")
            if not version:
                raise ValueError("Replicate model lookup missing latest_version id")

        prompt = payload.get("prompt", "")
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/raw_scenes"))
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"scene_{int(scene_id):02d}.mp4"

        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }
        submit = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json={
                "version": version,
                "input": {
                    "prompt": prompt,
                },
            },
            timeout=120,
        )
        if submit.status_code not in (200, 201):
            raise ValueError(f"Replicate error {submit.status_code}: {submit.text}")

        data = submit.json()
        status_url = data.get("urls", {}).get("get")
        if not status_url:
            raise ValueError("Replicate response missing status URL")

        for _ in range(90):
            status_resp = requests.get(status_url, headers=headers, timeout=60)
            if status_resp.status_code >= 400:
                raise ValueError(
                    f"Replicate status error {status_resp.status_code}: {status_resp.text}"
                )
            status_data = status_resp.json()
            status = status_data.get("status", "")
            if status == "succeeded":
                output = status_data.get("output")
                video_url = None
                if isinstance(output, list) and output:
                    video_url = output[0]
                elif isinstance(output, str):
                    video_url = output
                if not video_url:
                    raise ValueError("Replicate output missing video URL")
                video_resp = requests.get(video_url, timeout=300)
                if video_resp.status_code >= 400:
                    raise ValueError(
                        f"Replicate download error {video_resp.status_code}: {video_resp.text}"
                    )
                file_path.write_bytes(video_resp.content)
                return {"path": str(file_path)}
            if status == "failed":
                raise ValueError(f"Replicate job failed: {status_data}")
            time.sleep(2)

        raise ValueError("Replicate video generation timed out")

    def _invoke_edge_tts(self, tool: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        config = tool.get("config", {})
        voice = config.get("voice", "en-US-JennyNeural")
        text = payload.get("text", "")
        scene_id = payload.get("scene_id", 0)
        output_dir = Path(payload.get("output_dir", "data/audio"))
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = payload.get("run_tag")
        suffix = f"_{run_tag}" if run_tag else ""
        file_path = output_dir / f"scene_{int(scene_id):02d}{suffix}.wav"

        async def _synthesize() -> None:
            communicator = edge_tts.Communicate(text, voice=voice)
            await communicator.save(str(file_path))

        asyncio.run(_synthesize())
        return {"path": str(file_path)}

    def _invoke_memory_commit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.memory_store:
            return {"status": "skipped", "detail": "memory_store not configured"}
        self.memory_store.add(payload, metadata={"source": "mcp"})
        return {"status": "ok"}
