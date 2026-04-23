from typing import Dict


def run(
    task: Dict[str, object],
    tool_client,
    frames_dir: str,
    run_tag: str | None = None,
) -> Dict[str, object]:
    scene_id = task.get("scene_id", 0)
    prompt = task.get("prompt", "")
    if not prompt:
        # Pexels struggles with long action sentences. Use visual cues which are keyword-rich.
        dialogues = task.get("dialogue", [])
        visual_cues = [d.get("visual_cue") for d in dialogues if d.get("visual_cue")]
        location = task.get("location", "")
        
        if visual_cues:
            prompt = f"{location} {visual_cues[0]}"
        else:
            prompt = location

    result = tool_client.invoke_by_capability(
        "generate_scene_video",
        {
            "scene_id": scene_id,
            "prompt": prompt,
            "output_dir": frames_dir,  # Use the per-run directory
            "run_tag": run_tag,
        },
    )
    return {
        "scene_id": scene_id,
        "video_path": result.get("path", ""),
        "frames": [],
    }
