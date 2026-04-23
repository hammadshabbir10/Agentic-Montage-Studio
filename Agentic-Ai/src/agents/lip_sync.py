from typing import Dict, List


def run(
    task: Dict[str, object],
    audio_path: str,
    video_path: str,
    tool_client,
    out_dir: str,
    run_tag: str | None = None,
) -> Dict[str, object]:
    scene_id = task.get("scene_id", 0)
    payload = {
        "scene_id": scene_id,
        "audio_path": audio_path,
        "video_path": video_path,
        "output_dir": out_dir,
        "run_tag": run_tag,
        "dialogue": task.get("dialogue", []),
    }
    result = tool_client.invoke_by_capability("lip_sync_aligner", payload)
    return {"scene_id": scene_id, "path": result.get("path", "")}
