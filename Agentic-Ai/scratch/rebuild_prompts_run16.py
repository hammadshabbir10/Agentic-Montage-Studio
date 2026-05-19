import json
import os
from pathlib import Path

# Paths
ROOT = Path(r"c:\Users\Crown Tech\Documents\Agenitc Final Proj\Agentic-Montage-Studio\Agentic-Ai")
TIMING_PATH = ROOT / "data" / "phase2_runs" / "run16" / "timing_manifest_run16.json"
IMAGES_DIR = ROOT / "data" / "phase3_runs" / "run16" / "images"
PROMPTS_PATH = ROOT / "data" / "phase3_runs" / "run16" / "image_prompts_run16.json"

def rebuild():
    print("Rebuilding image_prompts_run16.json...")
    
    # 1. Load timing manifest
    with open(TIMING_PATH, "r", encoding="utf-8") as f:
        timing = json.load(f)
    
    # 2. Get all images
    images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")]
    
    prompts_payload = []
    
    # 3. Process each scene and line
    for scene in timing.get("scenes", []):
        sid = scene.get("scene_id")
        for i, line in enumerate(scene.get("lines", []), 1):
            # Find matching image for scene_SS_line_LLL
            pattern = f"scene_{sid:02d}_line_{i:03d}"
            
            # Special case for user's manual image
            if sid == 2 and i == 2:
                img_name = "scene_02_line_002_2112d59996a61.png"
            else:
                # Find best match in images list
                matches = [img for img in images if img.startswith(pattern)]
                if matches:
                    # Sort to get the most "canonical" one if multiple exist
                    img_name = sorted(matches)[0]
                else:
                    # Fallback to any image in this scene if possible
                    scene_matches = [img for img in images if img.startswith(f"scene_{sid:02d}")]
                    if scene_matches:
                        img_name = scene_matches[0]
                    else:
                        img_name = "placeholder.png" # Should not happen based on list_dir
            
            prompts_payload.append({
                "scene_id": sid,
                "kind": "line",
                "line_index": i,
                "speaker": line.get("speaker"),
                "prompt": line.get("visual_cue", ""),
                "backend": "pollinations:flux", # Default
                "image_path": str(Path("data/phase3_runs/run16/images") / img_name),
                "width": 1280,
                "height": 720,
                "start_ms": line.get("start_ms"),
                "end_ms": line.get("end_ms"),
                "duration_ms": line.get("duration_ms")
            })
            print(f"  Scene {sid} Line {i} -> {img_name}")

    # 4. Save
    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts_payload, f, indent=2)
    
    print(f"Successfully wrote {len(prompts_payload)} entries to {PROMPTS_PATH}")

if __name__ == "__main__":
    rebuild()
