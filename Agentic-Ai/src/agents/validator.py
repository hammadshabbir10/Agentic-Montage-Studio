from typing import Dict, List

from src.io.script_ingest import parse_script_to_manifest, validate_script_structure


def run(script_text: str) -> Dict[str, object]:
    from src.io.json_schema import build_scene_manifest
    errors: List[str] = validate_script_structure(script_text)
    manifest = {}
    if not errors:
        manifest_raw = parse_script_to_manifest(script_text, title="Manual Script")
        manifest = build_scene_manifest(manifest_raw.get("scenes", []))
    return {"valid": not errors, "errors": errors, "manifest": manifest}
