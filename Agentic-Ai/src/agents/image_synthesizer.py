from typing import Dict, List


def run(characters: List[Dict[str, object]], tool_client, image_dir: str) -> List[Dict[str, object]]:
    for character in characters:
        appearance_raw = character.get("appearance", "")
        if isinstance(appearance_raw, dict):
            description = appearance_raw.get("description", "")
        else:
            description = str(appearance_raw)
        result = tool_client.invoke_by_capability(
            "generate_character_image",
            {
                "name": character.get("name", "character"),
                "description": description,
                "output_dir": image_dir,
            },
        )
        path = result.get("path")
        if path:
            character.setdefault("visual_refs", []).append(path)
    return characters
