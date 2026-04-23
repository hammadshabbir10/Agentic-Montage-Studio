from typing import Dict, List


def run(characters: List[Dict[str, object]], tool_client, image_dir: str) -> List[Dict[str, object]]:
    for character in characters:
        result = tool_client.invoke_by_capability(
            "generate_character_image",
            {
                "name": character.get("name", "character"),
                "description": character.get("appearance", ""),
                "output_dir": image_dir,
            },
        )
        path = result.get("path")
        if path:
            character.setdefault("visual_refs", []).append(path)
    return characters
