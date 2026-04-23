import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ToolRegistry:
    def __init__(self, registry_path: Optional[str] = None) -> None:
        if registry_path:
            self._registry_path = Path(registry_path)
        else:
            root = Path(__file__).resolve().parents[2]
            self._registry_path = root / "data" / "mcp_registry.json"
        self._tools: Optional[List[Dict[str, Any]]] = None

    def _load(self) -> None:
        if self._tools is not None:
            return
        data = json.loads(self._registry_path.read_text(encoding="utf-8"))
        self._tools = data.get("tools", [])

    def list_tools(self) -> List[Dict[str, Any]]:
        self._load()
        return list(self._tools or [])

    def get_tool(self, name: str) -> Dict[str, Any]:
        self._load()
        for tool in self._tools or []:
            if tool.get("name") == name:
                return tool
        raise ValueError(f"Tool not found: {name}")

    def find_by_capability(self, capability: str) -> Dict[str, Any]:
        self._load()
        for tool in self._tools or []:
            if tool.get("capability") == capability:
                return tool
        raise ValueError(f"Tool capability not found: {capability}")
