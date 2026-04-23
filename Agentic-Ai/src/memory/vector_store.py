import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class MemoryStore:
    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._use_chroma = False
        self._collection = None
        self._jsonl_path = self.persist_dir / "memory.jsonl"
        self._init_store()

    def _init_store(self) -> None:
        if os.getenv("CHROMA_DISABLE") == "1":
            LOGGER.info("ChromaDB disabled via CHROMA_DISABLE=1")
            self._use_chroma = False
            return
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = client.get_or_create_collection("writer_room")
            self._use_chroma = True
            LOGGER.info("Using ChromaDB for memory store")
        except Exception as exc:
            LOGGER.warning("ChromaDB unavailable, using JSONL store: %s", exc)
            self._use_chroma = False

    def add(self, item: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())
        text = json.dumps(item, ensure_ascii=True)
        if self._use_chroma and self._collection:
            try:
                self._collection.add(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[metadata or {}],
                )
                return doc_id
            except Exception as exc:
                LOGGER.warning("ChromaDB add failed, falling back to JSONL: %s", exc)
                self._use_chroma = False

        record = {"id": doc_id, "text": text, "metadata": metadata or {}}
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return doc_id

    def query(self, text: str, k: int = 3) -> List[Dict[str, Any]]:
        if self._use_chroma and self._collection:
            try:
                result = self._collection.query(query_texts=[text], n_results=k)
                return result.get("documents", [[]])[0]
            except Exception as exc:
                LOGGER.warning("ChromaDB query failed, returning empty: %s", exc)
                self._use_chroma = False
        return []
