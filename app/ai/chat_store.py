import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.utils.app_info import AppInfo
from app.utils.json_utils import atomic_json_dump


class ChatStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (Path(AppInfo().app_storage_folder) / "ai_chat.json")
        self.messages: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.messages = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            raw = data.get("messages", [])
            if isinstance(raw, list):
                self.messages = []
                for m in raw:
                    if not isinstance(m, dict):
                        continue
                    entry: dict[str, Any] = {
                        "role": str(m.get("role", "user")),
                        "content": str(m.get("content", "")),
                    }
                    timestamp = m.get("timestamp")
                    if timestamp:
                        entry["timestamp"] = str(timestamp)
                    tool_trace = m.get("tool_trace")
                    if isinstance(tool_trace, list) and tool_trace:
                        entry["tool_trace"] = [str(t) for t in tool_trace]
                    mod_links = m.get("mod_links")
                    if isinstance(mod_links, dict) and mod_links:
                        entry["mod_links"] = {
                            str(k): str(v) for k, v in mod_links.items()
                        }
                    invalid_ids = m.get("invalid_ids")
                    if isinstance(invalid_ids, list) and invalid_ids:
                        entry["invalid_ids"] = [str(i) for i in invalid_ids]
                    self.messages.append(entry)
        except (json.JSONDecodeError, OSError):
            self.messages = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump({"messages": self.messages}, str(self.path), indent=2)

    def append(
        self,
        role: str,
        content: str,
        *,
        timestamp: str | None = None,
        tool_trace: list[str] | None = None,
        mod_links: dict[str, str] | None = None,
        invalid_ids: list[str] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": timestamp or datetime.now().strftime("%H:%M:%S"),
        }
        if tool_trace:
            entry["tool_trace"] = list(tool_trace)
        if mod_links:
            entry["mod_links"] = dict(mod_links)
        if invalid_ids:
            entry["invalid_ids"] = list(invalid_ids)
        self.messages.append(entry)

    def clear(self) -> None:
        self.messages = []
        self.save()

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.messages)
