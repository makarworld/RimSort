from pathlib import Path

from app.ai.chat_store import ChatStore


class TestChatStore:
    def test_roundtrip(self, tmp_path: Path) -> None:
        store = ChatStore(tmp_path / "chat.json")
        store.append("user", "hello")
        store.append("assistant", "hi")
        store.save()

        reloaded = ChatStore(tmp_path / "chat.json")
        assert len(reloaded.as_list()) == 2
        assert reloaded.as_list()[0]["content"] == "hello"
        assert "timestamp" in reloaded.as_list()[0]

    def test_roundtrip_preserves_tool_trace_and_mod_links(self, tmp_path: Path) -> None:
        store = ChatStore(tmp_path / "chat.json")
        store.append(
            "assistant",
            "Try mod 123.",
            tool_trace=['search_workshop_mods("x") -> 1 matches'],
            mod_links={
                "123": "https://steamcommunity.com/sharedfiles/filedetails/?id=123"
            },
            invalid_ids=["999999999"],
        )
        store.save()

        reloaded = ChatStore(tmp_path / "chat.json")
        msg = reloaded.as_list()[0]
        assert msg["tool_trace"] == ['search_workshop_mods("x") -> 1 matches']
        assert msg["mod_links"] == {
            "123": "https://steamcommunity.com/sharedfiles/filedetails/?id=123"
        }
        assert msg["invalid_ids"] == ["999999999"]

    def test_append_without_tool_trace_omits_optional_fields(
        self, tmp_path: Path
    ) -> None:
        store = ChatStore(tmp_path / "chat.json")
        store.append("user", "hello")
        assert "tool_trace" not in store.as_list()[0]
        assert "mod_links" not in store.as_list()[0]
        assert "invalid_ids" not in store.as_list()[0]
