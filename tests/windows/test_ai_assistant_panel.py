from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytestqt.qtbot import QtBot

from app.ai.chat_store import ChatStore
from app.models.settings import Settings
from app.windows.ai_assistant_panel import AiAssistantPanel, _CompletionWorker


@pytest.fixture()
def settings(_mock_settings_deps: None) -> Settings:
    model = Settings()
    model.save = MagicMock()  # type: ignore[method-assign]
    return model


@pytest.fixture()
def panel(settings: Settings, qtbot: QtBot) -> Generator[AiAssistantPanel, None, None]:
    metadata_controller = MagicMock()
    metadata_controller.game_version = "1.6"
    with patch("app.windows.ai_assistant_panel.ChatStore") as mock_store:
        mock_store.return_value.as_list.return_value = []
        widget = AiAssistantPanel(
            settings,
            metadata_controller,
            list,
        )
        qtbot.addWidget(widget)
        yield widget


class TestAiAssistantPanelCredentials:
    def test_persist_ai_credentials_saves_api_key_and_proxy(
        self, panel: AiAssistantPanel, settings: Settings
    ) -> None:
        panel.api_key_edit.setText("secret-key")
        panel.proxy_edit.setText("127.0.0.1:8080")

        panel._persist_ai_credentials()

        assert settings.ai_api_key == "secret-key"
        assert settings.ai_proxy == "127.0.0.1:8080"
        assert settings.ai_model == panel._current_model_id()
        settings.save.assert_called_once()  # type: ignore[attr-defined]

    def test_close_event_persists_credentials(
        self, panel: AiAssistantPanel, settings: Settings, qtbot: QtBot
    ) -> None:
        panel.api_key_edit.setText("another-key")
        panel.proxy_edit.setText("socks5://127.0.0.1:1080")

        panel.close()

        assert settings.ai_api_key == "another-key"
        assert settings.ai_proxy == "socks5://127.0.0.1:1080"
        settings.save.assert_called_once()  # type: ignore[attr-defined]

    def test_clear_chat_clears_store_and_history(
        self, panel: AiAssistantPanel, tmp_path: Path
    ) -> None:
        panel._store = ChatStore(path=tmp_path / "ai_chat.json")
        panel._store.append("user", "hello")
        panel._store.append("assistant", "world")
        panel._render_history()

        assert panel._MESSAGE_SEPARATOR in panel.history.toPlainText()

        panel._clear_chat()

        assert panel._store.as_list() == []
        assert panel.history.toPlainText() == ""
        assert not (tmp_path / "ai_chat.json").exists() or panel._store.as_list() == []


class TestAiAssistantPanelModLinks:
    def test_linkify_wraps_known_id_in_link(self, panel: AiAssistantPanel) -> None:
        html_out = panel._linkify(
            "Install mod 2884841315 for loadouts.",
            {
                "2884841315": "https://steamcommunity.com/sharedfiles/filedetails/?id=2884841315"
            },
            [],
        )
        assert (
            '<a href="https://steamcommunity.com/sharedfiles/filedetails/?id=2884841315">'
            "2884841315</a>" in html_out
        )

    def test_linkify_flags_invalid_id(self, panel: AiAssistantPanel) -> None:
        html_out = panel._linkify("Mod 1234567890 is great.", {}, ["1234567890"])
        assert "1234567890" in html_out
        assert "<a href" not in html_out
        assert "color:#c0392b" in html_out

    def test_linkify_leaves_unknown_id_plain(self, panel: AiAssistantPanel) -> None:
        html_out = panel._linkify("Mod 1234567890 is great.", {}, [])
        assert html_out == "Mod 1234567890 is great."

    def test_linkify_escapes_html_special_chars(self, panel: AiAssistantPanel) -> None:
        html_out = panel._linkify("<script>alert(1)</script>", {}, [])
        assert "<script>" not in html_out
        assert "&lt;script&gt;" in html_out

    def test_on_response_persists_mod_links_and_renders_link(
        self, panel: AiAssistantPanel, tmp_path: Path
    ) -> None:
        panel._store = ChatStore(path=tmp_path / "ai_chat.json")
        panel._tool_traces = ['search_workshop_mods("ce loadout") -> 1 matches']

        panel._on_response(
            "Try mod 2884841315 for auto loadouts.",
            {
                "2884841315": "https://steamcommunity.com/sharedfiles/filedetails/?id=2884841315"
            },
            [],
        )

        saved = panel._store.as_list()[-1]
        assert saved["mod_links"]["2884841315"].endswith("id=2884841315")
        rendered = panel.history.toHtml()
        assert (
            'href="https://steamcommunity.com/sharedfiles/filedetails/?id=2884841315"'
            in rendered
        )
        assert "2884841315" in rendered
        assert "Tool: search_workshop_mods" in panel.history.toPlainText()


class TestCompletionWorkerFinalizeModLinks:
    def _worker(self) -> _CompletionWorker:
        return _CompletionWorker(provider=MagicMock(), messages=[])

    def test_known_link_from_tool_call_is_kept(self) -> None:
        worker = self._worker()
        worker._on_tool_call(
            "search_workshop_mods",
            {"query": "x"},
            {
                "matches": [
                    {
                        "publishedfileid": "111",
                        "url": "https://steamcommunity.com/sharedfiles/filedetails/?id=111",
                    }
                ]
            },
        )
        mod_links, invalid_ids = worker._finalize_mod_links("Mod 111 is good.")
        assert mod_links == {
            "111": "https://steamcommunity.com/sharedfiles/filedetails/?id=111"
        }
        assert invalid_ids == []

    def test_unconfirmed_id_gets_validated_and_flagged(self) -> None:
        worker = self._worker()
        with patch(
            "app.windows.ai_assistant_panel.validate_publishedfileids",
            return_value={"valid": [], "invalid": ["222222222"], "valid_details": []},
        ) as mock_validate:
            mod_links, invalid_ids = worker._finalize_mod_links(
                "Try mod 222222222 for combat."
            )
        mock_validate.assert_called_once_with(["222222222"])
        assert mod_links == {}
        assert invalid_ids == ["222222222"]

    def test_unconfirmed_id_that_validates_gets_linked(self) -> None:
        worker = self._worker()
        with patch(
            "app.windows.ai_assistant_panel.validate_publishedfileids",
            return_value={
                "valid": ["333333333"],
                "invalid": [],
                "valid_details": [
                    {
                        "publishedfileid": "333333333",
                        "url": "https://steamcommunity.com/sharedfiles/filedetails/?id=333333333",
                    }
                ],
            },
        ):
            mod_links, invalid_ids = worker._finalize_mod_links("Mod 333333333 works.")
        assert mod_links["333333333"].endswith("id=333333333")
        assert invalid_ids == []

    def test_no_id_shaped_text_skips_validation(self) -> None:
        worker = self._worker()
        with patch(
            "app.windows.ai_assistant_panel.validate_publishedfileids"
        ) as mock_validate:
            mod_links, invalid_ids = worker._finalize_mod_links("No IDs here.")
        mock_validate.assert_not_called()
        assert mod_links == {}
        assert invalid_ids == []
