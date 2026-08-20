import html
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.ai.chat_store import ChatStore
from app.ai.gemini_models import (
    GEMINI_MODELS,
    is_quota_error_message,
    normalize_model_id,
    suggest_models_excluding,
)
from app.ai.gemini_provider import GeminiProvider
from app.ai.tools import mod_context
from app.ai.tools.mod_tools import (
    GEMINI_MOD_TOOL_DECLARATIONS,
    ModToolExecutor,
    extract_mod_links,
    summarize_tool_result,
)
from app.controllers.metadata_controller import MetadataController
from app.models.settings import Settings
from app.utils.app_info import AppInfo
from app.utils.steam.workshop_validate import validate_publishedfileids

_WORKSHOP_ID_PATTERN = re.compile(r"\b\d{6,10}\b")


class _CompletionWorker(QThread):
    finished_ok = Signal(str, dict, list)
    finished_error = Signal(str)
    tool_trace = Signal(str)
    progress = Signal(int, int, str)

    def __init__(
        self,
        provider: GeminiProvider,
        messages: list[dict[str, str]],
        steam_apikey_override: str | None = None,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._messages = messages
        self._steam_apikey_override = steam_apikey_override
        self._known_links: dict[str, str] = {}

    def _on_tool_call(
        self, name: str, args: dict[str, Any], result: dict[str, Any]
    ) -> None:
        self.tool_trace.emit(_format_tool_trace(name, args, result))
        self._known_links.update(extract_mod_links(name, result))

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress.emit(current, total, message)

    def _finalize_mod_links(self, text: str) -> tuple[dict[str, str], list[str]]:
        """Linkify only IDs backed by a tool result; validate any other IDs the
        model mentioned in prose so hallucinated ones can be flagged, not linked."""
        mod_links = dict(self._known_links)
        candidates = {
            m for m in _WORKSHOP_ID_PATTERN.findall(text) if m not in mod_links
        }
        if not candidates:
            return mod_links, []
        validation = validate_publishedfileids(sorted(candidates))
        for detail in validation.get("valid_details", []) or []:
            pfid = str(detail.get("publishedfileid", "")).strip()
            url = str(detail.get("url", "")).strip()
            if pfid and url:
                mod_links[pfid] = url
        invalid_ids = [str(i) for i in validation.get("invalid", []) or []]
        return mod_links, invalid_ids

    def run(self) -> None:
        tool_executor = ModToolExecutor(
            steam_apikey_override=self._steam_apikey_override,
            on_progress=self._on_progress,
        )
        try:
            text = self._provider.complete(
                self._messages,
                tools=GEMINI_MOD_TOOL_DECLARATIONS,
                tool_executor=tool_executor,
                on_tool_call=self._on_tool_call,
            )
            mod_links, invalid_ids = self._finalize_mod_links(text)
            self.finished_ok.emit(text, mod_links, invalid_ids)
        except Exception as exc:  # noqa: BLE001
            self.finished_error.emit(str(exc))


def _brief_tool_args(name: str, args: dict[str, Any]) -> str:
    if args.get("query"):
        return f'"{args["query"]}"'
    if name in ("queue_download", "validate_workshop_ids"):
        pfids = args.get("publishedfileids") or []
        if isinstance(pfids, list):
            return f"{len(pfids)} ids"
    if name == "find_russian_localizations_for_active_mods":
        raw = args.get("package_ids")
        if isinstance(raw, list) and raw:
            return f"{len(raw)} mods"
        return "all active"
    if args.get("package_id"):
        return str(args["package_id"])
    if args.get("source"):
        return str(args["source"])
    return ""


def _format_tool_trace(name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    brief = _brief_tool_args(name, args)
    args_part = f"({brief})" if brief else ""
    summary = summarize_tool_result(name, result)
    return f"{name}{args_part} -> {summary}"


class AiAssistantPanel(QDialog):
    _LOADING_INTERVAL_MS = 400
    _MESSAGE_SEPARATOR = "==========================="
    _LOCALIZATION_PROGRESS_PREFIXES = (
        "Scanning active mods for existing localizations",
        "Searching Steam Workshop for Russian localizations",
        "Checking:",
        "Workshop search:",
    )

    def __init__(
        self,
        settings: Settings,
        metadata_controller: MetadataController,
        get_active_paths: Callable[[], list[str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.metadata_controller = metadata_controller
        self._get_active_paths = get_active_paths
        self._store = ChatStore()
        self._worker: _CompletionWorker | None = None
        self._tool_traces: list[str] = []
        self._progress_current = 0
        self._progress_total = 0
        self._progress_message = ""
        self._loading_dot_count = 0
        self._loading_timer = QTimer(self)
        self._loading_timer.timeout.connect(self._tick_loading)
        self.setWindowTitle(self.tr("AI Assistant"))
        self.setMinimumSize(520, 420)
        self._build_ui()
        self._render_history()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel(self.tr("Provider:")))
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Gemini", "gemini")
        top.addWidget(self.provider_combo)
        top.addWidget(QLabel(self.tr("Model:")))
        self.model_combo = QComboBox()
        for spec in GEMINI_MODELS:
            limit_bits: list[str] = []
            if spec.rpm is not None:
                limit_bits.append(f"{spec.rpm} RPM")
            if spec.rpd is not None:
                limit_bits.append(f"{spec.rpd} RPD")
            limits = f" ({', '.join(limit_bits)})" if limit_bits else ""
            self.model_combo.addItem(f"{spec.label}{limits}", spec.model_id)
        saved_model = normalize_model_id(self.settings.ai_model)
        model_index = self.model_combo.findData(saved_model)
        if model_index >= 0:
            self.model_combo.setCurrentIndex(model_index)
        top.addWidget(self.model_combo)
        top.addWidget(QLabel(self.tr("API key:")))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setText(self.settings.ai_api_key)
        top.addWidget(self.api_key_edit)
        layout.addLayout(top)

        proxy_row = QHBoxLayout()
        proxy_row.addWidget(QLabel(self.tr("Proxy:")))
        self.proxy_edit = QLineEdit()
        self.proxy_edit.setPlaceholderText(
            self.tr("host:port, user:pass@host:port, login:pass:host:port")
        )
        self.proxy_edit.setText(self.settings.ai_proxy)
        proxy_row.addWidget(self.proxy_edit)
        layout.addLayout(proxy_row)

        self.history = QTextBrowser()
        self.history.setReadOnly(True)
        self.history.setOpenExternalLinks(True)
        layout.addWidget(self.history)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        input_row = QHBoxLayout()
        self.clear_btn = QPushButton(self.tr("Clear chat"))
        self.clear_btn.clicked.connect(self._clear_chat)
        input_row.addWidget(self.clear_btn)
        self.input_edit = QLineEdit()
        self.input_edit.returnPressed.connect(self._send)
        input_row.addWidget(self.input_edit)
        self.send_btn = QPushButton(self.tr("Send"))
        self.send_btn.clicked.connect(self._send)
        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

    def _role_label(self, role: str) -> str:
        if role == "assistant":
            return self.tr("Assistant")
        return self.tr("User")

    def _linkify(
        self, text: str, mod_links: dict[str, str], invalid_ids: list[str]
    ) -> str:
        invalid_set = set(invalid_ids)
        out: list[str] = []
        last = 0
        for m in _WORKSHOP_ID_PATTERN.finditer(text):
            out.append(html.escape(text[last : m.start()]))
            token = m.group(0)
            if token in mod_links:
                url = html.escape(mod_links[token], quote=True)
                out.append(f'<a href="{url}">{html.escape(token)}</a>')
            elif token in invalid_set:
                title = html.escape(
                    self.tr("Unverified/non-existent Workshop ID"), quote=True
                )
                out.append(
                    f'<span style="color:#c0392b" title="{title}">'
                    f"{html.escape(token)} ⚠</span>"
                )
            else:
                out.append(html.escape(token))
            last = m.end()
        out.append(html.escape(text[last:]))
        return "".join(out)

    def _format_message_line(self, msg: dict[str, Any]) -> str:
        timestamp = msg.get("timestamp", "")
        prefix = f"[{timestamp}] " if timestamp else ""
        header = html.escape(f"{prefix}{self._role_label(msg['role'])}: ")
        content = self._linkify(
            str(msg["content"]),
            msg.get("mod_links") or {},
            msg.get("invalid_ids") or [],
        )
        return f"{header}{content}"

    def _render_history(
        self, *, loading_text: str | None = None, error_text: str | None = None
    ) -> None:
        parts: list[str] = []
        for msg in self._store.as_list():
            if msg.get("role") == "assistant":
                for trace in msg.get("tool_trace") or []:
                    parts.append(
                        html.escape(self.tr("Tool: {call}").format(call=trace))
                    )
            parts.append(self._format_message_line(msg))
            parts.append(html.escape(self._MESSAGE_SEPARATOR))
        if loading_text:
            for trace in self._tool_traces:
                parts.append(html.escape(self.tr("Tool: {call}").format(call=trace)))
            parts.append(html.escape(loading_text))
        if error_text:
            parts.append(html.escape(error_text))
            parts.append(html.escape(self._MESSAGE_SEPARATOR))
        body = "\n\n".join(parts)
        self.history.setHtml(
            f'<pre style="white-space:pre-wrap; font-family:inherit; margin:0;">'
            f"{body}</pre>"
        )
        scrollbar = self.history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _loading_text(self) -> str:
        if self._progress_message:
            if self._progress_total > 0:
                return self.tr("{current}/{total}: {message}").format(
                    current=self._progress_current,
                    total=self._progress_total,
                    message=self._progress_message,
                )
            return self._progress_message
        dots = "." * self._loading_dot_count
        return self.tr("Loading{dots}").format(dots=dots)

    def _start_loading(self) -> None:
        self._loading_dot_count = 1
        self._progress_current = 0
        self._progress_total = 0
        self._progress_message = ""
        self.progress_bar.setVisible(False)
        self._render_history(loading_text=self._loading_text())
        self._loading_timer.start(self._LOADING_INTERVAL_MS)

    def _stop_loading(self) -> None:
        self._loading_timer.stop()
        self.progress_bar.setVisible(False)
        self._progress_current = 0
        self._progress_total = 0
        self._progress_message = ""

    def _tick_loading(self) -> None:
        if self._progress_message:
            return
        self._loading_dot_count = (self._loading_dot_count % 3) + 1
        self._render_history(loading_text=self._loading_text())

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self._progress_current = current
        self._progress_total = total
        self._progress_message = message
        show_bar = message.startswith(self._LOCALIZATION_PROGRESS_PREFIXES)
        self.progress_bar.setVisible(show_bar)
        if show_bar:
            if total > 0:
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(min(current, total))
                self.progress_bar.setFormat(
                    self.tr("{current}/{total}: {message}").format(
                        current=current,
                        total=total,
                        message=message,
                    )
                )
            else:
                self.progress_bar.setRange(0, 0)
                self.progress_bar.setFormat(message)
        self._render_history(loading_text=self._loading_text())

    def _clear_chat(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._store.clear()
        self._tool_traces.clear()
        self._render_history()

    def _current_model_id(self) -> str:
        return normalize_model_id(self.model_combo.currentData())

    def _persist_ai_credentials(self) -> None:
        self.settings.ai_api_key = self.api_key_edit.text().strip()
        self.settings.ai_proxy = self.proxy_edit.text().strip()
        self.settings.ai_model = self._current_model_id()
        self.settings.save()

    def _select_model(self, model_id: str) -> None:
        index = self.model_combo.findData(model_id)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.settings.ai_model = normalize_model_id(model_id)
        self.settings.save()

    def _steam_apikey_context_line(self) -> str:
        key = self.settings.steam_apikey.strip()
        if key:
            return f"Steam Web API key: configured ({len(key)} chars)."
        return "Steam Web API key: not configured."

    def _system_context(self) -> str:
        version = AppInfo().app_version
        game_version = self.metadata_controller.game_version
        active_paths = self._get_active_paths()
        active_mods = mod_context.list_active_mods(
            self.metadata_controller, active_paths
        )
        mods_preview = mod_context.format_active_mods_context(
            self.metadata_controller, active_paths, limit=200
        )
        steam_line = self._steam_apikey_context_line()
        return (
            "CRITICAL: NEVER invent Steam Workshop publishedfileid values. "
            "Only cite IDs returned by search_workshop_mods, "
            "find_russian_localizations_for_active_mods, or validate_workshop_ids. "
            "If you need to mention an ID that did not come from one of those tools "
            "in this turn, call validate_workshop_ids on it first and report only "
            "valid IDs. "
            f"You are a helpful assistant for RimSort {version}. "
            f"RimWorld version: {game_version}. Active mods: {len(active_mods)}. "
            f"{steam_line} "
            "Answer questions about mod lists. Read-only changes are not applied directly; "
            "use queue_sort_mods, queue_save_mods, queue_download, or queue_run_game only "
            "when the user explicitly asks and RimSort GUI is running. "
            "Do not use Markdown. Reply in plain text only (no **, ##, ```, or bullet lists). "
            "For Russian localization requests, ALWAYS call "
            "find_russian_localizations_for_active_mods first. "
            "Before showing a final localization list, verify IDs with "
            "validate_workshop_ids and report only valid IDs. "
            "Before queue_download, use only IDs from tool results; "
            "queue_download validates IDs automatically. "
            "NEVER claim steam_apikey is missing unless get_instance_summary or "
            "a workshop search tool returned that error. "
            f"When calling search_workshop_mods or search_steam_workshop, ALWAYS pass "
            f"version='{game_version}' unless the user explicitly asks for a different "
            "RimWorld version, so results are not for the wrong game version. "
            "Tools: describe_mod, search_installed_mods, search_workshop_mods, "
            "search_steam_workshop (same as search_workshop_mods), "
            "find_russian_localizations_for_active_mods, validate_workshop_ids, "
            "check_mod_conflicts, list_missing_deps, get_instance_summary, "
            "read_log (source player or rimsort), list_active_mods, list_installed_mods. "
            "check_mod_conflicts only works for a mod that is already INSTALLED "
            "(RimSort can only read declared incompatibilities from a downloaded mod's "
            "About.xml). For a Workshop mod that is NOT yet installed, RimSort has no "
            "way to verify compatibility with active mods before download — never claim "
            "or imply a not-yet-installed mod is compatible with anything (e.g. Combat "
            "Extended); say plainly that compatibility is unverified and tell the user "
            "to check the mod's own Workshop page/comments before installing it.\n\n"
            f"{mods_preview}"
        )

    def _on_tool_trace(self, line: str) -> None:
        self._tool_traces.append(line)
        self._render_history(loading_text=self._loading_text())

    def _send(self) -> None:
        text = self.input_edit.text().strip()
        if not text or (self._worker and self._worker.isRunning()):
            return
        self._persist_ai_credentials()
        self.input_edit.clear()
        self._tool_traces.clear()
        self._store.append("user", text)
        self._start_loading()

        messages = [{"role": "system", "content": self._system_context()}]
        messages.extend(
            {"role": str(m["role"]), "content": str(m["content"])}
            for m in self._store.as_list()
        )
        provider = GeminiProvider(
            self.settings.ai_api_key,
            proxy=self.settings.ai_proxy,
            model=self._current_model_id(),
        )
        self.send_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self._worker = _CompletionWorker(
            provider,
            messages,
            steam_apikey_override=self.settings.steam_apikey.strip() or None,
        )
        self._worker.tool_trace.connect(self._on_tool_trace)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_response)
        self._worker.finished_error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self) -> None:
        self._stop_loading()
        self.send_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

    def _on_response(
        self, text: str, mod_links: dict[str, str], invalid_ids: list[str]
    ) -> None:
        self._store.append(
            "assistant",
            text,
            tool_trace=self._tool_traces,
            mod_links=mod_links,
            invalid_ids=invalid_ids,
        )
        self._store.save()
        self._render_history()

    def _on_error(self, message: str) -> None:
        self._stop_loading()
        timestamp = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
        error_block = f"[{timestamp}] {self.tr('Error')}: {message}"
        self._render_history(error_text=error_block)

        if is_quota_error_message(message):
            suggestions = suggest_models_excluding(self._current_model_id())
            if not suggestions:
                return
            recommended = suggestions[0]
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Icon.Warning)
            dialog.setWindowTitle(self.tr("Model quota exceeded"))
            dialog.setText(message)
            switch_label = self.tr("Switch to {name}").format(name=recommended.label)
            switch_btn = dialog.addButton(
                switch_label, QMessageBox.ButtonRole.AcceptRole
            )
            dialog.addButton(QMessageBox.StandardButton.Close)
            dialog.exec()
            if dialog.clickedButton() == switch_btn:
                self._select_model(recommended.model_id)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._persist_ai_credentials()
        super().closeEvent(event)

    def show_non_modal(self) -> None:
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.show()
