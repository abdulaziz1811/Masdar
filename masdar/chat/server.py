"""A local web server for the chat page.

Standard library only, so `masdar serve` needs nothing beyond what the agent
already needs. It binds to 127.0.0.1 by default and has no login: it is a
tool on the user's own machine, not a public service.

Three rules shape it:

* **Only files this server wrote are downloadable.** An Excel file is served
  by an opaque token handed out when it was produced, never by a path taken
  from the request, so no request can reach any other file on disk.
* **Source text is data.** Titles, notes and publisher names come from
  external sites; the page inserts them as text, never as markup, and only
  http(s) links are rendered as links.
* **One question at a time through the agent.** Its HTTP client and adapters
  keep per-run state and are not built for concurrent use; a lock serialises
  answers while the page itself stays responsive.
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse

from masdar.chat.session import Sessions
from masdar.domain.models import Answer, Finding, Verdict
from masdar.pipeline.orchestrator import Agent

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 16_384
MAX_MESSAGE_CHARS = 500
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

VERDICT_LABELS = {
    Verdict.AVAILABLE: "متوفرة",
    Verdict.PARTIAL: "متوفرة جزئياً",
    Verdict.NOT_AVAILABLE: "غير متوفرة للفترة المطلوبة",
    Verdict.UNVERIFIED: "غير مؤكدة",
    Verdict.NO_SOURCE: "لا توجد نتيجة مطابقة",
    Verdict.SOURCE_UNREACHABLE: "تعذّر الوصول للمصادر",
}


class Downloads:
    """Tokens for the files this server produced, and nothing else."""

    def __init__(self, capacity: int = 1000):
        self._paths: dict[str, Path] = {}
        self._capacity = capacity
        self._lock = threading.Lock()

    def register(self, path: str | Path) -> str:
        token = secrets.token_urlsafe(18)
        with self._lock:
            if len(self._paths) >= self._capacity:
                self._paths.pop(next(iter(self._paths)))
            self._paths[token] = Path(path).resolve()
        return token

    def resolve(self, token: str) -> Path | None:
        with self._lock:
            return self._paths.get(token)


def _http_link(url: str | None) -> str | None:
    """Keep a link only if it is plainly http(s)."""
    if not url:
        return None
    return url if urlparse(url).scheme in ("http", "https") else None


def _finding_payload(finding: Finding, downloads: Downloads) -> dict:
    provenance = finding.provenance
    coverage = finding.coverage
    return {
        "title": finding.candidate.title_ar or finding.candidate.title_en,
        "source_id": finding.candidate.source_id,
        "publisher": provenance.publisher_ar or provenance.publisher_en,
        "verdict": finding.verdict.value,
        "verdict_label": VERDICT_LABELS.get(finding.verdict, finding.verdict.value),
        "last_updated": provenance.last_updated.isoformat() if provenance.last_updated else None,
        "last_updated_note": provenance.last_updated_note,
        "latest_year": coverage.latest(),
        "coverage": coverage.describe() if coverage.years else None,
        "evidence": coverage.origin.value,
        "matched_years": list(finding.matched_years),
        "missing_years": list(finding.missing_years),
        "landing_url": _http_link(provenance.landing_url),
        "resource_url": _http_link(provenance.resource_url),
        "retrieved_at": provenance.retrieved_at.isoformat(),
        "sha256": provenance.sha256,
        "download": (
            f"/files/{downloads.register(finding.export_path)}" if finding.export_path else None
        ),
        "notes": list(finding.notes),
    }


def answer_payload(answer: Answer, downloads: Downloads, follow_up: bool) -> dict:
    request = answer.request
    return {
        "verdict": answer.verdict.value,
        "verdict_label": VERDICT_LABELS.get(answer.verdict, answer.verdict.value),
        "headline": answer.message_ar.split("\n", 1)[0],
        "message": answer.message_ar,
        "understood": {
            "topic": request.topic.label_ar if request.topic else None,
            "period": request.period.label(),
            "dimensions": [d.label_ar for d in request.dimensions],
            "follow_up": follow_up,
        },
        "findings": [_finding_payload(f, downloads) for f in answer.findings[:4]],
        "suggestions": [
            {
                "year": s.year,
                "reason": s.reason_ar,
                "source": s.source_ar,
                "title": s.title_ar,
                # Sent back as the next message: the session reads a bare
                # year as "the same question, for this year".
                "ask": str(s.year),
            }
            for s in answer.suggestions
        ],
        "source_errors": [{"source": s, "reason": r} for s, r in answer.source_errors],
    }


@dataclass
class ChatApp:
    agent: Agent
    sessions: Sessions
    downloads: Downloads
    answer_lock: threading.Lock

    @classmethod
    def build(cls, agent: Agent) -> ChatApp:
        return cls(
            agent=agent,
            sessions=Sessions(agent.lexicon),
            downloads=Downloads(),
            answer_lock=threading.Lock(),
        )

    def ask(self, session_id: str | None, message: str) -> dict:
        session_id, conversation = self.sessions.get(session_id)
        with conversation.lock:
            turn = conversation.understand(message)
            with self.answer_lock:
                answer = self.agent.answer_request(turn.request)
        payload = answer_payload(answer, self.downloads, turn.follow_up)
        payload["session"] = session_id
        return payload

    def reset(self, session_id: str | None) -> dict:
        session_id, conversation = self.sessions.get(session_id)
        with conversation.lock:
            conversation.reset()
        return {"session": session_id}


def make_handler(app: ChatApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Masdar"
        sys_version = ""

        # -- responses -------------------------------------------------
        def _send(self, status: int, body: bytes, content_type: str, extra=None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _error(self, status: HTTPStatus, message_ar: str) -> None:
            self._json(status, {"error": message_ar})

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            pass  # questions are not written to the terminal log

        # -- routes ----------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                page = (STATIC_DIR / "index.html").read_bytes()
                self._send(
                    HTTPStatus.OK, page, "text/html; charset=utf-8",
                    {
                        "Content-Security-Policy": (
                            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                            "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
                        ),
                        "Referrer-Policy": "no-referrer",
                    },
                )
            elif path == "/api/health":
                self._json(HTTPStatus.OK, {"ok": True})
            elif path.startswith("/files/"):
                self._download(path.removeprefix("/files/"))
            else:
                self._error(HTTPStatus.NOT_FOUND, "غير موجود")

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            path = urlparse(self.path).path
            data = self._read_json()
            if data is None:
                return
            session_id = data.get("session") if isinstance(data.get("session"), str) else None
            if path == "/api/ask":
                message = data.get("message")
                if not isinstance(message, str) or not message.strip():
                    self._error(HTTPStatus.BAD_REQUEST, "اكتب سؤالاً.")
                    return
                if len(message) > MAX_MESSAGE_CHARS:
                    self._error(HTTPStatus.BAD_REQUEST, "السؤال أطول من اللازم.")
                    return
                try:
                    payload = app.ask(session_id, message.strip())
                except Exception as exc:  # the page must get an answer, not a hang
                    self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"خطأ غير متوقع: {exc}")
                    return
                self._json(HTTPStatus.OK, payload)
            elif path == "/api/reset":
                self._json(HTTPStatus.OK, app.reset(session_id))
            else:
                self._error(HTTPStatus.NOT_FOUND, "غير موجود")

        # -- helpers ---------------------------------------------------
        def _read_json(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "الطلب أكبر من المسموح.")
                return None
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._error(HTTPStatus.BAD_REQUEST, "صيغة الطلب غير صحيحة.")
                return None
            if not isinstance(data, dict):
                self._error(HTTPStatus.BAD_REQUEST, "صيغة الطلب غير صحيحة.")
                return None
            return data

        def _download(self, token: str) -> None:
            path = app.downloads.resolve(token)
            if path is None or not path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "الملف غير موجود أو انتهت صلاحية رابطه.")
                return
            disposition = f"attachment; filename*=UTF-8''{quote(path.name)}"
            self._send(
                HTTPStatus.OK, path.read_bytes(), XLSX_TYPE,
                {"Content-Disposition": disposition},
            )

    return Handler


def make_server(agent: Agent, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    app = ChatApp.build(agent)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    return server
