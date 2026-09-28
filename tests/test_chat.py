"""The chat: follow-ups understood in context, and a server that only hands
out what it produced."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import date

import pytest

from masdar.chat.server import make_server
from masdar.chat.session import Conversation, Sessions
from masdar.domain.models import Dimension, PeriodKind
from masdar.nlu.lexicon import load_lexicon
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.http import HttpClient
from masdar.sources.registry import DEMO_SOURCES_FILE, Registry, load_descriptors

FIRST = "ابي احصاءات الطاقه الكهربائيه لسنه 2026 حسب المناطق"


@pytest.fixture
def conversation():
    return Conversation(load_lexicon())


class TestFollowUps:
    def test_a_bare_year_keeps_the_topic_and_breakdown(self, conversation):
        conversation.understand(FIRST)
        turn = conversation.understand("طيب 2022")
        assert turn.follow_up
        assert turn.request.topic.id == "electricity"
        assert turn.request.period.years == (2022,)
        assert turn.request.dimensions == (Dimension.REGION,)

    def test_a_breakdown_is_added_and_the_period_kept(self, conversation):
        conversation.understand(FIRST)
        turn = conversation.understand("وحسب الجنس")
        assert turn.follow_up
        assert turn.request.period.years == (2026,)
        assert set(turn.request.dimensions) == {Dimension.REGION, Dimension.GENDER}

    def test_follow_ups_chain(self, conversation):
        conversation.understand(FIRST)
        conversation.understand("2021")
        turn = conversation.understand("وحسب الجنس")
        assert turn.request.period.years == (2021,)

    def test_latest_is_a_period_change(self, conversation):
        conversation.understand(FIRST)
        turn = conversation.understand("اعطني الأحدث")
        assert turn.follow_up and turn.request.period.kind is PeriodKind.LATEST

    def test_a_new_topic_is_a_new_question(self, conversation):
        conversation.understand(FIRST)
        turn = conversation.understand("السكان 2022")
        assert not turn.follow_up
        assert turn.request.topic.id == "population"
        assert turn.request.dimensions == ()

    def test_new_search_words_are_a_new_question(self, conversation):
        conversation.understand(FIRST)
        turn = conversation.understand("عدد الصيدليات 2023")
        assert not turn.follow_up
        assert turn.request.topic is None

    def test_the_first_message_is_never_a_follow_up(self, conversation):
        assert not conversation.understand("2022").follow_up

    def test_the_record_names_both_messages(self, conversation):
        conversation.understand(FIRST)
        raw = conversation.understand("2022").request.raw_query
        assert "2022" in raw and FIRST in raw

    def test_reset_forgets(self, conversation):
        conversation.understand(FIRST)
        conversation.reset()
        assert not conversation.understand("2022").follow_up


class TestSessions:
    def test_unknown_ids_get_a_fresh_conversation(self):
        sessions = Sessions(load_lexicon())
        new_id, _ = sessions.get("made-up")
        assert new_id != "made-up"

    def test_oldest_is_forgotten_beyond_capacity(self):
        sessions = Sessions(load_lexicon(), capacity=2)
        first, _ = sessions.get(None)
        sessions.get(None)
        sessions.get(None)
        again, _ = sessions.get(first)
        assert again != first


@pytest.fixture
def server(tmp_path):
    http = HttpClient(offline=True, use_cache=False, retries=1)
    registry = Registry(load_descriptors(DEMO_SOURCES_FILE), http)
    agent = Agent(
        registry=registry,
        http=http,
        config=AgentConfig(out_dir=tmp_path / "out", today=date(2026, 9, 27)),
    )
    srv = make_server(agent, host="127.0.0.1", port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url):
    with _OPENER.open(url, timeout=30) as response:
        return response.status, dict(response.headers), response.read()


def post(url, body, raw=None):
    data = raw if raw is not None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestServer:
    def test_serves_the_page_with_a_restrictive_policy(self, server):
        status, headers, body = get(server + "/")
        assert status == 200
        assert "dir=\"rtl\"" in body.decode("utf-8")
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]

    def test_the_unpublished_year_is_refused_with_the_latest_offered(self, server):
        status, data = post(server + "/api/ask", {"message": FIRST})
        assert status == 200
        assert data["verdict"] == "not_available"
        assert "ما لقيت بيانات" in data["headline"]
        assert [s["year"] for s in data["suggestions"]] == [2024]
        assert data["findings"][0]["download"] is None

    def test_a_suggestion_continues_the_conversation_and_yields_a_file(self, server):
        _, first = post(server + "/api/ask", {"message": FIRST})
        ask = first["suggestions"][0]["ask"]
        status, data = post(server + "/api/ask", {"session": first["session"], "message": ask})
        assert status == 200 and data["verdict"] == "available"
        assert data["understood"]["follow_up"]
        download = data["findings"][0]["download"]
        status, headers, body = get(server + download)
        assert status == 200 and body[:2] == b"PK"
        assert headers["Content-Type"].startswith("application/vnd.openxmlformats")

    def test_only_issued_tokens_download(self, server):
        for path in ("/files/../../etc/passwd", "/files/%2e%2e%2fpyproject.toml", "/files/x"):
            with pytest.raises(urllib.error.HTTPError) as caught:
                get(server + path)
            assert caught.value.code == 404

    def test_rejects_empty_oversized_and_malformed_requests(self, server):
        assert post(server + "/api/ask", {"message": "  "})[0] == 400
        assert post(server + "/api/ask", {"message": "س" * 600})[0] == 400
        assert post(server + "/api/ask", None, raw=b"{not json")[0] == 400
        assert post(server + "/api/ask", None, raw=b"x" * 20_000)[0] == 413

    def test_non_http_links_are_not_passed_to_the_page(self, server):
        _, data = post(server + "/api/ask", {"message": "الكهرباء 2024 حسب المناطق"})
        for finding in data["findings"]:
            for key in ("landing_url", "resource_url"):
                assert finding[key] is None or finding[key].startswith(("http://", "https://"))
