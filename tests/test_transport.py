"""Tests for the fetch layer.

The behaviour under test is mostly about honesty: a WAF rejection must not
arrive downstream as an empty result, and a deterministic refusal must not be
retried as though it were a hiccup.
"""

import pytest

from masdar.sources.base import (
    RetryableSourceError,
    SourceError,
    SourceRejected,
    SourceUnreachable,
)
from masdar.sources.http import HttpClient
from masdar.sources.transport import (
    DirectTransport,
    FirecrawlTransport,
    Response,
    Transport,
    build_transport,
    looks_rejected,
)

WAF_BODY = (
    b"<html><head><title>Request Rejected</title></head><body>"
    b"The requested URL was rejected. Please consult with your administrator."
    b"<br><br>Your support ID is: 3620635771771463286</body></html>"
)


class FakeTransport(Transport):
    """Replays scripted outcomes and counts calls."""

    name = "fake"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, source_id, params=None):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else self.outcomes
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def response(content=b"<html><body>ok</body></html>", media_type="text/html", status=200):
    return Response(
        url="https://example.invalid/x",
        final_url="https://example.invalid/x",
        status=status,
        content=content,
        media_type=media_type,
        headers={},
    )


class TestWafDetection:
    def test_recognises_the_rejection_page(self):
        assert looks_rejected(WAF_BODY, "text/html")

    def test_ordinary_html_is_not_a_rejection(self):
        assert not looks_rejected(b"<html><body><h1>datasets</h1></body></html>", "text/html")

    def test_json_body_is_not_a_rejection(self):
        assert not looks_rejected(b'{"data": []}', "application/json")

    def test_a_rejection_raises_rather_than_returning_a_body(self):
        # The crucial case: parsed as HTML this page yields zero links, which
        # would read as "this source has no data".
        client = HttpClient(
            use_cache=False, retries=1, transport=FakeTransport([response(WAF_BODY)])
        )
        with pytest.raises(SourceRejected) as caught:
            client.get("https://example.invalid/x", source_id="s")
        assert "ليس دليلاً على عدم وجود البيانات" in caught.value.reason


class TestRetryPolicy:
    def test_transient_failure_is_retried(self):
        transport = FakeTransport(
            [RetryableSourceError("s", "HTTP 503"), response()]
        )
        client = HttpClient(use_cache=False, retries=3, transport=transport)
        fetched = client.get("https://example.invalid/x", source_id="s")
        assert fetched.status == 200
        assert transport.calls == 2

    def test_rejection_is_not_retried(self):
        # Repeating a deterministic refusal wastes time and looks like abuse.
        transport = FakeTransport([SourceRejected("s", "rejected")] * 3)
        client = HttpClient(use_cache=False, retries=3, transport=transport)
        with pytest.raises(SourceRejected):
            client.get("https://example.invalid/x", source_id="s")
        assert transport.calls == 1

    def test_exhausted_transient_failures_become_unreachable(self):
        transport = FakeTransport([RetryableSourceError("s", "HTTP 503")] * 3)
        client = HttpClient(use_cache=False, retries=2, transport=transport)
        with pytest.raises(SourceUnreachable):
            client.get("https://example.invalid/x", source_id="s")


class TestProvenanceFromFetch:
    def test_sha256_and_timestamp_are_always_present(self):
        client = HttpClient(use_cache=False, retries=1, transport=FakeTransport([response()]))
        fetched = client.get("https://example.invalid/x", source_id="s")
        assert len(fetched.sha256) == 64
        assert fetched.retrieved_at is not None
        assert fetched.byte_size > 0


class TestBackendSelection:
    def test_default_is_direct(self, monkeypatch):
        monkeypatch.delenv("MASDAR_HTTP_BACKEND", raising=False)
        assert isinstance(build_transport(), DirectTransport)

    def test_env_selects_firecrawl(self, monkeypatch):
        monkeypatch.setenv("MASDAR_HTTP_BACKEND", "firecrawl")
        assert isinstance(build_transport(), FirecrawlTransport)

    def test_unknown_backend_fails_loudly(self):
        with pytest.raises(ValueError):
            build_transport("carrier-pigeon")

    def test_firecrawl_without_a_key_is_unreachable_not_silent(self, monkeypatch):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        with pytest.raises(SourceUnreachable):
            FirecrawlTransport(api_key="").get("https://example.invalid", "s")

    def test_direct_transport_identifies_itself(self):
        agent = DirectTransport()._session.headers["User-Agent"]
        assert "Masdar" in agent


class TestSpaGuard:
    """An SPA's empty HTML shell must not be reported as an empty source."""

    def _adapter(self, transport_name):
        from masdar.sources.adapters.html_index import HtmlIndexAdapter
        from masdar.sources.base import SourceDescriptor

        transport = FakeTransport([response()])
        transport.name = transport_name
        client = HttpClient(use_cache=False, retries=1, transport=transport)
        descriptor = SourceDescriptor(
            id="spa",
            name_ar="x",
            name_en="x",
            base_url="https://example.invalid",
            adapter="html_index",
            rendering="spa",
            topics=("electricity",),
        )
        return HtmlIndexAdapter(descriptor, client)

    def test_plain_http_refuses_an_spa(self):
        with pytest.raises(SourceError) as caught:
            self._adapter("direct").probe()
        assert "SPA" in caught.value.reason

    def test_a_javascript_transport_is_allowed(self):
        # Firecrawl executes the page, so the shell is not a problem.
        assert self._adapter("firecrawl").probe()


class TestSearchUrls:
    def test_search_path_is_used_when_configured(self):
        from masdar.sources.registry import Registry, load_descriptors

        client = HttpClient(offline=True, use_cache=False, retries=1)
        adapter = Registry(load_descriptors(), client).adapter("gastat")
        url = adapter._index_url("الكهرباء")
        assert "statistics-tabs?q=" in url
        assert "%D8%A7%D9%84%D9%83%D9%87%D8%B1%D8%A8%D8%A7%D8%A1" in url

    def test_plain_index_used_without_a_query(self):
        from masdar.sources.registry import Registry, load_descriptors

        client = HttpClient(offline=True, use_cache=False, retries=1)
        adapter = Registry(load_descriptors(), client).adapter("gastat")
        assert adapter._index_url().endswith("/ar/statistics-tabs")


class TestRecordedAccessFacts:
    """The registry must carry what was verified, not what was assumed."""

    def test_national_portal_facts_are_recorded(self):
        from masdar.sources.registry import load_descriptors

        portal = next(d for d in load_descriptors() if d.id == "saudi_open_data")
        assert portal.geo_restricted is True
        assert portal.waf is True
        assert portal.rendering == "spa"
        assert portal.api_verified is False  # path still unconfirmed

    def test_unverified_geo_claims_are_not_asserted(self):
        from masdar.sources.registry import load_descriptors

        # Only open.data.gov.sa was actually demonstrated to be restricted.
        restricted = {d.id for d in load_descriptors() if d.geo_restricted}
        assert restricted == {"saudi_open_data"}
