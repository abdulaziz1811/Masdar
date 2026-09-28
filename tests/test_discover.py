"""Discovering the national platform's publishers and their topics.

The listing call cannot be reached from the development environment, so its
fake follows what the platform's own page code reads (a Spring-style page of
items with `publisherID`, `nameAr`, `numberOfDatasets`). The classification
runs on the Ministry of Commerce and Ministry of Energy catalogues recorded
live on 2026-09-28.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

from masdar.nlu.lexicon import load_lexicon
from masdar.sources import discover
from masdar.sources.base import SourceError
from masdar.sources.http import HttpClient
from masdar.sources.registry import Registry, load_descriptors
from masdar.sources.transport import Response, Transport

OPEN_DATA = Path(__file__).resolve().parent / "fixtures" / "open_data"
ENERGY = "886f6d71-8034-47f7-914d-9613825d9153"
COMMERCE = "59a61901-0624-4616-b7b0-688a82fca411"


def page(items, number, total_pages):
    return {"content": items, "number": number, "totalPages": total_pages,
            "last": number + 1 >= total_pages}


class Platform(Transport):
    name = "platform"

    def __init__(self, listing=None):
        self.listing = listing or [
            page([{"publisherID": ENERGY, "nameAr": "وزارة الطاقة", "numberOfDatasets": 100},
                  {"publisherID": COMMERCE, "nameAr": "وزارة التجارة", "numberOfDatasets": 113}],
                 0, 2),
            page([{"publisherID": "ffffffff-0000-0000-0000-000000000001",
                   "nameAr": "جهة تجريبية", "numberOfDatasets": 3}], 1, 2),
        ]
        self.urls: list[str] = []

    def get(self, url, source_id, params=None, headers=None):
        self.urls.append(url)
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        if parts.path == "/api/publishers":
            number = int(query.get("page", 0))
            body = self.listing[number] if number < len(self.listing) else page([], number, 2)
            return self._json(url, body)
        if parts.path.endswith("/organizations"):
            org = query.get("organization")
            if org in (ENERGY, "وزارة الطاقة"):
                return self._raw(url, (OPEN_DATA / "organization_energy.json").read_bytes())
            if org in (COMMERCE, "وزارة التجارة"):
                return self._raw(url, (OPEN_DATA / "organization_commerce.json").read_bytes())
            raise SourceError(source_id, "not found: Publisher Not Found")
        raise AssertionError(f"unexpected request: {url}")

    @staticmethod
    def _json(url, body):
        return Response(url, url, 200, json.dumps(body, ensure_ascii=False).encode(),
                        "application/json", {})

    @staticmethod
    def _raw(url, body):
        return Response(url, url, 200, body, "application/json", {})


def adapter(transport=None, **api):
    base = next(d for d in load_descriptors() if d.id == "saudi_open_data")
    descriptor = replace(base, api={**base.api, **api})
    http = HttpClient(use_cache=False, retries=1, transport=transport or Platform())
    return Registry((descriptor,), http).adapter("saudi_open_data")


class TestListing:
    def test_every_page_is_read(self):
        found = discover.list_publishers(adapter())
        assert [p.name_ar for p in found] == ["وزارة الطاقة", "وزارة التجارة", "جهة تجريبية"]
        assert found[1].datasets == 113

    def test_an_unknown_shape_names_what_arrived(self):
        transport = Platform(listing=[{"unexpected": True}])
        with pytest.raises(SourceError, match="unexpected"):
            discover.list_publishers(adapter(transport))


class TestClassification:
    def test_topics_come_from_the_publishers_own_titles(self):
        payload = json.loads((OPEN_DATA / "organization_commerce.json").read_text("utf-8"))
        titles = [f"{d['titleAr']} {d['titleEn']}" for d in payload["datasets"]]
        topics = discover.infer_topics(titles, load_lexicon())
        assert topics[:2] == ("business", "trade")

    def test_classify_counts_and_labels(self):
        found = discover.resolve_names(adapter(), ["وزارة الطاقة"])
        discover.classify(adapter(), found)
        assert found[0].id == ENERGY
        assert found[0].datasets == 8
        assert "electricity" in found[0].topics

    def test_an_unknown_name_is_reported_not_invented(self):
        found = discover.resolve_names(adapter(), ["جهة لا وجود لها"])
        assert found[0].problems and not found[0].id


class TestTheFile:
    def test_round_trip_and_problems_left_out(self, tmp_path):
        ok = discover.Publisher(id=COMMERCE, name_ar="وزارة التجارة", datasets=113,
                                topics=("business",), found_by="test")
        broken = discover.Publisher(name_ar="جهة", problems=["not found"])
        path = tmp_path / "found.yaml"
        path.write_text(discover.to_yaml([ok, broken]), encoding="utf-8")
        loaded = discover.load_discovered(path)
        assert [e["name_ar"] for e in loaded] == ["وزارة التجارة"]
        assert loaded[0]["topics"] == ["business"]

    def test_the_adapter_searches_discovered_publishers(self, tmp_path):
        path = tmp_path / "found.yaml"
        path.write_text(discover.to_yaml([discover.Publisher(
            id=COMMERCE, name_ar="وزارة التجارة", topics=("business",), found_by="test",
        )]), encoding="utf-8")
        orgs = adapter(discovered_file=str(path))._organizations("business")
        assert COMMERCE in {o.get("id") for o in orgs}

    def test_configured_publishers_win_on_a_clash(self, tmp_path):
        path = tmp_path / "found.yaml"
        path.write_text(discover.to_yaml([discover.Publisher(
            id=ENERGY, name_ar="وزارة الطاقة", topics=("tourism",), found_by="test",
        )]), encoding="utf-8")
        orgs = adapter(discovered_file=str(path))._organizations()
        energy = [o for o in orgs if o.get("id") == ENERGY]
        assert len(energy) == 1 and "tourism" not in (energy[0].get("topics") or [])

    def test_unclassified_publishers_answer_only_topicless_questions(self, tmp_path):
        path = tmp_path / "found.yaml"
        path.write_text(discover.to_yaml([discover.Publisher(
            id="x-1", name_ar="جهة", topics=(discover.UNCLASSIFIED,), found_by="test",
        )]), encoding="utf-8")
        a = adapter(discovered_file=str(path))
        assert "x-1" in {o.get("id") for o in a._organizations(None)}
        assert "x-1" not in {o.get("id") for o in a._organizations("health")}


class TestPaging:
    def test_a_capped_page_size_does_not_end_the_listing(self):
        # Pages of 2 although 100 were asked for, and no paging metadata.
        items = [{"publisherID": f"p-{i}", "nameAr": f"جهة {i}"} for i in range(5)]
        pages = [items[0:2], items[2:4], items[4:5], []]
        found = discover.list_publishers(adapter(Platform(listing=pages)))
        assert len(found) == 5

    def test_a_server_that_ignores_paging_cannot_loop(self):
        same = [{"publisherID": "p-1", "nameAr": "جهة"}]
        found = discover.list_publishers(adapter(Platform(listing=[same] * 60)))
        assert len(found) == 1
