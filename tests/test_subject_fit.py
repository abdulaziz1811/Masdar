"""A result must be about what was asked, and hold the breakdown asked for.

The live answer that started this (2026-10-17): «احصاءات الطاقة الكهربائية
لسنة 2026 حسب المناطق» came back «متوفرة» from GASTAT's industrial
production index. Its description lists «إمدادات الكهرباء والغاز» among the
activities it covers, it holds 2026 (it is monthly), and it has no regions.
The electricity tables, which name the subject and the regions and end
before 2026, were ranked below it. Three faults, each tested here:

* a description naming the subject is not the result being about it;
* a table without the breakdown asked for is not «متوفرة»;
* a loosely related table that holds the year does not answer for the
  closely matching ones that show the year is not published.

The source here is a stand-in with the two kinds of result, shaped as
GASTAT's API returns them (rows of year, breakdown and value).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime

import pytest

from masdar.domain.models import Coverage, DatasetCandidate, Dimension, Resource, Verdict
from masdar.export.tabular import read_table
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.parser import parse
from masdar.pipeline import resolve
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.base import SourceAdapter, SourceDescriptor
from masdar.sources.http import Fetched, HttpClient
from masdar.sources.registry import Registry

QUESTION = "احصاءات الطاقة الكهربائية لسنة 2026 حسب المناطق"
TODAY = date(2026, 10, 17)

ELECTRICITY = DatasetCandidate(
    source_id="stand_in",
    dataset_id="HES0303",
    title_ar="استهلاك الطاقة الكهربائية في المساكن حسب فترة الاستهلاك والمناطق الإدارية",
    resources=(Resource(url="memory://HES0303", format="JSON"),),
)
INDEX = DatasetCandidate(
    source_id="stand_in",
    dataset_id="EFIB1001",
    title_ar="الرقم القياسي للإنتاج الصناعي حسب النشاط الإقتصادي (2010 = 100)",
    description=(
        "مؤشر يعكس التغيرات في حجم الإنتاج الصناعي في نشاط التعدين، والصناعة "
        "التحويلية، ونشاط إمدادات الكهرباء والغاز."
    ),
    keywords="إحصاءات الأعمال قصيرة المدى Industrial Production Index",
    resources=(Resource(url="memory://EFIB1001", format="JSON"),),
)
ROWS = {
    "memory://HES0303": [
        {"السنة": year, "المنطقة الإدارية": region, "الاستهلاك": value}
        for year in (2022, 2023)
        for region, value in (("الرياض", 41.2), ("مكة المكرمة", 37.9), ("الشرقية", 30.4))
    ],
    "memory://EFIB1001": [
        {"السنة": 2026, "الشهر": month, "النشاط": "إمدادات الكهرباء والغاز", "الرقم": 101.3}
        for month in ("يناير", "فبراير", "مارس")
    ],
}


class StandIn(SourceAdapter):
    def search(self, request, limit=10):
        return [ELECTRICITY, INDEX]

    def fetch(self, url):
        body = json.dumps(ROWS[url], ensure_ascii=False).encode("utf-8")
        return Fetched(url, url, 200, body, "application/json", datetime.now(UTC),
                       hashlib.sha256(body).hexdigest())

    def probe(self):
        return "stand-in"


class StandInRegistry(Registry):
    def adapter(self, source_id):
        return StandIn(self.get(source_id), self._http)


DESCRIPTOR = SourceDescriptor(
    id="stand_in", name_ar="واجهة إحصاءات الهيئة العامة للإحصاء", name_en="stand-in",
    base_url="memory://", adapter="stand_in", authority=95, api_verified=True,
    operator_ar="الهيئة العامة للإحصاء",
    topics=tuple(t.id for t in load_lexicon().topics),
)


def ask(question: str, tmp_path):
    http = HttpClient(use_cache=False, retries=1, offline=True)
    registry = StandInRegistry((DESCRIPTOR,), http)
    return Agent(registry=registry, http=http,
                 config=AgentConfig(out_dir=tmp_path / "out", today=TODAY)).answer(question)


class TestNamedForTheSubject:
    def test_a_description_mentioning_it_does_not_name_it(self):
        request = parse(QUESTION)
        assert resolve.on_subject(request, INDEX)  # still a lead
        assert not resolve.names_subject(request, INDEX)
        assert resolve.names_subject(request, ELECTRICITY)

    def test_the_table_read_from_the_file_names_it_too(self):
        # A bulletin titled «إحصاءات سوق العمل للربع الرابع 2025» is about
        # unemployment once the table opened is «المعدل الربعي للبطالة».
        request = parse("عدد المشتغلين 2025")
        bulletin = DatasetCandidate(source_id="x", dataset_id="b",
                                    title_ar="نشرة الربع الرابع 2025")
        assert not resolve.names_subject(request, bulletin)
        assert resolve.names_subject(request, bulletin, "عدد المشتغلين حسب الجنسية")

    def test_without_a_recognised_topic_nothing_is_held_to_a_name(self):
        request = parse("عدد المستفيدين من معسكرات سدايا")
        assert resolve.names_subject(request, INDEX)


class TestTheBreakdownAsked:
    def test_regions_down_the_first_column_are_a_regional_breakdown(self):
        table = read_table(json.dumps([
            {"البيان": name, "2023": 1} for name in ("منطقة الرياض", "منطقة مكة المكرمة",
                                                      "المنطقة الشرقية", "منطقة عسير")
        ], ensure_ascii=False).encode(), "JSON")
        lexicon = load_lexicon()
        assert Dimension.REGION not in table.observed_dimensions(lexicon.dimension_words)
        assert Dimension.REGION in table.observed_dimensions(
            lexicon.dimension_words, lexicon.places)

    def test_two_regions_named_are_not_a_breakdown(self):
        table = read_table(json.dumps([
            {"البيان": "الرياض", "2023": 1}, {"البيان": "الإجمالي", "2023": 2},
            {"البيان": "جدة", "2023": 3},
        ], ensure_ascii=False).encode(), "JSON")
        lexicon = load_lexicon()
        assert Dimension.REGION not in table.observed_dimensions(
            lexicon.dimension_words, lexicon.places)


class TestTheAnswer:
    def test_the_year_not_yet_published_is_said_so(self, tmp_path):
        answer = ask(QUESTION, tmp_path)
        assert answer.verdict is Verdict.NOT_AVAILABLE
        assert answer.primary.candidate is ELECTRICITY
        assert answer.message_ar.startswith("❌")
        # 2023 is offered, never the index's 2026.
        assert [s.year for s in answer.suggestions][:1] == [2023]

    def test_the_index_is_listed_for_what_it_is(self, tmp_path):
        answer = ask(QUESTION, tmp_path)
        index = next(f for f in answer.findings if f.candidate is INDEX)
        assert not index.names_subject
        assert index.verdict is Verdict.PARTIAL
        assert index.missing_dimensions == (Dimension.REGION,)
        assert any("لا يحتوي التفصيل المطلوب (المناطق)" in n for n in index.notes)

    def test_without_the_breakdown_the_index_is_partial_not_available(self, tmp_path):
        # Asked about the index itself, by region: the year is there and the
        # regions are not.
        answer = ask("الرقم القياسي للإنتاج الصناعي 2026 حسب المناطق", tmp_path)
        assert answer.primary.candidate is INDEX
        assert answer.verdict is Verdict.PARTIAL
        assert "غير متاح: التفصيل المطلوب (المناطق)" in answer.message_ar

    @pytest.mark.parametrize("question", [
        "الرقم القياسي للإنتاج الصناعي 2026",
        "الرقم القياسي للإنتاج الصناعي لشهر مارس 2026",
    ])
    def test_asked_without_a_breakdown_the_index_is_available(self, question, tmp_path):
        answer = ask(question, tmp_path)
        assert answer.primary.candidate is INDEX
        assert answer.verdict is Verdict.AVAILABLE


def test_the_coverage_helper_used_here_is_the_real_one():
    # Guard against the stand-in drifting from what the pipeline reads.
    table = read_table(json.dumps(ROWS["memory://HES0303"], ensure_ascii=False).encode(), "JSON")
    assert table.observed_years() == frozenset({2022, 2023})
    assert Coverage.unknown().is_empty
