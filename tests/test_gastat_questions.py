"""Questions a GASTAT statistician asks, over GASTAT's website as it answered.

`fixtures/gastat_site/questions/` holds GASTAT's own search pages, recorded
through Firecrawl on 2026-09-30 and 2026-10-03 and trimmed to their result
lists (gzip-compressed): one page per query the agent sends -- the asker's
words, and GASTAT's own name for the subject («سوق العمل» for «البطالة»,
«الحج» for «الحجاج»). Bulletin pages are not recorded, so bulletins are
offered as links here; what is tested is that the right publication comes
first, and that nothing GASTAT published is reported missing.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

from masdar.domain.models import Verdict
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.http import HttpClient
from masdar.sources.registry import Registry, load_descriptors
from masdar.sources.transport import Response

PAGES = Path(__file__).resolve().parent / "fixtures" / "gastat_site" / "questions"
QUERIES = json.loads((PAGES / "queries.json").read_text(encoding="utf-8"))


class RecordedSite:
    """GASTAT's search as recorded; a query not recorded fails the test."""

    name = "recorded"

    def get(self, url, source_id, params=None, headers=None):
        parts = urlsplit(url)
        if parts.path == "/ar/search":
            query = dict(parse_qsl(parts.query)).get("q", "")
            assert query in QUERIES, f"query not recorded: {query!r}"
            body = gzip.decompress((PAGES / QUERIES[query]).read_bytes())
            return Response(url, url, 200, body, "text/html", {})
        # Bulletin pages were not recorded: each bulletin stays a link.
        return Response(url, url, 200, b"<html></html>", "text/html", {})

    def describe(self):
        return self.name


def ask(question: str, tmp_path: Path):
    http = HttpClient(use_cache=False, retries=1, transport=RecordedSite())
    registry = Registry(tuple(d for d in load_descriptors() if d.id == "gastat"), http)
    agent = Agent(registry=registry, http=http, config=AgentConfig(
        out_dir=tmp_path / "out", today=date(2026, 10, 3)))
    return agent.answer(question)


CASES = [
    # The site finds nothing under «البطالة» but the bulletins under «سوق العمل».
    ("معدل البطالة 2025", "إحصاءات سوق العمل للربع الرابع 2025"),
    ("أحدث بيانات سوق العمل", "إحصاءات سوق العمل للربع الثاني 2026"),
    # Four results under «التضخم», thirty under «أسعار المستهلك».
    ("معدل التضخم أغسطس 2026", "الرقم القياسي لأسعار المستهلك لشهر أغسطس 2026"),
    ("نسبة التضخم 2025", "الرقم القياسي لأسعار المستهلك لشهر ديسمبر 2025"),
    # «مؤشر ثقة المستهلك» shares two words, but not «لأسعار المستهلك».
    ("أحدث مؤشر لأسعار المستهلك", "الرقم القياسي لأسعار المستهلك لشهر أغسطس 2026"),
    ("عدد المعتمرين 2025", "إحصاءات العمرة للربع الرابع 2025"),
    ("عدد الحجاج 2026", "إحصاءات الحج لعام 2026"),
    ("الناتج المحلي الإجمالي 2026",
     "الناتج المحلي الإجمالي والحسابات القومية للربع الثاني لعام 2026"),
    ("إحصاءات الإعاقة 2025", "إحصاءات الإعاقة لعام 2025"),
    # A year asked: the annual release, not December's.
    ("الرقم القياسي للإنتاج الصناعي 2025", "الرقم القياسي السنوي للإنتاج الصناعي لعام 2025"),
    ("إحصاءات العمل التطوعي 2025", "إحصاءات العمل التطوعي لعام 2025"),
    ("الصادرات غير البترولية أغسطس 2026",
     "التجارة الدولية السلعية غير البترولية للمملكة العربية السعودية لشهر أغسطس 2026"),
    ("الرقم القياسي لأسعار العقارات 2026", "الرقم القياسي لأسعار العقارات الربع الثاني 2026"),
    ("الرقم القياسي لتكاليف البناء يوليو 2026", "الرقم القياسي لتكاليف البناء لشهر يوليو 2026"),
    ("التجارة الدولية السلعية 2025", "التجارة الدولية السلعية لعام 2025"),
    ("إحصاءات التعليم والتدريب 2025", "إحصاءات التعليم والتدريب 2025"),
    ("متوسطات أسعار السلع والخدمات أغسطس 2026",
     "متوسطات أسعار السلع والخدمات لشهر أغسطس 2026"),
]


@pytest.mark.parametrize("question, publication", CASES)
def test_the_publication_gastat_issued_comes_first(question, publication, tmp_path):
    answer = ask(question, tmp_path)
    assert answer.primary is not None, answer.message_ar
    assert answer.primary.candidate.title_ar == publication
    # Found on GASTAT's site, it is never reported missing.
    assert answer.verdict is not Verdict.NOT_AVAILABLE
    assert answer.verdict is not Verdict.NO_SOURCE
