"""Live indicators that ministries publish as small JSON endpoints.

Some official sources publish the figure as it stands today rather than an
annual table: HRSD's real-time labour-market indicators, MEWA's counters of
registered camels or bee colonies. They are listed on the national portal's
real-time API page, answer without a key, and change as the underlying
registers change.

What such a figure is, and is not, decides how it is handled:

* **A snapshot.** The value is what the source reports at the moment of
  retrieval, and the source keeps no history. Every row carries the date it
  was retrieved, and coverage is that year alone -- observed, and exhaustive
  for this source, which cannot show any other year. A question about an
  earlier year is therefore answered "not available here", and the verdict
  logic offers the annual sources instead.
* **Not an annual statistic.** A value read in September is not the year's
  total; each answer says so in its notes.
* **The fingerprint is of what the source sent.** The response is reshaped
  into rows (adding the retrieval date, and a label for a bare counter), but
  the sha256 cited is of the original bytes, so it can be checked against the
  source.

Indicators are configuration, one entry per endpoint, in two shapes:
`records` (a JSON array of flat objects) and `counter` (MEWA's
`{"IsSuccess": true, "Data": "123"}`).
"""

from __future__ import annotations

import json
from dataclasses import replace

from masdar.domain.models import (
    Coverage,
    DataRequest,
    DatasetCandidate,
    Dimension,
    Resource,
)
from masdar.nlu.normalize import contains_phrase
from masdar.nlu.parser import typed_phrase_score
from masdar.sources.base import SourceAdapter, SourceError, SourceUnreachable

YEAR_COLUMN = "السنة"
RETRIEVED_COLUMN = "تاريخ الاستخراج"
SNAPSHOT_CAVEAT = (
    "بيانات لحظية: القيم كما يعرضها المصدر وقت الاستخراج (عمود «تاريخ الاستخراج»)، "
    "وليست إحصاءً سنوياً مكتملاً، ولا يحتفظ المصدر بقيم سنوات سابقة."
)
SHAPES = ("records", "counter")


class LiveDataAdapter(SourceAdapter):
    def _indicators(self) -> list[dict]:
        configured = self.descriptor.api.get("indicators") or []
        indicators = [i for i in configured if isinstance(i, dict) and i.get("id") and i.get("url")]
        for indicator in indicators:
            if indicator.get("shape", "records") not in SHAPES:
                raise SourceError(self.id, f"شكل غير معروف للمؤشر {indicator['id']}")
        return indicators

    def _by_url(self, url: str) -> dict | None:
        return next((i for i in self._indicators() if i["url"] == url), None)

    # -- search --------------------------------------------------------
    def _score(self, request: DataRequest, indicator: dict) -> float:
        text = " ".join(
            str(indicator.get(k) or "") for k in ("title_ar", "title_en", "keywords")
        )
        score, _ = typed_phrase_score(request, text)
        score += sum(2.0 for t in request.free_terms if contains_phrase(text, t))
        if score <= 0:
            # Sharing a topic is not enough: a question about camels must not
            # open the bee-colony counter because both are "agriculture".
            return 0.0
        if request.topic and request.topic.id in (indicator.get("topics") or ()):
            score += 3.0
        return score

    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        scored = [(self._score(request, i), i) for i in self._indicators()]
        ranked = sorted((p for p in scored if p[0] > 0), key=lambda p: -p[0])
        return [self._candidate(indicator) for _, indicator in ranked[:limit]]

    def _candidate(self, indicator: dict) -> DatasetCandidate:
        dimensions = []
        for value in indicator.get("dimensions") or ():
            try:
                dimensions.append(Dimension(value))
            except ValueError:
                continue
        return DatasetCandidate(
            source_id=self.id,
            dataset_id=str(indicator["id"]),
            title_ar=str(indicator.get("title_ar") or ""),
            title_en=str(indicator.get("title_en") or ""),
            description=str(indicator.get("description") or ""),
            keywords=str(indicator.get("keywords") or ""),
            landing_url=str(
                indicator.get("landing") or self.descriptor.site_url or indicator["url"]
            ),
            publisher_ar=self.descriptor.operator_ar or self.descriptor.name_ar,
            publisher_en=self.descriptor.operator_en or self.descriptor.name_en,
            resources=(Resource(url=indicator["url"], format="JSON", title="live"),),
            # Nothing is known until the endpoint is read; the rows then carry
            # the retrieval year, which is the whole of this source's coverage.
            claimed_coverage=Coverage.unknown(),
            provided_dimensions=tuple(dimensions),
            # What exactly is counted, in the publisher's words, comes first:
            # a reader must not take "tagged camels" for "all camels".
            caveats=tuple(
                note for note in (str(indicator.get("note_ar") or "").strip(), SNAPSHOT_CAVEAT)
                if note
            ),
        )

    # -- fetch ---------------------------------------------------------
    def fetch(self, url: str):
        fetched = super().fetch(url)
        indicator = self._by_url(url)
        if indicator is None:
            return fetched  # not one of ours; pass through untouched
        try:
            payload = json.loads(fetched.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceError(self.id, f"استجابة غير صالحة: {exc}") from exc

        rows = self._rows(indicator, payload)
        stamp = fetched.retrieved_at.date().isoformat()
        stamped = [
            {YEAR_COLUMN: fetched.retrieved_at.year, RETRIEVED_COLUMN: stamp, **row}
            for row in rows
        ]
        body = json.dumps(stamped, ensure_ascii=False).encode("utf-8")
        # Same sha256, retrieval time and URL as the response itself: the
        # citation points at what the source sent.
        return replace(fetched, content=body, media_type="application/json")

    def _rows(self, indicator: dict, payload: object) -> list[dict]:
        if indicator.get("shape") == "counter":
            if not isinstance(payload, dict) or not payload.get("IsSuccess"):
                raise SourceError(self.id, "المصدر لم يُرجع قيمة لهذا المؤشر حالياً")
            value = payload.get("Data", payload.get("Count"))
            try:
                number = int(str(value).replace(",", ""))
            except (TypeError, ValueError):
                raise SourceError(self.id, f"قيمة غير رقمية: {value!r}") from None
            label = indicator.get("value_label_ar") or indicator.get("title_ar") or "القيمة"
            return [{"المؤشر": label, "القيمة": number}]

        if not isinstance(payload, list) or not all(isinstance(r, dict) for r in payload):
            raise SourceError(self.id, "المتوقع مصفوفة سجلات JSON")
        if not payload:
            raise SourceError(self.id, "المصدر أعاد قائمة فارغة")
        return payload

    # -- diagnosis -----------------------------------------------------
    def probe(self) -> str:
        results = []
        for indicator in self._indicators():
            try:
                self.fetch(indicator["url"])
                results.append(f"{indicator['id']}: يعمل")
            except SourceUnreachable as exc:
                results.append(f"{indicator['id']}: تعذّر الوصول ({exc.reason})")
            except SourceError as exc:
                results.append(f"{indicator['id']}: {exc.reason}")
        return "؛ ".join(results) or "لا مؤشرات مُعدّة"
