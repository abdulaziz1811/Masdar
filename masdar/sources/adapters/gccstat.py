"""GCC-Stat's public SDMX service, for the Kingdom's figures.

The Statistical Centre for the Cooperation Council for the Arab Countries of
the Gulf (GCC-Stat) publishes the member states' official figures through a
Fusion Registry SDMX service that answers anonymously. It compiles them from
the national offices -- for the Kingdom, mostly GASTAT -- so it is a second
copy of a Saudi figure: ranked after the Saudi publishers and labelled as
regional in every answer.

* **The indicators are configuration.** The service lists 234 dataflows,
  English-named. The ones used here are GCC-Stat's "mobile" flows, one
  indicator each, keyed by country and year alone. Each is listed in
  `sources.yaml` with an Arabic name and the unit it is published in; the
  flows carry no unit, so each one was checked against GCC-Stat's detailed
  tables or against the figure's known size before it was listed.
* **Data: live.** The Kingdom's series (`.../<flow>/SAU`) is fetched at
  question time as SDMX-JSON and turned into one row per year, with the
  unit and GCC-Stat's own footnotes («بيان 2023م مرحل من عام 2022م»).
* **A repeated value is said to be one.** Several series end with the same
  figure year after year (deaths: 80,246 for 2018 to 2022), which is a value
  carried forward, not a new count. Those years keep their value, and the
  row and the answer say it repeats the earlier one.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from masdar.domain.models import Coverage, DataRequest, DatasetCandidate, Resource
from masdar.sources.adapters.worldbank import MIN_RELEVANCE, WorldBankAdapter
from masdar.sources.base import SourceAdapter, SourceError, SourceUnreachable

API = "https://sdmx.marsa.gccstat.org/FusionRegistry/ws/public/sdmxapi/rest"
AGENCY = "GCCSTAT.MOBILE"
COUNTRY = "SAU"
YEAR_COLUMN = "السنة"
REGIONAL_CAVEAT = (
    "مصدر إقليمي: المركز الإحصائي لدول مجلس التعاون الخليجي، ينقل عن الجهات الرسمية "
    "في المملكة. قد يختلف عن أحدث ما تنشره الجهة السعودية المختصة."
)
# A run of this many equal values at a series' end is a figure carried
# forward, not a count repeated: three identical yearly death totals do not
# happen. A small figure -- a rate of 2.5% -- can repeat by chance, so only
# totals this large are judged.
REPEATED_RUN = 3
REPEATED_AT_LEAST = 1000
_OTHER_MEMBERS = ("الامارات", "الإمارات", "البحرين", "عمان", "عُمان", "قطر", "الكويت",
                  "UAE", "Bahrain", "Oman", "Qatar", "Kuwait")
_SAUDI = ("السعودية", "المملكة", "KSA", "Saudi")


class GccStatAdapter(SourceAdapter):
    def _indicators(self) -> list[dict]:
        configured = self.descriptor.api.get("indicators") or []
        return [i for i in configured if isinstance(i, dict) and i.get("id") and i.get("title_ar")]

    def _data_url(self, flow: str) -> str:
        return (
            f"{API}/data/{AGENCY},{flow},1.0/{COUNTRY}"
            "?format=sdmx-json&dimensionAtObservation=AllDimensions"
        )

    def _by_url(self, url: str) -> dict | None:
        return next((i for i in self._indicators() if self._data_url(i["id"]) == url), None)

    # -- search --------------------------------------------------------
    def relevance(self, request: DataRequest, indicator: dict) -> float:
        """The World Bank's reading of an indicator name, on each Arabic name.

        An indicator is listed under its name and the other names people
        give it («السياحة الوافدة» for «عدد السياح القادمين»); the closest
        one counts.
        """
        names = [indicator["title_ar"], *(indicator.get("aliases") or ())]
        return max(
            WorldBankAdapter.relevance(self, request, {"ar": str(name), "en": ""})
            for name in names
        )

    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        scored = [(self.relevance(request, i), i) for i in self._indicators()]
        ranked = sorted((p for p in scored if p[0] >= MIN_RELEVANCE), key=lambda p: -p[0])
        candidates = []
        for relevance, indicator in ranked[:limit]:
            candidate = self._candidate(indicator)
            candidate.relevance = round(relevance, 3)
            candidates.append(candidate)
        return candidates

    def _candidate(self, indicator: dict) -> DatasetCandidate:
        unit = str(indicator.get("unit_ar") or "")
        caveats = [REGIONAL_CAVEAT]
        if unit:
            caveats.append(f"الوحدة: {unit}.")
        name_en = indicator.get("title_en") or indicator["id"]
        caveats.append(f"اسم المؤشر لدى المركز: {name_en} ({indicator['id']})")
        url = self._data_url(indicator["id"])
        return DatasetCandidate(
            source_id=self.id,
            dataset_id=str(indicator["id"]),
            title_ar=str(indicator["title_ar"]),
            title_en=str(indicator.get("title_en") or ""),
            # The other names, and the topics as the World Bank's indicators
            # carry them, so the two compete on the same terms.
            keywords=" ".join(
                str(w)
                for w in (*(indicator.get("aliases") or ()), *(indicator.get("topics") or ()))
            ),
            landing_url=self.descriptor.site_url or API,
            publisher_ar=self.descriptor.operator_ar or self.descriptor.name_ar,
            publisher_en=self.descriptor.operator_en or self.descriptor.name_en,
            resources=(Resource(url=url, format="JSON", title="GCC-Stat SDMX"),),
            claimed_coverage=Coverage.unknown(),
            caveats=tuple(caveats),
        )

    # -- fetch ---------------------------------------------------------
    def fetch(self, url: str):
        fetched = super().fetch(url)
        indicator = self._by_url(url)
        if indicator is None:
            return fetched
        try:
            payload = json.loads(fetched.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceError(self.id, f"استجابة غير صالحة: {exc}") from exc
        rows = sdmx_rows(payload, str(indicator["title_ar"]), str(indicator.get("unit_ar") or ""))
        if not rows:
            raise SourceError(self.id, "لا توجد قيم للمملكة في هذا المؤشر")
        repeated = mark_repeats(rows)
        body = json.dumps(rows, ensure_ascii=False).encode("utf-8")
        # The citation keeps the sha256 of what GCC-Stat sent.
        return replace(fetched, content=body, media_type="application/json",
                       notes=(*fetched.notes, repeated) if repeated else fetched.notes)

    def probe(self) -> str:
        indicators = self._indicators()
        if not indicators:
            return "لا مؤشرات مُعدّة"
        try:
            self.fetch(self._data_url(indicators[0]["id"]))
        except SourceUnreachable as exc:
            return f"{len(indicators)} مؤشراً، وتعذّر الوصول إلى الواجهة ({exc.reason})"
        return f"{len(indicators)} مؤشراً، والواجهة تستجيب ({indicators[0]['title_ar']})"


def sdmx_rows(payload: object, title: str, unit: str) -> list[dict]:
    """The Kingdom's observations in an SDMX-JSON data message, one row each.

    `dimensionAtObservation=AllDimensions` puts every observation under a key
    of dimension positions ("0:7": the first country, the eighth period);
    the value comes first in its array and the attributes follow, each an
    index into that attribute's values.
    """
    try:
        data = payload["data"]  # type: ignore[index]
        dataset = data["dataSets"][0]
        structure = data["structure"]
        dimensions = structure["dimensions"]["observation"]
        observations = dataset.get("observations") or {}
    except (KeyError, IndexError, TypeError):
        raise SourceError("gccstat", "رد SDMX بغير البنية المتوقعة") from None

    positions = {d.get("id"): i for i, d in enumerate(dimensions)}
    time_at = positions.get("TIME_PERIOD")
    country_at = next((i for d, i in positions.items() if d and "COUNTRY" in d), None)
    if time_at is None:
        raise SourceError("gccstat", "رد SDMX بلا بُعد زمني")
    attributes = (structure.get("attributes") or {}).get("observation") or []
    shown = _arabic_notes(attributes)
    dataset_notes = _dataset_notes(structure, dataset)

    rows = []
    for key, entry in observations.items():
        try:
            at = [int(p) for p in key.split(":")]
            period = dimensions[time_at]["values"][at[time_at]]["id"]
        except (ValueError, IndexError, KeyError):
            continue
        if country_at is not None:
            country = dimensions[country_at]["values"][at[country_at]].get("id")
            if country != COUNTRY:
                continue
        value = _number(entry[0] if entry else None)
        if value is None:
            continue
        notes = []
        for position, values in shown:
            index = entry[position + 1] if len(entry) > position + 1 else None
            if isinstance(index, int) and 0 <= index < len(values):
                notes.append(values[index])
        notes.extend(dataset_notes)
        year = re.match(r"\d{4}", str(period))
        if year is None:
            continue
        row = {YEAR_COLUMN: int(year.group())}
        if str(period) != year.group():
            row["الفترة"] = str(period)
        row.update({"المؤشر": title, "القيمة": value})
        if unit:
            row["الوحدة"] = unit
        row["ملاحظة"] = "؛ ".join(dict.fromkeys(notes))
        rows.append(row)
    rows.sort(key=lambda r: (r[YEAR_COLUMN], r.get("الفترة", "")))
    return rows


def mark_repeats(rows: list[dict]) -> str:
    """Say so when a series ends on the same value year after year.

    Returns the note for the answer ("" when there is none); the repeated
    years' own rows say which year they repeat.
    """
    if len(rows) < REPEATED_RUN or any("الفترة" in r for r in rows):
        return ""
    last = rows[-1]["القيمة"]
    if abs(last) < REPEATED_AT_LEAST:
        return ""
    run = 0
    for row in reversed(rows):
        if row["القيمة"] != last:
            break
        run += 1
    if run < REPEATED_RUN:
        return ""
    first = rows[-run]
    for row in rows[-run + 1:]:
        note = f"القيمة نفسها المنشورة لسنة {first[YEAR_COLUMN]}، فقد تكون مرحّلة"
        row["ملاحظة"] = "؛ ".join(n for n in (row["ملاحظة"], note) if n)
    return (
        f"القيمة نفسها ({_shown(last)}) منشورة لكل السنوات {first[YEAR_COLUMN]}–"
        f"{rows[-1][YEAR_COLUMN]}، فالأرجح أنها مرحّلة من {first[YEAR_COLUMN]} لا مرصودة "
        "لكل سنة؛ تُراجع الجهة السعودية المختصة لما بعدها."
    )


def _arabic_notes(attributes: list[dict]) -> list[tuple[int, list[str]]]:
    """The observation attributes worth showing: GCC-Stat's Arabic footnotes."""
    shown = []
    for position, attribute in enumerate(attributes):
        if not str(attribute.get("id", "")).endswith("FOOTNOTE_AR"):
            continue
        values = [
            str(v.get("name") or v.get("id") or "").strip()
            for v in attribute.get("values") or ()
        ]
        shown.append((position, values))
    return shown


def _dataset_notes(structure: dict, dataset: dict) -> list[str]:
    """Footnotes on the whole series that may bear on the Kingdom.

    One that names other member states and not the Kingdom («بيانات الإمارات
    والبحرين من منظمة العمل الدولية») is about them, and left out.
    """
    attributes = (structure.get("attributes") or {}).get("dataset") or []
    chosen = dataset.get("attributes") or []
    notes = []
    for position, attribute in enumerate(attributes):
        if not str(attribute.get("id", "")).endswith("FOOTNOTE_AR"):
            continue
        values = attribute.get("values") or []
        index = chosen[position] if position < len(chosen) else None
        if not isinstance(index, int) or not 0 <= index < len(values):
            continue
        text = str(values[index].get("name") or values[index].get("id") or "").strip()
        if not text:
            continue
        others = any(member in text for member in _OTHER_MEMBERS)
        if others and not any(name in text for name in _SAUDI):
            continue
        notes.append(text)
    return notes


def _number(value: object) -> int | float | None:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return int(number) if number.is_integer() else round(number, 4)


def _shown(value: int | float) -> str:
    return f"{value:,}"
