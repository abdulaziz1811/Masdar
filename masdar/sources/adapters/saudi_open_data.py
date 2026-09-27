"""Adapter for the Saudi Open Data Portal (open.data.gov.sa).

The portal's response shape could not be confirmed against the live service
while this was written (the host is unreachable from the build environment),
so the mapping below accepts the field names such portals commonly use and,
when it recognises none of them, raises a `SourceError` naming the keys it
actually received. That turns a wrong guess into a one-line fix in
sources.yaml instead of a silent empty result.
"""

from __future__ import annotations

from datetime import date, datetime

from masdar.domain.models import (
    Coverage,
    CoverageOrigin,
    DataRequest,
    DatasetCandidate,
    Resource,
)
from masdar.nlu.parser import extract_years
from masdar.sources.base import SourceAdapter, SourceError

# Candidate field names, tried in order. Extend rather than replace when the
# live shape is confirmed.
_LIST_KEYS = ("data", "datasets", "results", "content", "items", "records")
_ID_KEYS = ("id", "datasetId", "dataset_id", "identifier", "uuid", "slug")
_TITLE_AR_KEYS = ("titleAr", "title_ar", "nameAr", "title", "name")
_TITLE_EN_KEYS = ("titleEn", "title_en", "nameEn", "title", "name")
_DESC_KEYS = ("description", "descriptionAr", "description_ar", "notes", "summary")
_RESOURCE_KEYS = ("resources", "distributions", "files", "attachments", "dataFiles")
_URL_KEYS = ("url", "downloadUrl", "download_url", "accessUrl", "access_url", "link", "path")
_FORMAT_KEYS = ("format", "fileFormat", "file_format", "type", "extension", "mimeType")
_UPDATED_KEYS = (
    "lastModified", "last_modified", "updatedAt", "updated_at", "modified",
    "lastUpdated", "last_updated", "publishedDate", "issued",
)
_PUBLISHER_KEYS = ("publisher", "organization", "organisation", "agency", "entity", "provider")
_COVERAGE_KEYS = ("temporalCoverage", "temporal_coverage", "period", "coverage", "years")
_LICENSE_KEYS = ("license", "licence", "licenseName", "rights")


def _first(payload: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in payload and payload[key] not in (None, "", [], {}):
            return payload[key]
    return None


def _as_text(value: object) -> str:
    """Flatten the localised-string objects portals like to return."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("ar", "en", "value", "name", "label"):
            if key in value:
                return _as_text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(filter(None, (_as_text(v) for v in value)))
    return str(value)


def _as_date(value: object) -> date | None:
    text = _as_text(value)
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for parse in (
        lambda t: datetime.fromisoformat(t).date(),
        lambda t: datetime.strptime(t[:10], "%Y-%m-%d").date(),
        lambda t: datetime.strptime(t[:10], "%d/%m/%Y").date(),
    ):
        try:
            return parse(text)
        except (ValueError, TypeError):
            continue
    return None


def _normalise_format(value: object, url: str) -> str:
    text = _as_text(value).upper().strip(". ")
    if "/" in text:  # a media type such as "application/vnd...sheet"
        text = text.rsplit("/", 1)[-1]
    if text in {"", "OCTET-STREAM", "BINARY"}:
        tail = url.rsplit(".", 1)[-1].upper() if "." in url.rsplit("/", 1)[-1] else ""
        text = tail
    aliases = {
        "VND.OPENXMLFORMATS-OFFICEDOCUMENT.SPREADSHEETML.SHEET": "XLSX",
        "VND.MS-EXCEL": "XLS",
        "EXCEL": "XLSX",
        "SHEET": "XLSX",
        "COMMA-SEPARATED-VALUES": "CSV",
        "PLAIN": "CSV",
    }
    return aliases.get(text, text)


class SaudiOpenDataAdapter(SourceAdapter):
    """Searches the national open-data portal's dataset API."""

    def _search_url(self) -> str:
        path = self.descriptor.api.get("dataset_search", "/data/api/v1/datasets")
        return self.descriptor.url(path)

    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        if self.http is None:
            raise SourceError(self.id, "no HTTP client configured")

        query = " ".join(request.search_terms()[:4]).strip()
        params = {"q": query, "limit": limit} if query else {"limit": limit}
        fetched = self.http.get(self._search_url(), source_id=self.id, params=params)
        payload = fetched.json()

        rows = self._extract_rows(payload)
        return [c for c in (self._to_candidate(row) for row in rows[:limit]) if c]

    def _extract_rows(self, payload: object) -> list[dict]:
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        if not isinstance(payload, dict):
            raise SourceError(self.id, f"unexpected response type: {type(payload).__name__}")

        for key in _LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
            # Some portals nest one level deeper: {"data": {"datasets": [...]}}
            if isinstance(value, dict):
                for inner in _LIST_KEYS:
                    nested = value.get(inner)
                    if isinstance(nested, list):
                        return [r for r in nested if isinstance(r, dict)]

        raise SourceError(
            self.id,
            "could not find a dataset list in the response; top-level keys were "
            f"{sorted(payload)[:12]}. Update _LIST_KEYS in this adapter or the "
            "dataset_search path in sources.yaml.",
        )

    def _to_candidate(self, row: dict) -> DatasetCandidate | None:
        dataset_id = _as_text(_first(row, _ID_KEYS))
        title_ar = _as_text(_first(row, _TITLE_AR_KEYS))
        title_en = _as_text(_first(row, _TITLE_EN_KEYS))
        if not (dataset_id or title_ar or title_en):
            return None

        resources: list[Resource] = []
        raw_resources = _first(row, _RESOURCE_KEYS) or []
        if isinstance(raw_resources, dict):
            raw_resources = [raw_resources]
        for entry in raw_resources:
            if not isinstance(entry, dict):
                continue
            url = _as_text(_first(entry, _URL_KEYS))
            if not url:
                continue
            resources.append(
                Resource(
                    url=self.descriptor.url(url),
                    format=_normalise_format(_first(entry, _FORMAT_KEYS), url),
                    title=_as_text(_first(entry, _TITLE_AR_KEYS)),
                    byte_size=entry.get("size") if isinstance(entry.get("size"), int) else None,
                )
            )

        # Declared coverage is a claim; a year read out of the title is only a
        # hint. The two are kept apart because they carry different weight.
        declared = _as_text(_first(row, _COVERAGE_KEYS))
        if declared:
            coverage = Coverage(
                years=frozenset(extract_years(declared)),
                origin=CoverageOrigin.METADATA_CLAIM,
                note=f"التغطية المعلنة: {declared}",
            )
        else:
            hinted = extract_years(f"{title_ar} {title_en}")
            coverage = Coverage(
                years=frozenset(hinted),
                origin=CoverageOrigin.INFERRED_TITLE,
                note="مستنتجة من العنوان" if hinted else "غير معلنة",
            )

        landing = _as_text(_first(row, ("landingPage", "landing_page", "url", "link")))
        if not landing and dataset_id:
            detail = self.descriptor.api.get("dataset_detail", "/data/api/v1/datasets/{id}")
            landing = self.descriptor.url(detail.replace("{id}", dataset_id))

        publisher = _as_text(_first(row, _PUBLISHER_KEYS)) or self.descriptor.operator_ar

        return DatasetCandidate(
            source_id=self.id,
            dataset_id=dataset_id or title_en or title_ar,
            title_ar=title_ar or title_en,
            title_en=title_en,
            description=_as_text(_first(row, _DESC_KEYS)),
            landing_url=landing or self.descriptor.base_url,
            publisher_ar=publisher,
            publisher_en=_as_text(_first(row, _PUBLISHER_KEYS)) or self.descriptor.operator_en,
            resources=tuple(resources),
            claimed_coverage=coverage,
            last_updated=_as_date(_first(row, _UPDATED_KEYS)),
            license_name=_as_text(_first(row, _LICENSE_KEYS)) or None,
        )

    def probe(self) -> str:
        fetched = self.http.get(self._search_url(), source_id=self.id, params={"limit": 1})
        payload = fetched.json()
        rows = self._extract_rows(payload)
        sample = sorted(rows[0]) if rows else []
        return f"HTTP {fetched.status}, {len(rows)} row(s); first-row keys: {sample[:15]}"
