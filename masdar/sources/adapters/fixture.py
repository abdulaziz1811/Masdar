"""Adapter that serves recorded datasets from disk.

Two uses:

* Tests exercise the whole pipeline -- search, verification, export -- with
  no network at all.
* `--demo` lets someone see the agent work end to end before any live source
  is reachable.

Fixture sources are marked `synthetic: true` in their descriptor, which the
exporter turns into a visible warning inside the workbook. Sample data must
never be able to pass for an official figure.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from masdar.domain.models import (
    Coverage,
    CoverageOrigin,
    DataRequest,
    DatasetCandidate,
    Resource,
)
from masdar.nlu.normalize import contains_phrase
from masdar.sources.base import SourceAdapter, SourceError

FIXTURE_ROOT = Path(__file__).resolve().parent.parent.parent / "fixtures"


def _as_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


class FixtureAdapter(SourceAdapter):
    """Reads a manifest.json describing datasets and their local files."""

    def _dir(self) -> Path:
        configured = self.descriptor.api.get("fixture_dir")
        base = Path(configured) if configured else FIXTURE_ROOT / self.id
        if not base.is_absolute():
            base = FIXTURE_ROOT / base
        return base

    def _manifest(self) -> dict:
        path = self._dir() / "manifest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise SourceError(self.id, f"no fixture manifest at {path}") from None
        except json.JSONDecodeError as exc:
            raise SourceError(self.id, f"invalid fixture manifest {path}: {exc}") from exc

    def search(self, request: DataRequest, limit: int = 10) -> list[DatasetCandidate]:
        manifest = self._manifest()
        terms = request.search_terms()
        topic_id = request.topic.id if request.topic else None

        scored: list[tuple[float, DatasetCandidate]] = []
        for entry in manifest.get("datasets", []):
            score = 0.0
            if topic_id and topic_id in entry.get("topics", []):
                score += 3.0
            haystack = " ".join(
                str(entry.get(key, "")) for key in ("title_ar", "title_en", "description")
            )
            score += sum(1 for term in terms if term and contains_phrase(haystack, term))
            if score <= 0:
                continue

            resources = tuple(
                Resource(
                    url=self._resource_url(res["path"]),
                    format=(
                        str(res.get("format", "")).upper()
                        or Path(res["path"]).suffix.upper().lstrip(".")
                    ),
                    title=res.get("title", ""),
                )
                for res in entry.get("resources", [])
            )

            declared = entry.get("claimed_years")
            coverage = (
                Coverage(
                    years=frozenset(int(y) for y in declared),
                    origin=CoverageOrigin.METADATA_CLAIM,
                    is_exhaustive=bool(entry.get("claimed_years_exhaustive", False)),
                    note="التغطية المعلنة في بيانات المصدر الوصفية",
                )
                if declared
                else Coverage.unknown()
            )

            scored.append((
                score,
                DatasetCandidate(
                    source_id=self.id,
                    dataset_id=entry.get("dataset_id", entry.get("title_ar", ""))[:120],
                    title_ar=entry.get("title_ar", ""),
                    title_en=entry.get("title_en", ""),
                    description=entry.get("description", ""),
                    landing_url=entry.get("landing_url", self.descriptor.base_url),
                    publisher_ar=entry.get("publisher_ar", self.descriptor.name_ar),
                    publisher_en=entry.get("publisher_en", self.descriptor.name_en),
                    resources=resources,
                    claimed_coverage=coverage,
                    last_updated=_as_date(entry.get("last_updated")),
                    license_name=entry.get("license"),
                ),
            ))

        scored.sort(key=lambda pair: -pair[0])
        return [candidate for _, candidate in scored[:limit]]

    def _resource_url(self, relative: str) -> str:
        return (self._dir() / relative).resolve().as_uri()

    def probe(self) -> str:
        manifest = self._manifest()
        return f"fixture ok, {len(manifest.get('datasets', []))} dataset(s) in {self._dir()}"
