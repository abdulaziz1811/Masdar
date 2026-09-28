"""Find the national platform's publishers, and what each one publishes about.

Only a few publishers are hand-configured, because the platform's documented
API has no call that lists them and its listing endpoint does not answer
from outside the Kingdom. This module does the listing from the user's own
machine, where it does answer, and writes a file the portal adapter reads:

* **Which publishers.** Either the platform's own list (`/api/publishers`,
  the call its publishers page makes -- the item fields used here,
  `publisherID`, `nameAr`, `nameEn`, `numberOfDatasets`, are the ones that
  page's code reads), or a list of exact Arabic names resolved one by one
  through the documented `organizations` call.
* **What about.** Each new publisher's catalogue is fetched through the
  documented call and its dataset titles are labelled with the lexicon's
  topics. A topic held by enough of the titles becomes one the publisher is
  asked about. None of this is guessed from the name.

The output is a reviewable YAML file, not a silent change: publishers
already in `sources.yaml` are left out, and the file says how each entry
was found.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from masdar.nlu.lexicon import Lexicon, load_lexicon
from masdar.nlu.parser import topics_in_text
from masdar.sources.base import SourceError, SourceUnreachable

LIST_PATH = "/api/publishers"
PAGE_SIZE = 100
MAX_PAGES = 50
# A topic held by at least this share of a publisher's titles (and by at
# least MIN_TITLES of them) is one the publisher is asked about.
MIN_SHARE = 0.15
MIN_TITLES = 2
MAX_TOPICS = 4
# Never matches a topic id: a publisher whose titles fit no topic is still
# searched for questions that name no topic, and for nothing else.
UNCLASSIFIED = "unclassified"
DISCOVERED_FILE = Path(__file__).resolve().parent.parent / "config" / "publishers_discovered.yaml"


@dataclass
class Publisher:
    id: str = ""
    name_ar: str = ""
    name_en: str = ""
    datasets: int | None = None
    topics: tuple[str, ...] = ()
    found_by: str = ""
    problems: list[str] = field(default_factory=list)


def _items(payload: object) -> list[dict]:
    """The records of one page, whatever the envelope."""
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        for key in ("content", "data", "items", "results", "publishers"):
            value = payload.get(key)
            if isinstance(value, list):
                return [p for p in value if isinstance(p, dict)]
            if isinstance(value, dict):
                nested = _items(value)
                if nested:
                    return nested
    return []


def _is_last(payload: object, page: int) -> bool:
    """Only the server's own paging metadata says a page is the last.

    A short page is not proof: a server may cap the page size below what was
    asked, and stopping there would silently drop publishers.
    """
    if isinstance(payload, dict):
        if payload.get("last") is True:
            return True
        total_pages = payload.get("totalPages")
        if isinstance(total_pages, int) and page + 1 >= total_pages:
            return True
    return False


def list_publishers(adapter) -> list[Publisher]:
    """Every publisher the platform lists, page by page."""
    publishers: dict[str, Publisher] = {}
    for page in range(MAX_PAGES):
        url = adapter.descriptor.url(
            f"{LIST_PATH}?page={page}&size={PAGE_SIZE}&sort=nameAr,ASC"
        )
        fetched = adapter.http.get(url, source_id=adapter.id, headers={"Accept-Language": "ar"})
        payload = fetched.json()
        items = _items(payload)
        if not items:
            if page == 0:
                keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
                raise SourceError(adapter.id, f"شكل غير متوقع لقائمة الجهات: {keys}")
            break
        before = len(publishers)
        for item in items:
            pid = str(item.get("publisherID") or item.get("publisherId") or item.get("id") or "")
            if not pid or pid in publishers:
                continue
            count = item.get("numberOfDatasets")
            publishers[pid] = Publisher(
                id=pid,
                name_ar=str(item.get("nameAr") or "").strip(),
                name_en=str(item.get("nameEn") or "").strip(),
                datasets=count if isinstance(count, int) else None,
                found_by="قائمة المنصة (/api/publishers)",
            )
        # A page that adds no one means the server ignores paging and returns
        # the same list each time.
        if _is_last(payload, page) or len(publishers) == before:
            break
    return list(publishers.values())


def _catalogue(adapter, key: str) -> dict:
    return adapter._get_json(adapter._api(
        "organization", "/data/api/organizations?version=-1&organization={id}", id=key,
    ))


def resolve_names(adapter, names: list[str]) -> list[Publisher]:
    """Exact Arabic names, looked up through the documented call."""
    def one(name: str) -> Publisher:
        try:
            payload = _catalogue(adapter, name)
        except (SourceError, SourceUnreachable) as exc:
            return Publisher(name_ar=name, problems=[exc.reason])
        return Publisher(
            id=str(payload.get("id") or ""),
            name_ar=str(payload.get("nameAr") or name).strip(),
            name_en=str(payload.get("nameEn") or "").strip(),
            found_by="الاسم عبر الواجهة الموثّقة",
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one, [n.strip() for n in names if n.strip()]))


def infer_topics(titles: list[str], lexicon: Lexicon) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for title in titles:
        for topic in topics_in_text(title, lexicon)[:2]:
            counts[topic] += 1
    floor = max(MIN_TITLES, MIN_SHARE * len(titles))
    kept = [topic for topic, n in counts.most_common() if n >= floor]
    return tuple(kept[:MAX_TOPICS])


def classify(adapter, publishers: list[Publisher], lexicon: Lexicon | None = None) -> None:
    """Fill in each publisher's dataset count and topics from its catalogue."""
    lexicon = lexicon or load_lexicon()

    def one(publisher: Publisher) -> None:
        key = publisher.id or publisher.name_ar
        if not key or publisher.problems:
            return
        try:
            payload = _catalogue(adapter, key)
        except (SourceError, SourceUnreachable) as exc:
            publisher.problems.append(exc.reason)
            return
        datasets = [d for d in payload.get("datasets") or [] if isinstance(d, dict)]
        publisher.datasets = len(datasets)
        titles = [f"{d.get('titleAr') or ''} {d.get('titleEn') or ''}" for d in datasets]
        publisher.topics = infer_topics(titles, lexicon) or (UNCLASSIFIED,)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(one, publishers))


def configured_keys(adapter) -> set[str]:
    orgs = adapter.descriptor.api.get("organizations") or []
    keys: set[str] = set()
    for org in orgs:
        if isinstance(org, dict):
            keys.update(str(v) for v in (org.get("id"), org.get("name_ar")) if v)
    return keys


def to_yaml(publishers: list[Publisher], today: date | None = None) -> str:
    today = today or date.today()
    entries = [
        {
            k: v for k, v in {
                "id": p.id or None,
                "name_ar": p.name_ar,
                "datasets": p.datasets,
                "topics": list(p.topics),
                "found_by": p.found_by,
            }.items() if v not in (None, "", [])
        }
        for p in publishers if not p.problems and p.topics
    ]
    header = (
        f"# Publishers found by `masdar publishers` on {today.isoformat()}.\n"
        "# Read by the portal adapter alongside sources.yaml; entries there win.\n"
        "# Topics were inferred from each publisher's own dataset titles.\n"
        f"# `{UNCLASSIFIED}`: no topic fitted; searched only for questions without one.\n"
    )
    return header + yaml.safe_dump(
        {"organizations": entries}, allow_unicode=True, sort_keys=False, width=100
    )


def load_discovered(path: Path = DISCOVERED_FILE) -> list[dict]:
    """Entries from a discovery run, or none."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    orgs = data.get("organizations") if isinstance(data, dict) else None
    return [o for o in orgs or [] if isinstance(o, dict) and (o.get("id") or o.get("name_ar"))]
