"""Shared test doubles.

`paged` makes a recorded response behave like the live API with respect to
`$top`, `$skip` and `$orderby`, so the adapter's paging is exercised against
server semantics rather than against a fixture that ignores them.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit


def paged(body: bytes, url: str, server_cap: int | None = None) -> bytes:
    """Slice an OData `{"value": [...]}` body the way the server would.

    `server_cap` imitates a server that silently returns fewer rows than
    `$top` asked for, which paging must survive without losing rows.
    """
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    payload = json.loads(body.decode("utf-8"))
    rows = list(payload.get("value") or [])

    order = query.get("$orderby", "").split()
    if order:
        column = order[0]
        descending = len(order) > 1 and order[1].upper() == "DESC"
        rows.sort(key=lambda r: str(r.get(column, "")), reverse=descending)

    skip = int(query.get("$skip", "0") or 0)
    top = int(query["$top"]) if query.get("$top") else len(rows)
    if server_cap is not None:
        top = min(top, server_cap)
    return json.dumps({"value": rows[skip : skip + top]}, ensure_ascii=False).encode("utf-8")


_LABEL_SUFFIXES = ("_ARAB", "_ENGL", "_CODE", "_TIME")
_MEASURE_SUFFIXES = ("_OBSV", "OBS_VALUE", "OBSVALUE")


def aggregate(rows: list[dict], dimensions: list[str]) -> list[dict]:
    """Group rows by the requested dimensions and SUM every measure.

    This is what api.stats.gov.sa does with a dimension a request leaves out,
    verified on 2026-09-28 (life expectancy by YEAR alone returns the sum of
    the female, total and male rows). A fake that returns the full table
    whatever is asked cannot catch a regression to partial requests.
    """
    wanted = set(dimensions)
    groups: dict[tuple, dict] = {}
    for row in rows:
        kept = {
            k: v for k, v in row.items()
            if k.upper().endswith(_LABEL_SUFFIXES)
            and next((k[: -len(s)] for s in _LABEL_SUFFIXES if k.upper().endswith(s)), "") in wanted
        }
        key = tuple(sorted(kept.items(), key=lambda kv: kv[0]))
        group = groups.setdefault(key, dict(kept))
        for column, value in row.items():
            if column.upper().endswith(_MEASURE_SUFFIXES) and isinstance(value, (int, float)):
                group[column] = group.get(column, 0) + value
    return list(groups.values())


FIXTURE_SPECS = Path(__file__).resolve().parent / "fixtures" / "cdata_specs"


def cdata_descriptor(spec_dir: Path | None = None):
    """The configured gastat_cdata source, reading specs from `spec_dir`.

    Tests must not read the real specs directory: every imported file would
    change what they find. By default they see the hand-declared datasets and
    the health spec kept under tests/fixtures/cdata_specs.
    """
    from dataclasses import replace

    from masdar.sources.registry import load_descriptors

    base = next(d for d in load_descriptors() if d.id == "gastat_cdata")
    directory = spec_dir if spec_dir is not None else FIXTURE_SPECS
    return replace(base, api={**base.api, "spec_dir": str(directory)})
