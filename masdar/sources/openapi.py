"""Reading GASTAT's OpenAPI specifications into dataset entries.

GASTAT's cdata API has no catalogue endpoint: what datasets exist is known
only from the developer portal, one OpenAPI 3.0 document per API product,
each with several `/v1/stats/{dataset_id}` paths. Transcribing those by hand
invites exactly the silent errors this project exists to avoid, so the
downloaded specification is read directly and becomes the configuration.

Everything a dataset needs is in the spec:

* the path, whose last segment is the dataset id;
* a bilingual summary, "العربية / English", which becomes the titles;
* the `dimensions[]` parameter, whose enum lists the breakdowns;
* `<DIM>_TIME` filter parameters, which mark the time dimension;
* the response schema, whose `*_OBSV` properties are the measures.

Two cautions shape the code. The portal may emit parameters inline or behind
`$ref`, and nothing verified which, so both are resolved. And a spec's
`servers` URL decides where requests -- and therefore any API key -- are
sent, so it is honoured only on the host the source is already configured
for.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import yaml

_ARABIC = re.compile(r"[؀-ۿ]")
_BILINGUAL_SPLIT = re.compile(r"\s+/\s+")

# Time dimensions, most preferred first. YEAR is what annual statistics use
# and what the verified datasets use.
_TIME_PREFERENCE = ("YEAR", "QUARTER", "MONTH", "PERIOD", "TIME", "DATE")

# Keys whose string value would be a credential if filled in.
_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|authorization|bearer|credential)",
    re.IGNORECASE,
)


class SpecError(ValueError):
    """A file that is not a usable OpenAPI document."""


# -- loading ----------------------------------------------------------------
def load_spec(path: Path) -> dict:
    """Parse a JSON or YAML OpenAPI document, or raise `SpecError`."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SpecError(f"cannot read {path}: {exc}") from exc
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SpecError(f"{path.name}: not valid JSON/YAML ({exc})") from exc

    if not isinstance(data, dict):
        raise SpecError(f"{path.name}: top level is not an object")
    if not (data.get("openapi") or data.get("swagger")):
        raise SpecError(f"{path.name}: missing the 'openapi' version field")
    if not isinstance(data.get("paths"), dict) or not data["paths"]:
        raise SpecError(f"{path.name}: has no paths")
    return data


def find_secrets(node: object, trail: str = "") -> list[str]:
    """Locations of filled-in credential-looking values.

    A downloaded spec should describe *where* a key goes (a security scheme
    naming the `apikey` header), never contain one. Some portals embed the
    caller's key in examples; such a file must not be committed.
    """
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{trail}/{key}"
            if (
                isinstance(value, str)
                and _SENSITIVE_KEY.search(str(key))
                and len(value) >= 16
                and " " not in value
            ):
                found.append(here)
            else:
                found.extend(find_secrets(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(find_secrets(value, f"{trail}[{index}]"))
    return found


# -- helpers ----------------------------------------------------------------
def _resolve(spec: dict, node: object, depth: int = 0) -> dict:
    """Follow local `$ref`s ("#/components/...") to the node they name."""
    while isinstance(node, dict) and "$ref" in node and depth < 32:
        ref = str(node["$ref"])
        if not ref.startswith("#/"):
            return {}  # external references are not followed
        target: object = spec
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            target = target.get(part, {}) if isinstance(target, dict) else {}
        node = target
        depth += 1
    return node if isinstance(node, dict) else {}


def split_bilingual(text: str | None) -> tuple[str, str]:
    """Split "العربية / English" into its two halves by script."""
    if not text:
        return "", ""
    parts = [p.strip() for p in _BILINGUAL_SPLIT.split(str(text)) if p.strip()]
    arabic = [p for p in parts if _ARABIC.search(p)]
    latin = [p for p in parts if not _ARABIC.search(p)]
    return " / ".join(arabic), " / ".join(latin)


def _dataset_id(path: str) -> str | None:
    segments = [s for s in path.split("/") if s]
    if not segments or "{" in segments[-1]:
        return None
    return segments[-1]


def _enum_of(spec: dict, schema: object) -> list[str]:
    schema = _resolve(spec, schema)
    if schema.get("type") == "array" or "items" in schema:
        schema = _resolve(spec, schema.get("items", {}))
    return [str(v) for v in schema.get("enum", []) if v is not None]


def _measures(spec: dict, operation: dict) -> list[str]:
    responses = operation.get("responses") or {}
    ok = _resolve(spec, responses.get("200") or responses.get(200) or {})
    content = ok.get("content") or {}
    media = content.get("application/json") or next(iter(content.values()), {})
    schema = _resolve(spec, (media or {}).get("schema", {}))
    records = _resolve(spec, (schema.get("properties") or {}).get("value", {}))
    item = _resolve(spec, records.get("items", {}))
    return sorted(k for k in (item.get("properties") or {}) if _is_measure(k))


# Measure columns seen in the live specs: *_OBSV (energy) and TOTAL_OBS_VALUE
# (health). Both are matched rather than assuming one convention.
_MEASURE = re.compile(r"(_OBSV|OBS_VALUE|OBSVALUE)$", re.IGNORECASE)


def _is_measure(name: str) -> bool:
    return bool(_MEASURE.search(name))


def _same_host(url: str, base_url: str) -> bool:
    return bool(url) and urlparse(url).netloc.lower() == urlparse(base_url).netloc.lower()


# -- extraction ---------------------------------------------------------------
def datasets_from_spec(spec: dict, base_url: str, source_name: str = "") -> list[dict]:
    """One dataset entry per GET path, in the shape `sources.yaml` uses."""
    from masdar.nlu.parser import topics_in_text

    info = spec.get("info") or {}
    api_ar, api_en = split_bilingual(info.get("title"))

    # A server on another host is ignored: requests carry the API key to
    # wherever this points.
    server = ""
    for entry in spec.get("servers") or []:
        url = str((entry or {}).get("url") or "").rstrip("/")
        if _same_host(url, base_url):
            server = url
            break

    datasets: list[dict] = []
    for path, item in (spec.get("paths") or {}).items():
        operation = _resolve(spec, (item or {}).get("get"))
        if not operation:
            continue
        dataset_id = _dataset_id(str(path))
        if not dataset_id:
            continue

        parameters = [
            _resolve(spec, p)
            for p in list((item or {}).get("parameters") or [])
            + list(operation.get("parameters") or [])
        ]
        names = [str(p.get("name", "")) for p in parameters]

        dimensions: list[str] = []
        for param in parameters:
            if str(param.get("name")) in ("dimensions[]", "dimensions"):
                dimensions = _enum_of(spec, param.get("schema", {}))
                break
        # Fall back on the filter parameters when no enum is declared.
        if not dimensions:
            for name in names:
                upper = name.upper()
                for suffix in ("_CODE", "_TIME"):
                    if upper.endswith(suffix):
                        stem = name[: -len(suffix)]
                        if stem and stem not in dimensions:
                            dimensions.append(stem)

        timed = [n[: -len("_TIME")] for n in names if n.upper().endswith("_TIME")]
        time_dimension = next(
            (t for t in _TIME_PREFERENCE if t in dimensions),
            next((t for t in timed if t in dimensions), ""),
        )

        title_ar, title_en = split_bilingual(operation.get("summary"))
        desc_ar, desc_en = split_bilingual(operation.get("description"))
        tag = " ".join(str(t) for t in operation.get("tags") or [])
        tag_ar, tag_en = split_bilingual(tag)

        text = " ".join(filter(None, (api_ar, api_en, tag_ar, tag_en, title_ar, title_en)))
        entry = {
            "id": dataset_id,
            "path": str(path),
            "title_ar": title_ar or title_en or dataset_id,
            "title_en": title_en,
            "description": " — ".join(filter(None, (desc_ar, desc_en))),
            "keywords": " ".join(filter(None, (api_ar, api_en, tag_ar, tag_en))),
            "api_ar": api_ar,
            "api_en": api_en,
            "dimensions": dimensions,
            "time_dimension": time_dimension,
            "measures": _measures(spec, operation),
            "topics": topics_in_text(text),
            "origin": f"spec:{source_name}" if source_name else "spec",
        }
        if server:
            entry["server"] = server
        datasets.append(entry)
    return datasets


@functools.lru_cache(maxsize=64)
def _cached(path_str: str, mtime_ns: int, base_url: str) -> tuple[tuple[dict, ...], str]:
    """Parse a spec once per file version; adapters are rebuilt per request."""
    path = Path(path_str)
    try:
        spec = load_spec(path)
    except SpecError as exc:
        return (), str(exc)
    return tuple(datasets_from_spec(spec, base_url, path.stem)), ""


def datasets_from_dir(directory: Path, base_url: str) -> tuple[list[dict], list[str]]:
    """All datasets from every spec in a directory, plus per-file problems."""
    datasets: list[dict] = []
    problems: list[str] = []
    if not directory.is_dir():
        return datasets, problems
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".json", ".yaml", ".yml"):
            continue
        entries, problem = _cached(str(path), path.stat().st_mtime_ns, base_url)
        if problem:
            problems.append(problem)
        datasets.extend(dict(e) for e in entries)
    return datasets, problems
