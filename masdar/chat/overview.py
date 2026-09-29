"""What the agent can search, for the chat page's landing screen.

Counted from the configuration and the local catalogues at start-up, never
from a network call, so the page opens instantly and the numbers are the
ones the agent actually works with.
"""

from __future__ import annotations

from masdar.nlu.lexicon import Lexicon
from masdar.sources.registry import Registry, reachable_here

# Questions shown on the landing screen and run by `masdar warmup`. Each one
# shows a different part of what the agent does; `shows` says which, for the
# presenter.
SHOWCASE: tuple[dict[str, str], ...] = (
    {"q": "احصاءات الطاقة الكهربائية لسنة 2026 حسب المناطق",
     "shows": "سنة لم تُنشر بعد: يقول ذلك صراحة ويقترح أحدث سنة متاحة"},
    {"q": "استهلاك الكهرباء 2022 حسب المناطق",
     "shows": "ملف إكسل جاهز مفصّل حسب المناطق، بمصدره وتاريخ تحديثه"},
    {"q": "أحدث بيانات متوسط العمر المتوقع",
     "shows": "«الأحدث»: يبحث عن آخر سنة منشورة في كل المصادر"},
    {"q": "معدل البطالة 2023",
     "shows": "مؤشر اقتصادي من الهيئة العامة للإحصاء"},
    {"q": "مؤشرات سوق العمل من وزارة الموارد البشرية",
     "shows": "بيانات لحظية من الوزارة: العمالة، السعوديون وغير السعوديين"},
    {"q": "استهلاك الكهرباء للفرد 2023",
     "shows": "مصدر دولي (البنك الدولي) يأتي بعد الجهات السعودية ويُذكر أنه دولي"},
    {"q": "عدد الحجاج 2023",
     "shows": "سؤال قصير يكفي: الموضوع والسنة، والباقي على الوكيل"},
    {"q": "الناتج المحلي الإجمالي من 2019 إلى 2022",
     "shows": "نطاق سنوات: يتحقق من كل سنة في الملف نفسه"},
)

# How each adapter's reach is counted, and what the count is called.
_UNITS = {
    "gastat_cdata": "مجموعة بيانات",
    "saudi_open_data": "جهة حكومية",
    "live_data": "مؤشراً لحظياً",
    "worldbank": "مؤشراً",
}


def _reach(registry: Registry, descriptor) -> int | None:
    try:
        adapter = registry.adapter(descriptor.id)
        if descriptor.adapter == "gastat_cdata":
            return len(adapter._datasets())
        if descriptor.adapter == "saudi_open_data":
            return len(adapter._organizations())
        if descriptor.adapter in ("live_data", "worldbank"):
            return len(adapter._indicators())
    except Exception:  # an overview must never stop the page from opening
        return None
    return None


def _portal_publishers(registry: Registry) -> set[str]:
    names: set[str] = set()
    for d in registry.descriptors:
        if d.adapter == "saudi_open_data" and d.enabled and reachable_here(d):
            try:
                orgs = registry.adapter(d.id)._organizations()
            except Exception:
                continue
            names.update(str(o.get("name_ar") or o.get("id")) for o in orgs)
    return names


def _portal_datasets(registry: Registry) -> int:
    """Datasets the configured publishers list on the national platform."""
    total = 0
    for descriptor in registry.descriptors:
        if descriptor.adapter != "saudi_open_data" or not descriptor.enabled:
            continue
        if not reachable_here(descriptor):
            continue
        for org in descriptor.api.get("organizations") or ():
            count = org.get("datasets") if isinstance(org, dict) else None
            if isinstance(count, int):
                total += count
    return total


def overview(
    registry: Registry, lexicon: Lexicon, llm_status: str = "", llm_state: str = "off"
) -> dict:
    sources, elsewhere = [], []
    for d in registry.descriptors:
        if not d.enabled or d.synthetic or not d.api_verified:
            continue
        if not reachable_here(d):
            # Listed, not counted: the page says plainly what this server
            # cannot see, instead of every answer saying it again.
            elsewhere.append({"id": d.id, "name": d.name_ar,
                              "reason": "لا تُفتح إلا من خادم داخل المملكة"})
            continue
        count = _reach(registry, d)
        kind = "دولي" if d.international else "لحظي" if d.adapter == "live_data" else "رسمي"
        sources.append({
            "id": d.id,
            "name": d.name_ar,
            "operator": d.operator_ar or d.name_ar,
            "kind": kind,
            "count": count,
            "unit": _UNITS.get(d.adapter, ""),
            "checked": d.access_checked,
        })
    # Saudi first, then live, then international; the national statistics
    # office leads.
    order = {"رسمي": 0, "لحظي": 1, "دولي": 2}
    sources.sort(key=lambda s: (order[s["kind"]], s["operator"] != "الهيئة العامة للإحصاء"))

    publishers = {s["operator"] for s in sources if s["unit"] != "جهة حكومية"}
    publishers.update(_portal_publishers(registry))
    items = _portal_datasets(registry) + sum(
        s["count"] or 0 for s in sources if s["unit"] != "جهة حكومية"
    )
    return {
        "stats": {
            "sources": len(sources),
            "publishers": len(publishers),
            "datasets": items,
            "topics": len(lexicon.topics),
        },
        "sources": sources,
        "unavailable": elsewhere,
        "topics": [t.label_ar for t in lexicon.topics],
        "examples": [dict(e) for e in SHOWCASE],
        "llm": llm_status,
        # "on", "off", or "error" (a key is set but does not work).
        "llm_state": llm_state,
        "demo": any(d.synthetic for d in registry.descriptors),
    }
