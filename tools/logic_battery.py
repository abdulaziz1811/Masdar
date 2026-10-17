"""Logic battery: how each question is read, and what ranks first.

    python tools/logic_battery.py              # the questions below
    python tools/logic_battery.py "سؤال" ...    # your own

Offline, so it runs anywhere in seconds: only the local catalogues answer
(the cdata specs, the World Bank and GCC-Stat indicator lists, the live
indicators' configuration). Live sources (GASTAT's site, the national
platform) are not asked. For each question it prints the reading (topic,
years, breakdowns, the words left over) and the three results ranked first,
flagged when one does not name the subject (resolve.names_subject) or answers
a count with a rate. Read it after any change to the lexicon or the ranker:
the tests pin what is known to be right, this shows what a change moved.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

# Run from a checkout without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MASDAR_OUTSIDE_KSA", "1")

from masdar.nlu.lexicon import load_lexicon  # noqa: E402
from masdar.nlu.parser import parse  # noqa: E402
from masdar.pipeline import resolve  # noqa: E402
from masdar.sources.http import HttpClient  # noqa: E402
from masdar.sources.registry import load_registry  # noqa: E402

QUESTIONS = """
احصاءات الطاقة الكهربائية لسنة 2026 حسب المناطق
استهلاك الكهرباء 2022 حسب المناطق
أحدث بيانات متوسط العمر المتوقع
معدل البطالة 2023
مؤشرات سوق العمل من وزارة الموارد البشرية
استهلاك الكهرباء للفرد 2023
عدد الحجاج 2023
الناتج المحلي الإجمالي من 2019 إلى 2022
عدد السكان 2024
عدد سكان الرياض 2022
عدد السكان حسب الجنسية 2022
عدد المواليد 2022
عدد الوفيات 2021
معدل الخصوبة الكلي
عدد المستشفيات حسب المناطق
عدد الأطباء 2023
عدد الطلاب في الجامعات 2023
عدد المدارس حسب المنطقة
معدل التضخم 2025
الرقم القياسي لأسعار المستهلك أغسطس 2026
إجمالي الصادرات السلعية 2024
الصادرات غير النفطية 2024
الواردات حسب الدولة 2023
عدد السياح 2024
إنفاق السياح 2023
إنتاج النفط الخام 2024
إنتاج الكهرباء 2024
كمية المياه المحلاة 2024
استهلاك المياه حسب المنطقة 2022
إنتاج التمور 2024
عدد الإبل 2024
عدد المشتغلين 2025
متوسط الأجر الشهري حسب النشاط الاقتصادي
عدد السعوديين في القطاع الخاص
معدل المشاركة في القوى العاملة للإناث
عدد المنشآت حسب النشاط الاقتصادي
عدد المساكن حسب نوع الحيازة
نسبة تملك المساكن للأسر السعودية
عدد المركبات 2023
عدد الحوادث المرورية 2023
مستخدمي الإنترنت 2024
عدد الزوار للمتاحف
حالات الطلاق 2022
عدد عقود الزواج 2021
""".strip().splitlines()



def main(questions: list[str]) -> int:
    http = HttpClient(offline=True, use_cache=False, retries=1)
    registry = load_registry(http=http)
    lexicon = load_lexicon()
    descriptors = {d.id: d for d in registry.descriptors}
    today = date.today()
    flags = 0
    for question in questions:
        flags += _one(question, registry, lexicon, descriptors, today)
    print(f"\nflags: {flags}")
    return 0


def _one(question, registry, lexicon, descriptors, today) -> int:
    """Print one question's reading and first results; 1 if it is flagged."""
    request = parse(question, lexicon)
    topic_id = request.topic.id if request.topic else None
    preferred = request.topic.preferred_sources if request.topic else ()
    plan = [d for d in registry.plan(topic_id, preferred) if not d.synthetic][:9]
    found = []
    for descriptor in plan:
        try:
            found.extend(registry.adapter(descriptor.id).search(request, limit=8))
        except Exception:  # offline: sources that need the network fail here
            continue
    ranked = resolve.rank(request, found, descriptors, lexicon, today)
    kept = [c for c in ranked if resolve.on_subject(request, c)]
    # As the answer orders them: named for the subject first (see
    # orchestrator._finding_sort_key), then by score.
    kept.sort(key=lambda c: (not resolve.names_subject(request, c), -c.score))

    dims = ",".join(d.value for d in request.dimensions) or "-"
    print(f"Q: {question}\n   topic={topic_id or '-'} years={list(request.period.years)} "
          f"dims={dims} free={list(request.free_terms)} typed={list(request.typed_phrases)}")
    flagged = request.topic is None
    if flagged:
        print("   ⚑ no topic")
    for candidate in kept[:3]:
        marks = ""
        if not resolve.names_subject(request, candidate):
            marks += "  ⚑ not named"
        if resolve.measure_mismatch(request, candidate):
            marks += "  ⚑ rate-for-count"
        print(f"   {candidate.score:6.2f} {candidate.source_id:13s} "
              f"{candidate.title_ar[:70]}{marks}")
    if kept and not resolve.names_subject(request, kept[0]):
        flagged = True
    if not kept:
        print("   (nothing on subject offline)")
    return int(flagged)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or QUESTIONS))
