"""Turning an `Answer` into the sentence the user reads.

Wording is derived from the verdict, never chosen independently of it. In
particular the negative cases are kept distinct: "this year is not
published" and "I could not reach the sources" are different facts, and
collapsing them into one apology would quietly mislead.
"""

from __future__ import annotations

from masdar.domain.models import Answer, Finding, Verdict

_ORIGIN_LABELS = {
    "observed_data": "تم فتح الملف والتحقق من السنوات الموجودة فيه",
    "metadata_claim": "البيانات الوصفية المعلنة من المصدر",
    "inferred_title": "استنتاج من العنوان (غير مؤكد)",
}


def _years(values: tuple[int, ...]) -> str:
    return "، ".join(str(v) for v in values)



def _finding_block(finding: Finding, index: int | None = None) -> list[str]:
    candidate = finding.candidate
    provenance = finding.provenance
    prefix = f"{index}. " if index else ""
    lines = [f"{prefix}📊 {candidate.title_ar or candidate.title_en}"]
    lines.append(f"   🏛 الجهة الناشرة: {provenance.publisher_ar or '—'}")
    lines.append(
        "   📅 آخر تحديث معلن: "
        + (provenance.last_updated.isoformat() if provenance.last_updated else "غير معلن")
    )
    if provenance.last_updated_note:
        lines.append(f"      ({provenance.last_updated_note})")
    if finding.matched_years:
        lines.append(f"   🗓 السنوات المتاحة المطابقة: {_years(finding.matched_years)}")
    if finding.coverage.years:
        lines.append(f"   📚 تغطية المصدر: {finding.coverage.describe()}")
    lines.append(f"   🔗 صفحة المصدر: {provenance.landing_url or '—'}")
    if provenance.resource_url:
        lines.append(f"   📎 رابط الملف: {provenance.resource_url}")
    lines.append(
        "   🔍 أساس التحقق: "
        + _ORIGIN_LABELS.get(finding.coverage.origin.value, finding.coverage.origin.value)
    )
    if provenance.sha256:
        lines.append(f"   🔐 بصمة الملف: {provenance.sha256[:16]}…")
    if finding.export_path:
        lines.append(f"   📁 ملف إكسل جاهز: {finding.export_path}")
    return lines


def _notes_block(finding: Finding) -> list[str]:
    if not finding.notes:
        return []
    return ["", "ملاحظات:"] + [f"   • {note}" for note in finding.notes]


def _suggestions_block(answer: Answer) -> list[str]:
    if not answer.suggestions:
        return []
    lines = ["", "🔄 البديل المتاح:"]
    for suggestion in answer.suggestions:
        line = f"   • {suggestion.year} — {suggestion.reason_ar}"
        if suggestion.source_ar or suggestion.title_ar:
            line += f": {suggestion.source_ar}" + (
                f" («{suggestion.title_ar}»)" if suggestion.title_ar else ""
            )
        lines.append(line)
    lines.append("   اطلب أي سنة منها وسأجهّز الملف.")
    return lines


def _errors_block(answer: Answer) -> list[str]:
    if not answer.source_errors:
        return []
    lines = ["", "⚠️ مصادر لم تُجب في هذه المحاولة:"]
    for source_id, reason in answer.source_errors:
        lines.append(f"   • {source_id}: {reason}")
    return lines


def compose_message(answer: Answer) -> str:
    request = answer.request
    topic = request.topic.label_ar if request.topic else request.raw_query
    period = request.period.label()
    best = answer.primary
    lines: list[str] = []

    if answer.verdict is Verdict.NO_SOURCE and not request.is_answerable:
        lines = [
            "❓ ما وصلني موضوع واضح أبحث عنه.",
            "",
            "اكتب الطلب بالشكل التالي: الموضوع + السنة + التفصيل المطلوب.",
            "مثال: إحصاءات الطاقة الكهربائية لسنة 2024 حسب المناطق.",
        ]
        return "\n".join(lines)

    if answer.verdict is Verdict.AVAILABLE and best is not None:
        lines.append(f"✅ لقيت البيانات المطلوبة: {topic} — {period}")
        lines.append("")
        lines.extend(_finding_block(best))
        lines.extend(_notes_block(best))

    elif answer.verdict is Verdict.PARTIAL and best is not None:
        lines.append(f"⚠️ البيانات متوفرة جزئياً: {topic}")
        lines.append(f"   المتاح: {_years(best.matched_years)}")
        lines.append(f"   غير متاح: {_years(best.missing_years)}")
        lines.append("")
        lines.extend(_finding_block(best))
        lines.extend(_notes_block(best))

    elif answer.verdict is Verdict.NOT_AVAILABLE and best is not None:
        lines.append(f"❌ ما لقيت بيانات «{topic}» للفترة {period}.")
        lines.append("")
        lines.append("بحثت ووجدت مجموعة البيانات، لكن السنة المطلوبة غير موجودة فيها:")
        lines.extend(_finding_block(best))
        lines.extend(_notes_block(best))

    elif answer.verdict is Verdict.UNVERIFIED and best is not None:
        lines.append(f"🔶 لقيت نتائج محتملة عن «{topic}»، لكن ما أقدر أأكد توفر {period}.")
        lines.append("   ما أعطيك تأكيداً بدون دليل من الملف نفسه.")
        lines.append("")
        lines.extend(_finding_block(best))
        lines.extend(_notes_block(best))

    elif answer.verdict is Verdict.SOURCE_UNREACHABLE:
        lines.append("🚫 تعذّر الوصول إلى المصادر الرسمية في هذه المحاولة.")
        lines.append("")
        lines.append("مهم: هذا لا يعني أن البيانات غير موجودة — يعني أن الاتصال")
        lines.append("بالمصادر فشل، ولا أستطيع الحكم على التوفر بدون الوصول إليها.")

    else:
        lines.append(f"❌ ما لقيت أي مجموعة بيانات تطابق: {topic} — {period}")
        lines.append("")
        lines.append("جرّب صياغة أوسع، أو حدد الجهة التي تتوقع نشرها للبيانات.")

    lines.extend(_suggestions_block(answer))
    lines.extend(_errors_block(answer))

    if len(answer.findings) > 1:
        lines.append("")
        lines.append("نتائج أخرى ذات صلة:")
        for position, finding in enumerate(answer.findings[1:4], start=2):
            lines.extend(_finding_block(finding, index=position))

    return "\n".join(lines)
