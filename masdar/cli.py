"""Command line entry point.

`ask` is the agent; the rest exist to make the agent inspectable. `parse`
shows what the question was understood to mean and `doctor` reports what each
configured source actually does when contacted -- the two things you need to
debug a wrong answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from masdar.domain.models import Answer, Verdict
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.parser import parse as parse_query
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.base import SourceError
from masdar.sources.http import HttpClient
from masdar.sources.registry import load_registry

EXIT_BY_VERDICT = {
    Verdict.AVAILABLE: 0,
    Verdict.PARTIAL: 0,
    Verdict.NOT_AVAILABLE: 1,
    Verdict.UNVERIFIED: 2,
    Verdict.NO_SOURCE: 1,
    Verdict.SOURCE_UNREACHABLE: 3,
}


def _answer_to_dict(answer: Answer) -> dict:
    return {
        "query": answer.request.raw_query,
        "verdict": answer.verdict.value,
        "topic": answer.request.topic.id if answer.request.topic else None,
        "period": {
            "kind": answer.request.period.kind.value,
            "years": list(answer.request.period.years),
            "calendar": answer.request.period.calendar.value,
        },
        "dimensions": [d.value for d in answer.request.dimensions],
        "findings": [
            {
                "source_id": f.candidate.source_id,
                "title": f.candidate.title_ar or f.candidate.title_en,
                "verdict": f.verdict.value,
                "matched_years": list(f.matched_years),
                "missing_years": list(f.missing_years),
                "coverage_years": sorted(f.coverage.years),
                "coverage_origin": f.coverage.origin.value,
                "landing_url": f.provenance.landing_url,
                "resource_url": f.provenance.resource_url,
                "last_updated": f.provenance.last_updated.isoformat()
                if f.provenance.last_updated
                else None,
                "retrieved_at": f.provenance.retrieved_at.isoformat(),
                "sha256": f.provenance.sha256,
                "export_path": f.export_path,
                "score": f.candidate.score,
                "match_reasons": list(f.candidate.match_reasons),
                "notes": list(f.notes),
            }
            for f in answer.findings
        ],
        "suggestions": [
            {"year": s.year, "reason": s.reason_ar} for s in answer.suggestions
        ],
        "source_errors": [{"source_id": s, "reason": r} for s, r in answer.source_errors],
        "audit": list(answer.audit),
        "message_ar": answer.message_ar,
    }


def cmd_ask(args: argparse.Namespace) -> int:
    http = HttpClient(
        timeout=args.timeout,
        use_cache=not args.no_cache,
        offline=args.offline,
    )
    registry = load_registry(demo=args.demo, http=http)
    config = AgentConfig(
        max_sources=args.max_sources,
        max_downloads=0 if args.no_download else args.max_downloads,
        download=not args.no_download,
        out_dir=Path(args.out),
        export=not args.no_export,
        today=date.fromisoformat(args.today) if args.today else None,
    )
    agent = Agent(registry=registry, http=http, config=config)
    answer = agent.answer(args.query)

    if args.json:
        print(json.dumps(_answer_to_dict(answer), ensure_ascii=False, indent=2))
    else:
        print(answer.message_ar)
        if args.explain:
            print("\n" + "─" * 60)
            print("سجل البحث:")
            for line in answer.audit:
                print(f"   • {line}")

    return EXIT_BY_VERDICT.get(answer.verdict, 1)


def cmd_parse(args: argparse.Namespace) -> int:
    request = parse_query(args.query, load_lexicon())
    payload = {
        "raw_query": request.raw_query,
        "language": request.language,
        "topic": request.topic.id if request.topic else None,
        "topic_label": request.topic.label_ar if request.topic else None,
        "period": {
            "kind": request.period.kind.value,
            "years": list(request.period.years),
            "calendar": request.period.calendar.value,
            "raw": request.period.raw,
        },
        "dimensions": [d.value for d in request.dimensions],
        "free_terms": list(request.free_terms),
        "unresolved": list(request.unresolved),
        "search_terms": list(request.search_terms()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    registry = load_registry(demo=args.demo)
    for descriptor in sorted(registry.descriptors, key=lambda d: -d.authority):
        flags = []
        if descriptor.synthetic:
            flags.append("تجريبي")
        if not descriptor.api_verified:
            flags.append("واجهة غير مُتحقَّقة")
        suffix = f"  [{'، '.join(flags)}]" if flags else ""
        print(f"{descriptor.id:18} {descriptor.authority:>4}  {descriptor.name_ar}{suffix}")
        print(f"{'':18}      {descriptor.base_url}  (adapter: {descriptor.adapter})")
        print(f"{'':18}      المواضيع: {', '.join(descriptor.topics) or '—'}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Contact every source and report what it actually returns."""
    http = HttpClient(timeout=args.timeout, use_cache=not args.no_cache)
    registry = load_registry(demo=args.demo, http=http)
    failures = 0

    print("فحص المصادر:\n")
    for descriptor in registry.descriptors:
        label = f"{descriptor.id:18}"
        try:
            adapter = registry.adapter(descriptor.id)
            result = adapter.probe()
        except NotImplementedError:
            print(f"{label} ⚪ لا يوجد فحص لهذا المحول")
            continue
        except SourceError as exc:
            failures += 1
            print(f"{label} ❌ {exc.reason}")
            continue
        except Exception as exc:
            failures += 1
            print(f"{label} ❌ خطأ غير متوقع: {exc}")
            continue
        verified = "" if descriptor.api_verified else "  (⚠️ api_verified: false في sources.yaml)"
        print(f"{label} ✅ {result}{verified}")

    print(
        f"\nالنتيجة: {len(registry.descriptors) - failures} ناجح، {failures} فاشل."
    )
    if failures:
        print(
            "المصادر الفاشلة قد تكون محجوبة بسياسة الشبكة أو تحتاج تصحيح المسار "
            "في masdar/config/sources.yaml."
        )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="masdar",
        description="وكيل بحث في البيانات المفتوحة الرسمية السعودية",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="اسأل عن بيانات")
    ask.add_argument("query", help="السؤال بالعربية أو الإنجليزية")
    ask.add_argument("--demo", action="store_true", help="أضف المصادر التجريبية")
    ask.add_argument("--json", action="store_true", help="أخرج النتيجة كـ JSON")
    ask.add_argument("--explain", action="store_true", help="اعرض سجل البحث")
    ask.add_argument("--out", default="out", help="مجلد ملفات الإكسل")
    ask.add_argument("--max-sources", type=int, default=4)
    ask.add_argument("--max-downloads", type=int, default=3)
    ask.add_argument("--no-download", action="store_true", help="لا تنزّل الملفات للتحقق")
    ask.add_argument("--no-export", action="store_true", help="لا تُنشئ ملف إكسل")
    ask.add_argument("--no-cache", action="store_true")
    ask.add_argument("--offline", action="store_true", help="استخدم المخزن المؤقت فقط")
    ask.add_argument("--timeout", type=float, default=20.0)
    ask.add_argument("--today", help="تجاوز تاريخ اليوم (YYYY-MM-DD) للاختبار")
    ask.set_defaults(func=cmd_ask)

    parse_cmd = sub.add_parser("parse", help="اعرض تحليل السؤال فقط")
    parse_cmd.add_argument("query")
    parse_cmd.set_defaults(func=cmd_parse)

    sources = sub.add_parser("sources", help="اعرض سجل المصادر")
    sources.add_argument("--demo", action="store_true")
    sources.set_defaults(func=cmd_sources)

    doctor = sub.add_parser("doctor", help="افحص الوصول إلى كل مصدر")
    doctor.add_argument("--demo", action="store_true")
    doctor.add_argument("--no-cache", action="store_true")
    doctor.add_argument("--timeout", type=float, default=20.0)
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
