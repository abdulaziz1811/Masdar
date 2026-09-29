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
from masdar.envfile import load_env_file
from masdar.nlu.lexicon import load_lexicon
from masdar.nlu.llm import LlmUnderstanding
from masdar.nlu.parser import parse as parse_query
from masdar.pipeline.orchestrator import Agent, AgentConfig
from masdar.sources.base import SourceError, SourceRejected, SourceUnreachable
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
            {"year": s.year, "reason": s.reason_ar, "source": s.source_ar, "title": s.title_ar}
            for s in answer.suggestions
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
    lexicon = load_lexicon()
    llm, _ = LlmUnderstanding.from_environment(lexicon)
    agent = Agent(registry=registry, http=http, config=config, lexicon=lexicon, llm=llm)
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
    """Contact every source and report what it actually returns.

    The point is not a pass/fail tally but a diagnosis: which of
    geo-restriction, a WAF, a single-page app or a wrong path is in the way,
    and what to do about each.
    """
    http = HttpClient(timeout=args.timeout, use_cache=not args.no_cache)
    registry = load_registry(demo=args.demo, http=http)

    print(f"منفذ الجلب: {http.describe_transport()}")
    llm, llm_status = LlmUnderstanding.from_environment(load_lexicon())
    if llm is not None:
        ok, why = llm.verify()
        llm_status = f"{llm_status} — المفتاح يعمل" if ok else f"لا يعمل: {why}"
    print(f"الفهم بالذكاء الاصطناعي: {llm_status}")
    if http.transport.name == "direct":
        print(
            "ملاحظة: بعض نطاقات gov.sa مقيَّدة جغرافياً (تم التحقق من\n"
            "        open.data.gov.sa). من خارج السعودية استخدم\n"
            "        MASDAR_HTTP_BACKEND=firecrawl مع FIRECRAWL_API_KEY."
        )
    print()

    rejected: list[str] = []
    unreachable: list[str] = []
    failed: list[str] = []
    ok = 0

    for descriptor in registry.descriptors:
        label = f"{descriptor.id:18}"
        facts = []
        if descriptor.geo_restricted:
            facts.append("مقيَّد جغرافياً")
        if descriptor.waf:
            facts.append("جدار حماية")
        if descriptor.rendering == "spa":
            facts.append("SPA")
        suffix = f"  [{'، '.join(facts)}]" if facts else ""

        try:
            adapter = registry.adapter(descriptor.id)
            result = adapter.probe()
        except NotImplementedError:
            print(f"{label} ⚪ لا يوجد فحص لهذا المحول{suffix}")
            continue
        except SourceRejected as exc:
            rejected.append(descriptor.id)
            print(f"{label} 🛡 رُفض: {exc.reason}{suffix}")
            continue
        except SourceUnreachable as exc:
            unreachable.append(descriptor.id)
            print(f"{label} 🚫 تعذّر الوصول: {exc.reason}{suffix}")
            continue
        except SourceError as exc:
            failed.append(descriptor.id)
            print(f"{label} ❌ {exc.reason}{suffix}")
            continue
        except Exception as exc:
            failed.append(descriptor.id)
            print(f"{label} ❌ خطأ غير متوقع: {exc}{suffix}")
            continue

        ok += 1
        unverified = "" if descriptor.api_verified else "  (⚠️ api_verified: false)"
        print(f"{label} ✅ {result}{unverified}{suffix}")

    total = len(registry.descriptors)
    print(f"\nالنتيجة: {ok} من {total} استجاب.")

    if unreachable:
        print(
            f"\n🚫 لم يُستجَب ({', '.join(unreachable)}): الاتصال لم يكتمل. الأسباب\n"
            "   المرجّحة: تقييد جغرافي، أو سياسة شبكة تحجب gov.sa، أو المضيف متوقف.\n"
            "   جرّب MASDAR_HTTP_BACKEND=firecrawl، أو شغّل الأداة من داخل السعودية."
        )
    if rejected:
        print(
            f"\n🛡 رُفض ({', '.join(rejected)}): المضيف رد لكن جدار الحماية رفض الطلب.\n"
            "   هذا لا يعني عدم وجود البيانات. المسار قد يكون محمياً أو خاطئاً —\n"
            "   افتح الموقع في المتصفح، ثم أدوات المطور ← Network ← فلتر XHR،\n"
            "   وصحّح المسار في masdar/config/sources.yaml."
        )
    if failed:
        print(
            f"\n❌ أخطاء ({', '.join(failed)}): استجاب المضيف بشكل غير متوقع —\n"
            "   راجع المسار في masdar/config/sources.yaml."
        )
    if ok and not (unreachable or rejected or failed):
        print("كل المصادر استجابت. حدّث api_verified: true لكل مصدر تأكّد شكل استجابته.")

    return 1 if (unreachable or rejected or failed) else 0



def _slug(text: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug[:60] or "spec"


def _retire_reconstructions(directory: Path, keep: Path, ids: set[str]) -> None:
    """Remove hand-reconstructed specs that a downloaded one now covers.

    A spec typed out from a pasted portal page carries x-masdar-provenance.
    Once the real download for the same datasets arrives it is the better
    record, and keeping both would leave two sources for the same ids.
    """
    import json

    for other in sorted(directory.glob("*.json")):
        if other == keep:
            continue
        try:
            spec = json.loads(other.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not (spec.get("info") or {}).get("x-masdar-provenance"):
            continue
        theirs = {path.rstrip("/").rsplit("/", 1)[-1] for path in spec.get("paths") or {}}
        if theirs and theirs <= ids:
            other.unlink()
            print(f"   ↳ حُذفت النسخة المُعاد بناؤها يدوياً {other.name}: حلّ محلها الملف الأصلي")


def cmd_import_spec(args: argparse.Namespace) -> int:
    """Validate downloaded OpenAPI files and add them to the spec directory.

    Refuses a file that embeds a credential: some portals put the caller's
    key into examples, and a spec is committed to the repository.
    """
    import json

    from masdar.sources.openapi import SpecError, datasets_from_spec, find_secrets, load_spec

    registry = load_registry()
    descriptor = registry.get(args.source)
    if descriptor is None:
        print(f"❌ مصدر غير معروف: {args.source}")
        return 2
    adapter = registry.adapter(args.source)
    destination = Path(args.dest) if args.dest else adapter._spec_dir()
    if destination is None:
        print(f"❌ المصدر {args.source} لا يحدد spec_dir في sources.yaml")
        return 2
    destination.mkdir(parents=True, exist_ok=True)

    imported = failed = 0
    for raw in args.files:
        path = Path(raw)
        try:
            spec = load_spec(path)
        except SpecError as exc:
            failed += 1
            print(f"❌ {exc}")
            continue

        secrets = find_secrets(spec)
        if secrets:
            failed += 1
            print(f"🔐 {path.name}: رُفض — يحتوي ما يشبه مفتاحاً في:")
            for location in secrets[:5]:
                print(f"      {location}")
            print("   أعد تنزيله، أو احذف هذه القيم، ثم أعد المحاولة. لا تُرفع المفاتيح للمستودع.")
            continue

        template = str(descriptor.api.get("dataset_path") or "/v1/stats/{id}")
        datasets = datasets_from_spec(spec, descriptor.base_url, path.stem, template)
        if not datasets:
            failed += 1
            print(
                f"❌ {path.name}: لا يحتوي مسارات بيانات بالشكل {template} — "
                "ليست مواصفة لهذا المصدر"
            )
            continue

        from masdar.sources.openapi import split_bilingual

        # Portal downloads are all called swagger(-N).json; name the file
        # after what it holds. The operation tag is more specific than the
        # API title ("Electrical Energy Statistics" vs "Energy Statistics").
        first = next(iter(spec.get("paths", {}).values()), {}).get("get", {})
        tag_en = split_bilingual(" ".join(first.get("tags") or []))[1]
        title_en = split_bilingual((spec.get("info") or {}).get("title"))[1]
        generic = path.stem.lower().split("-")[0] in ("swagger", "openapi", "spec")
        stem = _slug(tag_en or title_en or path.stem) if generic else _slug(path.stem)
        body = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"

        target = destination / f"{stem}.json"
        counter = 2
        while target.exists() and target.read_text(encoding="utf-8") != body:
            target = destination / f"{stem}-{counter}.json"
            counter += 1
        if target.exists():
            print(f"⚪ {path.name}: مستورد مسبقاً ({target.name})")
            continue

        target.write_text(body, encoding="utf-8")
        imported += 1
        ids = ", ".join(d["id"] for d in datasets[:4]) + ("…" if len(datasets) > 4 else "")
        print(f"✅ {path.name} → {target.name}: {len(datasets)} مجموعة ({ids})")
        # After the line above, so a retirement reads as this file's doing.
        _retire_reconstructions(destination, target, {d["id"] for d in datasets})

    print(f"\nالنتيجة: {imported} مستورد، {failed} مرفوض.")
    return 1 if failed else 0


def cmd_specs(args: argparse.Namespace) -> int:
    """List the datasets a spec-driven source currently knows about."""
    registry = load_registry()
    if registry.get(args.source) is None:
        print(f"❌ مصدر غير معروف: {args.source}")
        return 2
    adapter = registry.adapter(args.source)
    datasets = adapter._datasets()

    by_api: dict[str, list[dict]] = {}
    for entry in datasets:
        label = entry.get("api_ar") or entry.get("api_en") or "معلنة يدوياً في sources.yaml"
        by_api.setdefault(label, []).append(entry)

    for label, entries in by_api.items():
        print(f"\n■ {label}  ({len(entries)})")
        for entry in entries:
            dimensions = entry.get("dimensions") or []
            time = entry.get("time_dimension") or ("YEAR" if "YEAR" in dimensions else "—")
            dims = [d for d in dimensions if d != time]
            topics = ", ".join(entry.get("topics") or []) or "—"
            print(f"  {entry['id']}")
            print(f"      {str(entry.get('title_ar', ''))[:90]}")
            print(f"      الزمن: {time} | الأبعاد: {', '.join(dims) or '—'} | المواضيع: {topics}")

    problems = adapter.spec_problems()
    print(f"\nالإجمالي: {len(datasets)} مجموعة بيانات.")
    if problems:
        print("\n⚠️ ملفات لم تُقرأ:")
        for problem in problems:
            print(f"   • {problem}")
    return 1 if problems else 0


def cmd_serve(args: argparse.Namespace) -> int:
    from masdar.chat import warmup as warmup_module
    from masdar.chat.overview import SHOWCASE
    from masdar.chat.server import make_server
    from masdar.sources.registry import outside_ksa

    config = AgentConfig(
        max_sources=args.max_sources,
        max_downloads=args.max_downloads,
        out_dir=Path(args.out),
        today=date.fromisoformat(args.today) if args.today else None,
        # An audience waits on every answer: past this, a silent source is
        # reported as such instead of holding the answer up.
        answer_seconds=args.answer_seconds or None,
    )
    lexicon = load_lexicon()

    # Shared by the warm-up and the page, so what one learns about a down or
    # slow source spares the other the wait.
    down_hosts: dict = {}
    slow_sources: dict = {}

    def make_agent(llm=None) -> Agent:
        http = HttpClient(timeout=args.timeout, offline=args.offline, down_hosts=down_hosts)
        registry = load_registry(demo=args.demo, http=http)
        return Agent(registry=registry, http=http, config=config, lexicon=lexicon, llm=llm,
                     slow=slow_sources)

    llm, llm_status = LlmUnderstanding.from_environment(lexicon)
    llm_state = "off"
    if llm is not None:
        ok, why = llm.verify()
        if ok:
            llm_state = "on"
        else:
            # A key that does not work is not used: each question would
            # otherwise wait on a failing call before falling back.
            llm, llm_state, llm_status = None, "error", f"لا يعمل: {why}"
    agent = make_agent(llm)
    warm = None
    if warmup_module.enabled():
        warm = warmup_module.Warmup([e["q"] for e in SHOWCASE])
    try:
        server = make_server(agent, host=args.host, port=args.port, llm_status=llm_status,
                             llm_state=llm_state, warmup=warm)
    except OSError as exc:
        print(f"تعذّر تشغيل الخادم على {args.host}:{args.port} — {exc}", file=sys.stderr)
        return 1
    host, port = server.server_address[:2]
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    print(f"مصدر يعمل على {url}")
    print(f"ملفات الإكسل تُحفظ في: {Path(args.out).resolve()}")
    loaded = getattr(args, "env_loaded", None) or []
    if loaded:
        # Names only: a value read from .env is never printed.
        print(f"قُرئ من ملف .env: {'، '.join(loaded)}")
    print(f"الفهم بالذكاء الاصطناعي: {llm_status}")
    if outside_ksa():
        print("الخادم خارج المملكة (MASDAR_OUTSIDE_KSA): المصادر المقيَّدة جغرافياً لن تُسأل.")
    if warm is not None:
        warm.start(make_agent)
        print(f"يجهّز {len(warm.questions)} من أسئلة العرض في الخلفية…")
    import os

    protected = bool(os.environ.get("MASDAR_ACCESS_CODE", "").strip())
    if protected:
        print("رمز الدخول مفعّل (MASDAR_ACCESS_CODE): الصفحة تطلبه قبل أول سؤال.")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not protected:
        print("⚠️ الخادم مفتوح لغير هذا الجهاز وليس عليه رمز دخول — "
              "عيّن MASDAR_ACCESS_CODE في ملف .env قبل نشره.")
    print("للإيقاف: Ctrl+C")
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_warmup(args: argparse.Namespace) -> int:
    """Ask the showcase questions now, so the presentation finds them ready.

    Every answer is kept in the cache (six hours fresh, and a saved copy for
    up to ninety days if a source is down later), and the table printed is
    a rehearsal: which questions answer, from where, and how fast.
    """
    import time

    from masdar.chat.overview import SHOWCASE

    http = HttpClient(timeout=args.timeout)
    registry = load_registry(http=http)
    lexicon = load_lexicon()
    llm, _ = LlmUnderstanding.from_environment(lexicon)
    config = AgentConfig(out_dir=Path(args.out), max_sources=args.max_sources)
    agent = Agent(registry=registry, http=http, config=config, lexicon=lexicon, llm=llm)
    questions = [e["q"] for e in SHOWCASE] + list(args.questions or [])

    print(f"تجهيز {len(questions)} سؤالاً (قد يستغرق بضع دقائق)…\n")
    ready = 0
    for question in questions:
        started = time.monotonic()
        try:
            answer = agent.answer(question)
        except Exception as exc:  # one broken question must not stop the rest
            print(f"❌ {question}\n     خطأ غير متوقع: {exc}\n")
            continue
        seconds = time.monotonic() - started
        primary = answer.primary
        good = answer.verdict in (Verdict.AVAILABLE, Verdict.PARTIAL, Verdict.NOT_AVAILABLE)
        ready += good
        mark = "✅" if good else "⚠️"
        print(f"{mark} {question}  ({seconds:.0f} ث)")
        print(f"     الحكم: {answer.verdict.value}"
              + (f" — {primary.provenance.publisher_ar}: {primary.candidate.title_ar}"
                 if primary else ""))
        for source, reason in answer.source_errors[:3]:
            print(f"     تعذّر: {source} — {reason}")
        print()
    print(f"جاهز: {ready} من {len(questions)}.")
    if ready < len(questions):
        print("الأسئلة المعلَّمة بـ ⚠️ لم تجد مصدراً يجيب الآن: أعد التشغيل لاحقاً، أو "
              "تجنّبها في العرض. (خارج المملكة بعض المواقع الحكومية لا تُفتح.)")
    return 0 if ready else 3


def cmd_publishers(args: argparse.Namespace) -> int:
    """List the national platform's publishers and classify the new ones.

    Run from a machine that can reach the platform (inside the Kingdom).
    """
    from masdar.sources import discover

    http = HttpClient(timeout=args.timeout)
    registry = load_registry(http=http)
    adapter = registry.adapter("saudi_open_data")

    if args.names:
        names = Path(args.names).read_text(encoding="utf-8").splitlines()
        found = discover.resolve_names(adapter, names)
    else:
        try:
            found = discover.list_publishers(adapter)
        except (SourceUnreachable, SourceError) as exc:
            print(f"تعذّر جلب قائمة الجهات من المنصة: {exc.reason}")
            print("البديل: ملف بأسماء الجهات العربية الدقيقة، اسم في كل سطر:")
            print("   masdar publishers --names names.txt")
            return 3

    known = discover.configured_keys(adapter)
    new = [p for p in found if not ({p.id, p.name_ar} & known)]
    print(f"وُجدت {len(found)} جهة، منها {len(new)} غير مُعدّة. تُصنَّف من عناوين مجموعاتها…")
    discover.classify(adapter, new)

    previous = {
        str(e.get("id") or e.get("name_ar")): e for e in discover.load_discovered(Path(args.out))
    }
    for p in new:
        mark = "✅" if not p.problems else "⚠️"
        detail = "، ".join(p.topics) if not p.problems else "؛ ".join(p.problems)
        print(f"  {mark} {p.name_ar or p.id} — {p.datasets if p.datasets is not None else '؟'}"
              f" مجموعة — {detail}")
        previous.pop(p.id or p.name_ar, None)
    kept = [
        discover.Publisher(
            id=str(e.get("id") or ""), name_ar=str(e.get("name_ar") or ""),
            datasets=e.get("datasets"), topics=tuple(e.get("topics") or ()),
            found_by=str(e.get("found_by") or ""),
        )
        for e in previous.values()
    ]
    Path(args.out).write_text(discover.to_yaml(kept + new), encoding="utf-8")
    usable = sum(1 for p in new if not p.problems)
    print(f"\nكُتب {args.out}: {usable} جهة جديدة جاهزة للبحث (والسابقة محفوظة).")
    return 0


def cmd_worldbank_catalogue(args: argparse.Namespace) -> int:
    """Rebuild the local list of World Bank indicators for the Kingdom."""
    from masdar.sources.adapters import worldbank

    http = HttpClient(timeout=args.timeout, use_cache=False)
    try:
        catalogue = worldbank.build_catalogue(http, load_lexicon())
    except (SourceUnreachable, SourceError) as exc:
        print(f"تعذّر الوصول إلى واجهة البنك الدولي: {exc.reason}")
        return 3
    worldbank.write_catalogue(catalogue, Path(args.out))
    print(f"كُتب {args.out}: {len(catalogue['indicators'])} مؤشراً "
          f"(آخر تحديث لدى البنك: {catalogue.get('lastupdated') or '؟'}).")
    return 0


def _env_port(default: int = 8000) -> int:
    import os

    try:
        return int(os.environ.get("PORT") or default)
    except ValueError:
        return default


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
    ask.add_argument("--max-sources", type=int, default=6)
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

    serve = sub.add_parser("serve", help="شغّل واجهة الشات في المتصفح")
    serve.add_argument("--host", default="127.0.0.1")
    # Hosting platforms (Render, Cloud Run) say which port to listen on in PORT.
    serve.add_argument("--port", type=int, default=_env_port())
    serve.add_argument("--open", action="store_true", help="افتح الصفحة في المتصفح")
    serve.add_argument("--demo", action="store_true", help="أضف المصادر التجريبية")
    serve.add_argument("--offline", action="store_true", help="استخدم المخزن المؤقت فقط")
    serve.add_argument("--out", default="out", help="مجلد ملفات الإكسل")
    serve.add_argument("--max-sources", type=int, default=6)
    serve.add_argument("--max-downloads", type=int, default=3)
    serve.add_argument("--timeout", type=float, default=20.0)
    serve.add_argument("--answer-seconds", type=float, default=25.0,
                       help="أقصى مدة للجواب الواحد بالثواني (0 = بلا حد)")
    serve.add_argument("--today", help="تجاوز تاريخ اليوم (YYYY-MM-DD) للاختبار")
    serve.set_defaults(func=cmd_serve)

    warm = sub.add_parser("warmup", help="جهّز أسئلة العرض مسبقاً وافحص جاهزيتها")
    warm.add_argument("questions", nargs="*", help="أسئلة إضافية غير أسئلة العرض")
    warm.add_argument("--out", default="out", help="مجلد ملفات الإكسل")
    warm.add_argument("--max-sources", type=int, default=6)
    warm.add_argument("--timeout", type=float, default=30.0)
    warm.set_defaults(func=cmd_warmup)

    from masdar.sources.discover import DISCOVERED_FILE

    pubs = sub.add_parser("publishers", help="اكتشف جهات المنصة الوطنية وصنّف مواضيعها")
    pubs.add_argument("--names", help="ملف بأسماء جهات عربية دقيقة، اسم في كل سطر")
    pubs.add_argument("--out", default=str(DISCOVERED_FILE), help="ملف النتيجة")
    pubs.add_argument("--timeout", type=float, default=30.0)
    pubs.set_defaults(func=cmd_publishers)

    from masdar.sources.adapters.worldbank import CATALOGUE

    wb = sub.add_parser("worldbank-catalogue", help="أعد بناء فهرس مؤشرات البنك الدولي")
    wb.add_argument("--out", default=str(CATALOGUE), help="ملف النتيجة")
    wb.add_argument("--timeout", type=float, default=60.0)
    wb.set_defaults(func=cmd_worldbank_catalogue)

    importer = sub.add_parser("import-spec", help="استورد ملفات مواصفات OpenAPI")
    importer.add_argument("files", nargs="+", help="ملفات JSON/YAML من بوابة المطوّرين")
    importer.add_argument("--source", default="gastat_cdata")
    importer.add_argument("--dest", help="مجلد الوجهة (الافتراضي: spec_dir للمصدر)")
    importer.set_defaults(func=cmd_import_spec)

    specs = sub.add_parser("specs", help="اعرض مجموعات البيانات المعروفة لمصدر")
    specs.add_argument("--source", default="gastat_cdata")
    specs.set_defaults(func=cmd_specs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.env_loaded = load_env_file()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
