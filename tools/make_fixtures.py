"""Regenerates the demo fixture tables.

The numbers are deliberately formulaic, not plausible estimates: fixtures
exist to prove the pipeline works, and inventing realistic-looking national
statistics would risk them being quoted as real. Anything exported from a
fixture source is stamped as sample data by the exporter.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "masdar" / "fixtures"

REGIONS = [
    "الرياض", "مكة المكرمة", "المدينة المنورة", "القصيم", "المنطقة الشرقية",
    "عسير", "تبوك", "حائل", "الحدود الشمالية", "جازان", "نجران", "الباحة", "الجوف",
]

ELECTRICITY_YEARS = range(2015, 2025)   # demo publisher has 2015-2024, no 2025+
POPULATION_YEARS = range(2019, 2024)


def write_electricity() -> None:
    out = ROOT / "demo_wera"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in ELECTRICITY_YEARS:
        for index, region in enumerate(REGIONS, start=1):
            # Formulaic on purpose -- see module docstring.
            subscribers = 100_000 * index + 1_000 * (year - 2015)
            consumption = 500 * index + 10 * (year - 2015)
            rows.append([year, region, subscribers, consumption])

    with (out / "electricity_by_region.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["السنة", "المنطقة", "عدد المشتركين", "الاستهلاك (جيجاوات ساعة)"])
        writer.writerows(rows)

    manifest = {
        "source_id": "demo_wera",
        "datasets": [
            {
                "dataset_id": "demo-electricity-by-region",
                "title_ar": "بيانات تجريبية: الطاقة الكهربائية حسب المناطق",
                "title_en": "SAMPLE: Electricity by administrative region",
                "description": (
                    "بيانات تجريبية مُولّدة للاختبار فقط — ليست بيانات رسمية. "
                    "تغطي استهلاك الكهرباء وعدد المشتركين حسب المنطقة الإدارية."
                ),
                "landing_url": "https://example.invalid/demo/electricity",
                "publisher_ar": "مصدر تجريبي (غير رسمي)",
                "publisher_en": "Demo source (not official)",
                "last_updated": "2025-06-30",
                "license": "Demo / sample data",
                "topics": ["electricity", "energy"],
                "claimed_years": list(ELECTRICITY_YEARS),
                "claimed_years_exhaustive": True,
                "resources": [
                    {
                        "path": "electricity_by_region.csv",
                        "format": "CSV",
                        "title": "الاستهلاك وعدد المشتركين حسب المنطقة",
                    }
                ],
            }
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_population() -> None:
    out = ROOT / "demo_gastat"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in POPULATION_YEARS:
        for index, region in enumerate(REGIONS, start=1):
            for gender in ("ذكور", "إناث"):
                count = 200_000 * index + (5_000 if gender == "ذكور" else 0) + 1_000 * (year - 2019)
                rows.append([year, region, gender, count])

    target = out / "population_by_region_gender.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["السنة", "المنطقة", "الجنس", "عدد السكان"])
        writer.writerows(rows)

    manifest = {
        "source_id": "demo_gastat",
        "datasets": [
            {
                "dataset_id": "demo-population-by-region-gender",
                "title_ar": "بيانات تجريبية: السكان حسب المنطقة والجنس",
                "title_en": "SAMPLE: Population by region and gender",
                "description": "بيانات تجريبية مُولّدة للاختبار فقط — ليست بيانات رسمية.",
                "landing_url": "https://example.invalid/demo/population",
                "publisher_ar": "مصدر تجريبي (غير رسمي)",
                "publisher_en": "Demo source (not official)",
                "last_updated": "2024-09-15",
                "license": "Demo / sample data",
                "topics": ["population"],
                "claimed_years": list(POPULATION_YEARS),
                "claimed_years_exhaustive": True,
                "resources": [
                    {
                        "path": "population_by_region_gender.csv",
                        "format": "CSV",
                        "title": "السكان حسب المنطقة والجنس",
                    }
                ],
            }
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    write_electricity()
    write_population()
    print(f"fixtures written to {ROOT}")
