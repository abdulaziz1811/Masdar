"""Tests for Arabic normalisation, Hijri handling and query parsing."""

from masdar.domain.models import Calendar, Dimension, PeriodKind
from masdar.nlu import hijri
from masdar.nlu.normalize import contains_phrase, fold_digits, normalize, stems
from masdar.nlu.parser import parse


class TestNormalize:
    def test_folds_arabic_indic_digits(self):
        assert fold_digits("٢٠٢٦") == "2026"
        assert fold_digits("۱۴۴۷") == "1447"

    def test_strips_diacritics_and_folds_hamza(self):
        assert normalize("الطَّاقَة الكَهْرَبائِيَّة") == normalize("الطاقة الكهربائية")
        assert normalize("إحصاءات") == normalize("احصاءات")

    def test_folds_ta_marbuta_and_alef_maksura(self):
        assert normalize("سنة") == "سنه"
        assert normalize("إلى") == "الي"

    def test_splits_on_punctuation(self):
        assert stems("الطاقة،الكهرباء") == ["طاقه", "كهرباء"]

    def test_phrase_match_ignores_articles_on_either_side(self):
        # article on the query side only
        assert contains_phrase("بالمناطق", "مناطق")
        # article on the lexicon side only
        assert contains_phrase("صادرات السعودية", "الصادرات")
        # and on neither
        assert contains_phrase("استهلاك كهرباء", "كهرباء")

    def test_phrase_match_respects_token_boundaries(self):
        assert not contains_phrase("المناطق", "طق")


class TestHijri:
    def test_known_anchors(self):
        # 1400 AH began Nov 1979, 1444 AH began Jul 2022, 1447 AH began Jun 2025.
        assert hijri.hijri_to_gregorian_years(1400) == (1979, 1980)
        assert hijri.hijri_to_gregorian_years(1444) == (2022, 2023)
        assert hijri.hijri_to_gregorian_years(1447) == (2025, 2026)

    def test_round_trip_is_self_consistent(self):
        # Both Gregorian years a Hijri year touches must map back to it.
        for year in range(1300, 1500):
            for g in hijri.hijri_to_gregorian_years(year):
                assert year in hijri.gregorian_to_hijri_years(g), (year, g)

    def test_gregorian_year_overlaps_consecutive_hijri_years(self):
        for g in range(1950, 2060):
            overlapping = hijri.gregorian_to_hijri_years(g)
            assert overlapping, g
            assert list(overlapping) == list(range(overlapping[0], overlapping[-1] + 1))

    def test_ranges_do_not_overlap(self):
        assert not hijri.looks_gregorian(1447)
        assert not hijri.looks_hijri(2026)


class TestParser:
    def test_parses_the_motivating_query(self):
        r = parse("ابي احصاءات الطاقه الكهربائيه لسنه 2026 حسب المناطق")
        assert r.topic is not None and r.topic.id == "electricity"
        assert r.period.years == (2026,)
        assert r.period.kind is PeriodKind.SINGLE
        assert Dimension.REGION in r.dimensions
        assert r.free_terms == ()

    def test_arabic_indic_year_equals_ascii_year(self):
        a = parse("إحصاءات الطاقة الكهربائية لسنة ٢٠٢٦ حسب المناطق")
        b = parse("إحصاءات الطاقة الكهربائية لسنة 2026 حسب المناطق")
        assert a.period == b.period
        assert a.topic == b.topic

    def test_specific_topic_beats_generic(self):
        # "الطاقة الكهربائية" must win over the shorter "الطاقة".
        assert parse("الطاقة الكهربائية 2024").topic.id == "electricity"
        assert parse("الطاقة المتجددة 2024").topic.id == "energy"

    def test_expands_inclusive_range(self):
        r = parse("بيانات السكان من 2019 إلى 2022")
        assert r.period.kind is PeriodKind.RANGE
        assert r.period.years == (2019, 2020, 2021, 2022)

    def test_dash_and_bayn_ranges(self):
        assert parse("المستشفيات 2023-2024").period.years == (2023, 2024)
        assert parse("الصادرات بين 2021 و 2022").period.years == (2021, 2022)

    def test_hijri_year_expands_to_both_gregorian_years(self):
        r = parse("تقرير المياه لعام 1447هـ حسب المدن")
        assert r.period.calendar is Calendar.HIJRI
        assert r.period.years == (2025, 2026)
        assert Dimension.CITY in r.dimensions

    def test_latest_marker(self):
        r = parse("أحدث بيانات البطالة")
        assert r.period.kind is PeriodKind.LATEST
        assert r.period.years == ()
        assert r.topic.id == "labour"

    def test_english_query(self):
        r = parse("electricity consumption by region 2024")
        assert r.language == "en"
        assert r.topic.id == "electricity"
        assert Dimension.REGION in r.dimensions

    def test_unknown_topic_is_reported_not_guessed(self):
        r = parse("ابي شي عن الفلافل")
        assert r.topic is None
        assert r.unresolved
        assert "الفلافل" in r.free_terms

    def test_explicit_years_beat_latest_marker(self):
        r = parse("أحدث بيانات الكهرباء لسنة 2023")
        assert r.period.kind is PeriodKind.SINGLE
        assert r.period.years == (2023,)
