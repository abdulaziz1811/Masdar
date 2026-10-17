"""Tests for Arabic normalisation, Hijri handling and query parsing."""

from datetime import date

import pytest

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


class TestRelativePeriods:
    """«العام الماضي» and the like, read against the day the question is asked."""

    TODAY = date(2026, 10, 17)

    @pytest.mark.parametrize("question, years", [
        ("كم عدد السكان العام الماضي", [2025]),
        ("البطالة السنة اللي فاتت", [2025]),
        ("عدد السياح هذا العام", [2026]),
        ("استهلاك الكهرباء قبل سنتين", [2024]),
        ("الصادرات قبل 3 سنوات", [2023]),
        # «The last five years» of statistics end last year: this year's
        # annual figures are not out until it ends.
        ("معدل التضخم آخر خمس سنوات", [2021, 2022, 2023, 2024, 2025]),
        ("معدل البطالة خلال آخر 3 سنوات", [2023, 2024, 2025]),
        ("السكان في السنوات الثلاث الأخيرة", [2023, 2024, 2025]),
        ("unemployment last 5 years", [2021, 2022, 2023, 2024, 2025]),
    ])
    def test_relative_years(self, question, years):
        request = parse(question, today=self.TODAY)
        assert list(request.period.years) == years
        # The words of the period are spent, not read as the subject.
        assert request.free_terms == ()

    def test_an_explicit_year_wins(self):
        assert parse("البطالة 2023 مقارنة بالعام الماضي", today=self.TODAY).period.years \
            == (2023,)

    def test_a_count_is_never_the_start_of_a_longer_number(self):
        # «قبل 100 عام» is not «قبل 10» (ten years ago) with a zero after it.
        assert parse("عدد السكان قبل 100 عام", today=self.TODAY).period.years == ()

    def test_the_latest_is_still_the_latest(self):
        request = parse("أحدث بيانات البطالة", today=self.TODAY)
        assert request.period.kind is PeriodKind.LATEST and request.period.years == ()


class TestWordsPeopleUse:
    """Words found missing by probing colloquial questions (2026-10-17)."""

    @pytest.mark.parametrize("question, topic", [
        ("ابي اعرف كم صيدلية في الرياض", "health"),
        ("كم سائح جا المملكة 2024", "tourism"),
        ("انتاج القمح في المملكة", "agriculture"),
        ("عدد الطلاب المبتعثين 2023", "education"),
        ("احصائيات الزواج والطلاق 2022", "justice"),
        ("كم واحد يشتغل عن بعد في السعودية", "labour"),
        ("عدد الحوادث المرورية 2023", "transport"),
    ])
    def test_the_topic_is_found(self, question, topic):
        assert parse(question).topic.id == topic

    def test_youth_asks_for_the_age_breakdown(self):
        assert Dimension.AGE in parse("وش نسبة البطالة عند الشباب").dimensions

    def test_joined_letters_do_not_leave_words_behind(self):
        # «والطلاق» spent with its topic; «لبيانات» is a stopword with «ل».
        assert parse("احصائيات الزواج والطلاق 2022").free_terms == ()
        assert parse("آخر تحديث لبيانات السكان").free_terms == ()
