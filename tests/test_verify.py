"""Tests for the claim-strength rules.

These encode the central promise: the strength of a claim never exceeds the
strength of its evidence.
"""

from datetime import date

import pytest

from masdar.domain.models import Coverage, CoverageOrigin, Verdict
from masdar.nlu.parser import parse
from masdar.pipeline.verify import suggest_years, verify

TODAY = date(2026, 9, 27)
YEARS = frozenset(range(2015, 2025))


def observed(years=YEARS):
    return Coverage(years=years, origin=CoverageOrigin.OBSERVED_DATA, is_exhaustive=True)


def claimed(years=YEARS, exhaustive=True):
    return Coverage(years=years, origin=CoverageOrigin.METADATA_CLAIM, is_exhaustive=exhaustive)


def hinted(years=frozenset({2024})):
    return Coverage(years=years, origin=CoverageOrigin.INFERRED_TITLE)


class TestObservedEvidence:
    def test_present_year_is_available(self):
        v = verify(parse("الكهرباء 2024"), observed(), today=TODAY)
        assert v.verdict is Verdict.AVAILABLE
        assert v.matched_years == (2024,)

    def test_absent_year_is_not_available(self):
        v = verify(parse("الكهرباء 2026"), observed(), today=TODAY)
        assert v.verdict is Verdict.NOT_AVAILABLE
        assert v.missing_years == (2026,)

    def test_range_partly_present_is_partial(self):
        v = verify(parse("الكهرباء من 2023 إلى 2026"), observed(), today=TODAY)
        assert v.verdict is Verdict.PARTIAL
        assert v.matched_years == (2023, 2024)
        assert v.missing_years == (2025, 2026)


class TestTitleHintsProveNothing:
    @pytest.mark.parametrize("query", ["الكهرباء 2024", "الكهرباء 2026"])
    def test_never_available_from_a_title(self, query):
        # Even when the hinted year matches, a title is not evidence.
        v = verify(parse(query), hinted(), today=TODAY)
        assert v.verdict is Verdict.UNVERIFIED

    def test_no_coverage_at_all_is_unverified(self):
        v = verify(parse("الكهرباء 2024"), Coverage.unknown(), today=TODAY)
        assert v.verdict is Verdict.UNVERIFIED


class TestMetadataClaims:
    def test_claim_supports_availability(self):
        v = verify(parse("الكهرباء 2024"), claimed(), today=TODAY)
        assert v.verdict is Verdict.AVAILABLE

    def test_exhaustive_claim_supports_absence(self):
        v = verify(parse("الكهرباء 2026"), claimed(exhaustive=True), today=TODAY)
        assert v.verdict is Verdict.NOT_AVAILABLE

    def test_non_exhaustive_claim_cannot_prove_absence(self):
        # The portal may simply not have declared every year it holds.
        v = verify(parse("الكهرباء 2026"), claimed(exhaustive=False), today=TODAY)
        assert v.verdict is Verdict.UNVERIFIED


class TestOpenEndedRequests:
    def test_latest_returns_newest_year(self):
        v = verify(parse("أحدث بيانات الكهرباء"), observed(), today=TODAY)
        assert v.verdict is Verdict.AVAILABLE
        assert v.matched_years == (2024,)

    def test_latest_from_a_title_hint_is_unverified(self):
        v = verify(parse("أحدث بيانات الكهرباء"), hinted(), today=TODAY)
        assert v.verdict is Verdict.UNVERIFIED


class TestPlausibilityNotes:
    def test_explains_a_year_that_has_not_finished(self):
        v = verify(parse("الكهرباء 2026"), observed(), today=TODAY)
        assert any("لم تنتهِ بعد" in note for note in v.notes)

    def test_explains_a_year_that_has_not_started(self):
        v = verify(parse("الكهرباء 2030"), observed(), today=TODAY)
        assert any("لم تبدأ بعد" in note for note in v.notes)

    def test_notes_do_not_override_evidence(self):
        # If a source really holds the current year, that beats the heuristic.
        coverage = observed(YEARS | {2026})
        v = verify(parse("الكهرباء 2026"), coverage, today=TODAY)
        assert v.verdict is Verdict.AVAILABLE


class TestSuggestions:
    def test_offers_latest_when_year_missing(self):
        request = parse("الكهرباء 2026")
        v = verify(request, observed(), today=TODAY)
        suggestions = suggest_years(request, observed(), v.matched_years)
        assert [s.year for s in suggestions] == [2024]

    def test_nothing_suggested_when_request_satisfied(self):
        request = parse("الكهرباء 2024")
        v = verify(request, observed(), today=TODAY)
        assert suggest_years(request, observed(), v.matched_years) == ()

    def test_nothing_suggested_for_an_answered_latest_request(self):
        request = parse("أحدث بيانات الكهرباء")
        v = verify(request, observed(), today=TODAY)
        assert suggest_years(request, observed(), v.matched_years) == ()

    def test_partial_range_does_not_suggest_older_years(self):
        # 2023-2024 were delivered; pointing at 2022 would be noise.
        request = parse("الكهرباء من 2023 إلى 2026")
        v = verify(request, observed(), today=TODAY)
        assert suggest_years(request, observed(), v.matched_years) == ()

    def test_offers_neighbour_above_an_old_year(self):
        request = parse("الكهرباء 2010")
        v = verify(request, observed(), today=TODAY)
        years = [s.year for s in suggest_years(request, observed(), v.matched_years)]
        assert 2024 in years and 2015 in years

    def test_never_suggests_from_a_title_hint(self):
        request = parse("الكهرباء 2026")
        assert suggest_years(request, hinted(), ()) == ()
