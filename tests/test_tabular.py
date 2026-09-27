"""Tests for reading published tables."""

from pathlib import Path

from masdar.export.tabular import YearAxis, coerce, read_csv
from masdar.nlu.lexicon import load_lexicon

FIXTURES = Path(__file__).resolve().parent.parent / "masdar" / "fixtures"


class TestCoerce:
    def test_numbers_become_numbers(self):
        assert coerce("109000") == 109000
        assert coerce("1,234") == 1234
        assert coerce("0.5") == 0.5
        assert coerce("-42") == -42

    def test_arabic_indic_digits(self):
        assert coerce("٥٠٠") == 500

    def test_identifiers_and_units_stay_text(self):
        # A leading zero is an identifier, not a quantity.
        assert coerce("00123") == "00123"
        assert coerce("12.5%") == "12.5%"
        assert coerce("الرياض") == "الرياض"


class TestLongLayout:
    def table(self):
        return read_csv((FIXTURES / "demo_wera" / "electricity_by_region.csv").read_bytes())

    def test_reads_header_and_rows(self):
        t = self.table()
        assert t.columns[0] == "السنة"
        assert len(t) == 130

    def test_detects_year_column(self):
        t = self.table()
        assert t.year_axis() is YearAxis.ROWS
        assert t.observed_years() == frozenset(range(2015, 2025))

    def test_filters_by_year(self):
        t = self.table()
        assert len(t.filter_years(frozenset({2024}))) == 13
        assert len(t.filter_years(frozenset({2026}))) == 0

    def test_detects_region_dimension(self):
        t = self.table()
        present = t.observed_dimensions(load_lexicon().dimension_words)
        assert any(d.value == "region" for d in present)


class TestWideLayout:
    def table(self):
        csv = (
            "التقرير الإحصائي السنوي للكهرباء,,,\n"
            "المنطقة,2022,2023,2024\n"
            "الرياض,10,11,12\n"
            "جازان,5,6,7\n"
        )
        return read_csv(csv.encode("utf-8"))

    def test_skips_title_row_above_header(self):
        t = self.table()
        assert t.columns == ["المنطقة", "2022", "2023", "2024"]
        assert t.preamble == ["التقرير الإحصائي السنوي للكهرباء"]

    def test_years_as_headers(self):
        t = self.table()
        assert t.year_axis() is YearAxis.COLUMNS
        assert t.observed_years() == frozenset({2022, 2023, 2024})

    def test_filtering_keeps_labels_and_requested_year(self):
        sliced = self.table().filter_years(frozenset({2023}))
        assert sliced.columns == ["المنطقة", "2023"]
        assert sliced.rows == [["الرياض", 11], ["جازان", 6]]

    def test_semicolon_delimiter(self):
        t = read_csv("السنة;المنطقة;القيمة\n2024;الرياض;5\n".encode())
        assert t.columns == ["السنة", "المنطقة", "القيمة"]

    def test_bom_is_stripped(self):
        t = read_csv("﻿السنة,القيمة\n2024,5\n".encode())
        assert t.columns[0] == "السنة"
