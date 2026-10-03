"""Tests for reading published tables."""

from pathlib import Path

from masdar.export.tabular import Table, YearAxis, coerce, read_csv
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


class TestInapplicableColumns:
    """SDG tables pad every row with columns that only say "not applicable"."""

    table = Table(
        ["YEAR_TIME", "AGE_ARAB", "UNIT_ENGL", "SEX_ARAB", "OBS_VALUE"],
        [
            ["2021", "لاينطبق", "Not-Applicable", "ذكر", 10.0],
            ["2022", "لا ينطبق", "not applicable", "لاينطبق", 12.5],
        ],
    )

    def test_columns_that_only_say_not_applicable_are_dropped(self):
        cleaned, dropped = self.table.without_inapplicable_columns()
        assert dropped == ["AGE_ARAB", "UNIT_ENGL"]
        assert cleaned.columns == ["YEAR_TIME", "SEX_ARAB", "OBS_VALUE"]
        assert cleaned.rows[1] == ["2022", "لاينطبق", 12.5]

    def test_a_column_with_one_real_value_is_kept(self):
        cleaned, _ = self.table.without_inapplicable_columns()
        assert "SEX_ARAB" in cleaned.columns

    def test_a_blank_cell_does_not_make_a_column_inapplicable(self):
        table = Table(["YEAR_TIME", "NOTE"], [["2021", "لاينطبق"], ["2022", None]])
        cleaned, dropped = table.without_inapplicable_columns()
        assert dropped == [] and cleaned is table

    def test_nothing_to_drop_returns_the_same_table(self):
        table = Table(["YEAR_TIME", "OBS_VALUE"], [["2021", 1.0]])
        cleaned, dropped = table.without_inapplicable_columns()
        assert cleaned is table and dropped == []


class TestBulletinWorkbooks:
    """GASTAT's bulletins: a contents sheet, then one titled table a sheet.

    `Hajj_Statistics_2026_AR.rebuilt.xlsx` is GASTAT's Hajj 2026 workbook
    rebuilt cell by cell from its contents (see tests/test_gastat_site.py).
    """

    WORKBOOK = (Path(__file__).resolve().parent / "fixtures" / "gastat_site"
                / "Hajj_Statistics_2026_AR.rebuilt.xlsx").read_bytes()

    def read(self, question=""):
        from masdar.export.tabular import read_xlsx

        return read_xlsx(self.WORKBOOK, question)

    def test_the_contents_sheet_is_never_the_data(self):
        # Every row of it names 2026, so it once won as "the sheet with years".
        assert self.read().sheet_name != "الفهرس"

    def test_the_sheet_worded_like_the_question_is_read(self):
        assert self.read("عدد الحجاج 2026").sheet_name == "1"
        assert self.read("عدد الحجاج حسب الجنس 2026").sheet_name == "2"
        assert self.read("حجاج الداخل حسب الجنسية").sheet_name == "3"
        assert self.read("المتطوعين في الحج").sheet_name == "8"

    def test_a_tables_title_inside_the_file_gives_its_year(self):
        table = self.read("عدد الحجاج 2026")
        assert table.observed_years() == frozenset({2026})
        assert table.years_from_title

    def test_a_row_of_figures_is_never_the_header(self):
        table = self.read("عدد الحجاج حسب الجنس 2026")
        assert table.columns[0] == "الجنس"
        assert [893396, 813905, 1707301] in table.rows

    def test_a_header_of_years_beats_a_fuller_row_of_figures(self):
        from masdar.export.tabular import _grid_to_table

        table = _grid_to_table([
            ["جدول 3", None, None, None],
            [None, 2019, 2020, 2021],
            ["الرياض", 10, 11, 12],
            ["مكة", 5, 6, 7],
        ])
        assert table.columns[1:] == ["2019", "2020", "2021"]
        assert table.rows[0] == ["الرياض", 10, 11, 12]
