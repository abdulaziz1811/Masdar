"""The chat's preview of an exported file, and when it draws a chart.

A chart is drawn only where the file leaves nothing to choose: one number
per year, or one breakdown for one year. Anything else must stay a table,
because a chart that summed or picked a slice on its own could mislead.
"""

from __future__ import annotations

from openpyxl import Workbook

from masdar.chat.preview import PREVIEW_ROWS, read_preview
from masdar.export.excel import DATA_SHEET


def workbook(tmp_path, columns, rows, name="x.xlsx"):
    book = Workbook()
    sheet = book.active
    sheet.title = DATA_SHEET
    sheet.append(columns)
    for row in rows:
        sheet.append(row)
    path = tmp_path / name
    book.save(path)
    return path


def charts(tmp_path, columns, rows):
    return read_preview(workbook(tmp_path, columns, rows))["charts"]


class TestTable:
    def test_the_first_rows_and_the_total_are_shown(self, tmp_path):
        rows = [[2000 + i, i * 1.5] for i in range(20)]
        preview = read_preview(workbook(tmp_path, ["السنة", "القيمة"], rows))
        assert preview["columns"] == ["السنة", "القيمة"]
        assert len(preview["rows"]) == PREVIEW_ROWS and preview["total_rows"] == 20

    def test_long_text_is_shortened(self, tmp_path):
        preview = read_preview(workbook(tmp_path, ["الوصف"], [["ن" * 300]]))
        assert len(preview["rows"][0][0]) <= 80

    def test_a_file_that_is_not_a_workbook_has_no_preview(self, tmp_path):
        path = tmp_path / "broken.xlsx"
        path.write_bytes(b"not a workbook")
        assert read_preview(path) is None


class TestYearCharts:
    def test_one_value_per_year_is_a_time_chart(self, tmp_path):
        columns = ["السنة", "المؤشر", "القيمة", "رمز المؤشر"]
        rows = [[2022, "استهلاك", 11840.6, "EG"], [2023, "استهلاك", 11910.6, "EG"]]
        (chart,) = charts(tmp_path, columns, rows)
        assert chart["kind"] == "years" and chart["measure"] == "القيمة"
        assert [p["label"] for p in chart["points"]] == ["2022", "2023"]

    def test_a_repeated_year_is_not_charted(self, tmp_path):
        rows = [[2022, 1], [2022, 2], [2023, 3]]
        assert charts(tmp_path, ["السنة", "القيمة"], rows) == []

    def test_the_api_year_column_and_measure_are_recognised(self, tmp_path):
        rows = [["2021", 5.0], ["2022", 6.0]]
        (chart,) = charts(tmp_path, ["YEAR_TIME", "OBSVALUE_OBSV"], rows)
        assert chart["kind"] == "years" and chart["measure"] == "القيمة"


class TestBreakdownCharts:
    COLUMNS = ["REGION_ARAB", "REGION_CODE", "REGION_ENGL", "YEAR_TIME", "OBSVALUE_OBSV"]

    def test_the_latest_year_is_drawn_by_region_largest_first(self, tmp_path):
        rows = [
            ["الرياض", "1", "Riyadh", "2021", 10.0], ["مكة المكرمة", "2", "Makkah", "2021", 9.0],
            ["الرياض", "1", "Riyadh", "2022", 12.0], ["مكة المكرمة", "2", "Makkah", "2022", 15.0],
        ]
        (chart,) = charts(tmp_path, self.COLUMNS, rows)
        assert chart["kind"] == "categories" and chart["note"] == "سنة 2022"
        assert [(p["label"], p["value"]) for p in chart["points"]] == [
            ("مكة المكرمة", 15.0), ("الرياض", 12.0),
        ]

    def test_the_national_total_is_left_out_and_the_chart_says_so(self, tmp_path):
        rows = [
            ["الرياض", "1", "Riyadh", "2022", 12.0], ["مكة المكرمة", "2", "Makkah", "2022", 15.0],
            ["إجمالي المملكة", "0", "Total", "2022", 27.0],
        ]
        (chart,) = charts(tmp_path, self.COLUMNS, rows)
        assert "إجمالي المملكة" not in [p["label"] for p in chart["points"]]
        assert "دون صف الإجمالي" in chart["note"]

    def test_two_breakdowns_at_once_are_not_charted(self, tmp_path):
        columns = ["المنطقة", "الجنس", "السنة", "القيمة"]
        rows = [["الرياض", "ذكر", 2022, 1], ["الرياض", "أنثى", 2022, 2],
                ["مكة المكرمة", "ذكر", 2022, 3], ["مكة المكرمة", "أنثى", 2022, 4]]
        assert charts(tmp_path, columns, rows) == []

    def test_each_measure_gets_its_own_chart(self, tmp_path):
        columns = ["السنة", "المنطقة", "عدد المشتركين", "الاستهلاك"]
        rows = [[2022, "الرياض", 100, 5], [2022, "تبوك", 50, 9]]
        found = charts(tmp_path, columns, rows)
        assert [c["measure"] for c in found] == ["عدد المشتركين", "الاستهلاك"]
