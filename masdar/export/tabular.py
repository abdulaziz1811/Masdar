"""Reading published tables into a normalised form.

Official spreadsheets are messy in predictable ways, and three of those ways
decide whether the year a user asked for can be found at all:

* the year may be a column ("long" layout) or a set of column headers
  ("wide" layout, very common in statistical yearbooks);
* the real header row is often buried under title and logo rows;
* CSVs arrive with BOMs and with comma, semicolon or tab separators.

Getting the year axis right is what lets the verifier make a claim about a
year from the data itself instead of from a filename.
"""

from __future__ import annotations

import csv
import enum
import io
import re
from dataclasses import dataclass, field

from masdar.nlu.normalize import contains_phrase, fold_digits
from masdar.nlu.parser import extract_years

MAX_HEADER_SCAN = 15
MAX_ROWS = 200_000

_YEAR_COLUMN_NAMES = ("السنة", "سنة", "العام", "عام", "year", "السنه", "الفترة", "period")


class YearAxis(enum.Enum):
    ROWS = "rows"        # one column holds the year
    COLUMNS = "columns"  # each year is its own column
    NONE = "none"


@dataclass
class Table:
    columns: list[str]
    rows: list[list[object]]
    sheet_name: str = ""
    # Rows skipped above the header, kept so the export can show provenance
    # of the layout decision.
    preamble: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    # -- layout --------------------------------------------------------
    def year_column_index(self) -> int | None:
        """Index of a column holding years, by name then by content."""
        for index, name in enumerate(self.columns):
            named_year = any(
                contains_phrase(str(name), candidate) for candidate in _YEAR_COLUMN_NAMES
            )
            if named_year and self._column_is_yearlike(index):
                return index
        for index in range(len(self.columns)):
            if self._column_is_yearlike(index, threshold=0.9):
                return index
        return None

    def _column_is_yearlike(self, index: int, threshold: float = 0.6) -> bool:
        values = [row[index] for row in self.rows[:500] if index < len(row)]
        values = [v for v in values if v not in (None, "")]
        if not values:
            return False
        hits = sum(1 for v in values if extract_years(str(v)))
        return hits / len(values) >= threshold

    def year_columns(self) -> dict[int, int]:
        """Map column index -> year, for headers that are themselves years."""
        found: dict[int, int] = {}
        for index, name in enumerate(self.columns):
            years = extract_years(str(name))
            # A header naming exactly one year is a year column; one naming a
            # range ("2015-2024") is a label, not a data column.
            if len(years) == 1:
                found[index] = years[0]
        return found

    def year_axis(self) -> YearAxis:
        year_cols = self.year_columns()
        # Years as headers only counts when most columns are years; a single
        # "2024" column next to labels is a long table with an odd name.
        if len(year_cols) >= 2 and len(year_cols) >= (len(self.columns) - len(year_cols)):
            return YearAxis.COLUMNS
        if self.year_column_index() is not None:
            return YearAxis.ROWS
        if year_cols:
            return YearAxis.COLUMNS
        return YearAxis.NONE

    # -- observation ---------------------------------------------------
    def observed_years(self) -> frozenset[int]:
        """Years actually present in the data. Authoritative evidence."""
        axis = self.year_axis()
        if axis is YearAxis.COLUMNS:
            return frozenset(self.year_columns().values())
        if axis is YearAxis.ROWS:
            index = self.year_column_index()
            if index is None:
                return frozenset()
            years: set[int] = set()
            for row in self.rows:
                if index < len(row):
                    years.update(extract_years(str(row[index])))
            return frozenset(years)
        return frozenset()

    def observed_dimensions(self, dimension_words: dict) -> set:
        """Which breakdown axes the columns provide."""
        present = set()
        header_text = " ".join(str(c) for c in self.columns)
        for dimension, words in dimension_words.items():
            if any(contains_phrase(header_text, word) for word in words):
                present.add(dimension)
        return present

    # -- slicing -------------------------------------------------------
    def filter_years(self, years: frozenset[int]) -> Table:
        """Narrow the table to the requested years, whatever the layout."""
        if not years:
            return self
        axis = self.year_axis()

        if axis is YearAxis.ROWS:
            index = self.year_column_index()
            if index is None:
                return self
            kept = [
                row for row in self.rows
                if index < len(row) and years & set(extract_years(str(row[index])))
            ]
            return Table(list(self.columns), kept, self.sheet_name, list(self.preamble))

        if axis is YearAxis.COLUMNS:
            year_cols = self.year_columns()
            keep_indexes = [i for i in range(len(self.columns)) if i not in year_cols]
            keep_indexes += [i for i, y in sorted(year_cols.items()) if y in years]
            keep_indexes.sort()
            columns = [self.columns[i] for i in keep_indexes]
            rows = [[row[i] if i < len(row) else None for i in keep_indexes] for row in self.rows]
            return Table(columns, rows, self.sheet_name, list(self.preamble))

        return self


_NUMERIC = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")


def coerce(value: object) -> object:
    """Turn numeric-looking text into a number.

    CSV gives everything back as strings; leaving them that way produces a
    spreadsheet whose columns cannot be summed or charted, which defeats the
    point of exporting one. Percentages, codes and anything ambiguous are
    left as text.
    """
    if not isinstance(value, str):
        return value
    text = fold_digits(value.strip())
    if not text or not _NUMERIC.match(text):
        return value.strip()
    bare = text.replace(",", "")
    # A leading zero is usually an identifier, not a quantity.
    if len(bare.lstrip("-")) > 1 and bare.lstrip("-").startswith("0") and "." not in bare:
        return value.strip()
    try:
        number = float(bare)
    except ValueError:
        return value.strip()
    return int(number) if number.is_integer() and "." not in bare else number


def _find_header_row(grid: list[list[object]]) -> int:
    """Pick the header row, skipping title and logo rows above it.

    The header is the row with the most filled cells among the first few,
    provided real rows follow it. Ties go to the earliest row.
    """
    best_index, best_filled = 0, -1
    limit = min(MAX_HEADER_SCAN, len(grid))
    for index in range(limit):
        filled = sum(1 for cell in grid[index] if cell not in (None, ""))
        if filled < 2:
            continue
        if index + 1 >= len(grid):
            continue
        if filled > best_filled:
            best_index, best_filled = index, filled
    return best_index if best_filled > 0 else 0


def _grid_to_table(grid: list[list[object]], sheet_name: str = "") -> Table:
    grid = [row for row in grid if any(cell not in (None, "") for cell in row)]
    if not grid:
        return Table([], [], sheet_name)
    header_index = _find_header_row(grid)
    header = grid[header_index]
    columns = [
        (str(cell).strip() if cell not in (None, "") else f"عمود {i + 1}")
        for i, cell in enumerate(header)
    ]
    rows = [row for row in grid[header_index + 1:]]
    preamble = [
        " | ".join(str(c) for c in row if c not in (None, ""))
        for row in grid[:header_index]
    ]
    return Table(columns, rows, sheet_name, preamble)


def read_csv(content: bytes) -> Table:
    text = content.decode("utf-8-sig", errors="replace")
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    grid = [
        [coerce(cell) for cell in row]
        for _, row in zip(range(MAX_ROWS), reader, strict=False)
    ]
    return _grid_to_table(grid)


def read_xlsx(content: bytes) -> Table:
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)

    def rank(table: Table) -> tuple[bool, int]:
        # A sheet carrying years beats one without; among equals, the longer
        # one wins. Yearbooks often open with a contents or notes sheet.
        return bool(table.observed_years()), len(table)

    best: Table | None = None
    for sheet in workbook.worksheets:
        grid = [
            list(row)
            for _, row in zip(range(MAX_ROWS), sheet.iter_rows(values_only=True), strict=False)
        ]
        table = _grid_to_table(grid, sheet.title)
        if best is None or rank(table) > rank(best):
            best = table
    workbook.close()
    return best or Table([], [])


def read_table(content: bytes, fmt: str) -> Table:
    fmt = (fmt or "").upper()
    if fmt in ("CSV", "TSV", "TXT"):
        return read_csv(content)
    if fmt in ("XLSX", "XLSM", "XLS"):
        return read_xlsx(content)
    raise ValueError(f"unsupported tabular format: {fmt or 'unknown'}")
