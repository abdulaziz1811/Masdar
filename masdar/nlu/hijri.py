"""Hijri year handling.

Saudi publications are dated in either calendar, and a Hijri year overlaps
two Gregorian ones (1447 AH runs from mid-2025 into mid-2026). We therefore
never collapse a Hijri year to a single Gregorian year -- we return both
candidates and let the verifier decide which one the data actually holds.
"""

from __future__ import annotations

import math

# Plausible Hijri years for statistical data; outside this range a 4-digit
# number is almost certainly Gregorian or not a year at all.
HIJRI_MIN = 1300
HIJRI_MAX = 1500

GREGORIAN_MIN = 1900
GREGORIAN_MAX = 2100

# Mean Hijri year is ~354.367 days, so ~33.7 Hijri years pass per 32.7
# Gregorian ones. This tabular approximation is accurate to the year across
# the whole range above, which is all the precision a year-level query needs.
_DRIFT = 33.7
_EPOCH = 621.5


def hijri_to_gregorian_years(hijri_year: int) -> tuple[int, int]:
    """Return the two Gregorian years a Hijri year can fall in, in order."""
    start = math.floor(hijri_year - hijri_year / _DRIFT + _EPOCH)
    return start, start + 1


def gregorian_to_hijri_years(gregorian_year: int) -> tuple[int, ...]:
    """Hijri years that can overlap the given Gregorian year.

    Derived by scanning `hijri_to_gregorian_years` rather than by inverting
    the formula: two independent approximations drift apart by a year around
    the edges, and a converter that disagrees with itself is worse than a
    coarse one. Scanning keeps the two directions consistent by construction.
    """
    approx = math.floor((gregorian_year - _EPOCH) * _DRIFT / (_DRIFT - 1))
    return tuple(
        h
        for h in range(max(1, approx - 2), approx + 3)
        if gregorian_year in hijri_to_gregorian_years(h)
    )


def looks_hijri(year: int) -> bool:
    return HIJRI_MIN <= year <= HIJRI_MAX


def looks_gregorian(year: int) -> bool:
    return GREGORIAN_MIN <= year <= GREGORIAN_MAX
