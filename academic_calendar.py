"""Hardcoded NC State academic calendar flags.

Dates are inclusive and interpreted in America/New_York. Unknown dates
return all-zero flags so the model does not invent a term.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from config import Config

ET = ZoneInfo("America/New_York")

# Inclusive [start, end] ranges. Source: studentservices.ncsu.edu academic calendar
# plus the published three-year calendar for the prior year.
_INSTRUCTIONAL: tuple[tuple[date, date], ...] = (
    (date(2025, 8, 18), date(2025, 12, 2)),
    (date(2026, 1, 12), date(2026, 4, 28)),
    (date(2026, 8, 17), date(2026, 12, 1)),
    (date(2027, 1, 11), date(2027, 4, 27)),
    (date(2027, 5, 19), date(2027, 6, 23)),
    (date(2027, 6, 28), date(2027, 7, 30)),
)

_EXAM: tuple[tuple[date, date], ...] = (
    (date(2025, 12, 4), date(2025, 12, 10)),
    (date(2026, 4, 30), date(2026, 5, 6)),
    (date(2026, 12, 3), date(2026, 12, 9)),
    (date(2027, 4, 29), date(2027, 5, 5)),
    (date(2027, 6, 24), date(2027, 6, 25)),
    (date(2027, 8, 2), date(2027, 8, 3)),
)

_BREAK: tuple[tuple[date, date], ...] = (
    (date(2025, 10, 13), date(2025, 10, 14)),
    (date(2025, 11, 26), date(2025, 11, 28)),
    (date(2025, 12, 11), date(2026, 1, 11)),
    (date(2026, 3, 16), date(2026, 3, 20)),
    (date(2026, 5, 7), date(2026, 8, 16)),
    (date(2026, 10, 19), date(2026, 10, 20)),
    (date(2026, 11, 25), date(2026, 11, 27)),
    (date(2026, 12, 10), date(2027, 1, 10)),
    (date(2027, 3, 15), date(2027, 3, 19)),
    (date(2027, 5, 6), date(2027, 5, 18)),
)

_HOLIDAY: frozenset[date] = frozenset({
    date(2025, 9, 1),
    date(2025, 11, 27),
    date(2025, 11, 28),
    date(2026, 1, 19),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 11, 27),
    date(2027, 1, 18),
    date(2027, 5, 31),
    date(2027, 6, 18),
    date(2027, 7, 5),
})

_NO_CLASS: frozenset[date] = frozenset({
    date(2025, 9, 1),
    date(2026, 2, 17),
    date(2026, 9, 7),
    date(2026, 9, 29),
    date(2026, 12, 2),
    date(2027, 1, 18),
    date(2027, 2, 16),
    date(2027, 4, 28),
})


def _as_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=ZoneInfo(Config.TIMEZONE))
        return value.astimezone(ET).date()
    return value


def _in_ranges(day: date, ranges: tuple[tuple[date, date], ...]) -> bool:
    return any(start <= day <= end for start, end in ranges)


def calendar_flags(value: date | datetime) -> dict[str, int]:
    """Return 0/1 flags for an Eastern-local date."""
    day = _as_date(value)
    is_exam = 1 if _in_ranges(day, _EXAM) else 0
    is_holiday = 1 if day in _HOLIDAY else 0
    is_break = 1 if (
        _in_ranges(day, _BREAK) or day in _NO_CLASS or is_holiday
    ) and not is_exam else 0
    is_instructional = 1 if (
        _in_ranges(day, _INSTRUCTIONAL) and not is_break and not is_exam and not is_holiday
    ) else 0
    return {
        "is_instructional": is_instructional,
        "is_break": is_break,
        "is_exam_week": is_exam,
        "is_holiday": is_holiday,
    }
