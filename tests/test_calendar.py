from datetime import date, datetime
from zoneinfo import ZoneInfo

from academic_calendar import calendar_flags


def test_fall_2026_first_day_is_instructional():
    flags = calendar_flags(date(2026, 8, 17))
    assert flags["is_instructional"] == 1
    assert flags["is_break"] == 0
    assert flags["is_exam_week"] == 0


def test_exam_week():
    flags = calendar_flags(date(2026, 12, 3))
    assert flags["is_exam_week"] == 1
    assert flags["is_instructional"] == 0


def test_fall_break():
    flags = calendar_flags(date(2026, 10, 19))
    assert flags["is_break"] == 1
    assert flags["is_instructional"] == 0


def test_labor_day_holiday():
    flags = calendar_flags(date(2026, 9, 7))
    assert flags["is_holiday"] == 1
    assert flags["is_break"] == 1
    assert flags["is_instructional"] == 0


def test_unknown_future_date_is_neutral():
    flags = calendar_flags(date(2035, 3, 4))
    assert flags == {
        "is_instructional": 0,
        "is_break": 0,
        "is_exam_week": 0,
        "is_holiday": 0,
    }


def test_utc_datetime_uses_eastern_date():
    # 2026-10-19 03:30 UTC is still Oct 18 in Eastern (EDT, UTC-4)
    ts = datetime(2026, 10, 19, 3, 30, tzinfo=ZoneInfo("UTC"))
    flags = calendar_flags(ts)
    assert flags["is_break"] == 0
