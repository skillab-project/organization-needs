"""
test_demand_analysis.py
=====================================================================
Pytest test-suite for demand_analysis.py — SKILLAB Demand Analysis
=====================================================================

KNOWN BUGS DOCUMENTED BY THIS SUITE
-------------------------------------
BUG-1  _SECTOR_FIELDS = ("sectors")   → this is the *string* "sectors",
       not a tuple. Iterating over it yields individual chars ('s','e',…),
       so _extract_sectors() always returns ["unknown"].
       Fix: change to ("sectors",)  or  ["sectors"].

BUG-2  run_long_term_skills() references the name `sector` which is not
       a parameter of that function → NameError at runtime when data is
       non-empty.
       Fix: add  sector: str = "Unknown"  to the signature.

Tests for intended behaviour are written against the *correct* semantics.
They will fail on the buggy code and pass once the bugs are fixed.
Tests documenting current (buggy) behaviour are in class TestKnownBugs.

Run with:
    pytest test_demand_analysis.py -v
    pytest test_demand_analysis.py -v -k "not KnownBugs"   # skip bug docs
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── module under test ────────────────────────────────────────────
# Adjust the import if your working directory or package differs.
import service as da
from service import app

# ══════════════════════════════════════════════════════════════════
#  SHARED HELPERS & FIXTURES
# ══════════════════════════════════════════════════════════════════

SKILL_URIS = [f"http://data.europa.eu/esco/skill/{i}" for i in range(6)]
OCC_URIS   = [f"http://data.europa.eu/esco/occupation/{i}" for i in range(3)]
ALL_URIS   = SKILL_URIS + OCC_URIS


def _make_items(n: int = 15) -> list[dict]:
    """Return *n* synthetic job-posting dicts spanning the last ~18 months."""
    base  = datetime(2023, 6, 1)
    items = []
    for i in range(n):
        d = (base - timedelta(days=i * 30)).strftime("%Y-%m-%d")
        items.append({
            "upload_date":   d,
            "skills":        SKILL_URIS[: (i % 3) + 1],
            "occupation_id": OCC_URIS[i % 3],
            "sectors":       [["J", "K"][i % 2]],
            "country":       ["DE", "FR", "IT"][i % 3],
        })
    return items


def _fake_label_dict() -> dict[str, str]:
    d: dict[str, str] = {}
    d.update({u: f"Skill {i}"      for i, u in enumerate(SKILL_URIS)})
    d.update({u: f"Occupation {i}" for i, u in enumerate(OCC_URIS)})
    return d


def _fake_esco_df() -> pd.DataFrame:
    ld = _fake_label_dict()
    return pd.DataFrame({
        "conceptUri":     list(ld.keys()),
        "preferredLabel": list(ld.values()),
    })


# Re-usable mock return values for the LT pipeline
def _make_lt_skills_output() -> dict:
    return {
        "metadata":       {"analysis_type": "lt_skills_test"},
        "skills":         [],
        "sector_summary": {"total_entities_analyzed": 0, "category_distribution": {}},
    }


def _make_lt_occs_output() -> dict:
    return {
        "metadata":       {"analysis_type": "lt_occs_test"},
        "occupations":    [],
        "sector_summary": {"total_occupations_analyzed": 0, "category_distribution": {}},
    }


# ── pytest fixtures ───────────────────────────────────────────────

@pytest.fixture
def sample_items() -> list[dict]:
    return _make_items(15)


@pytest.fixture
def label_dict() -> dict[str, str]:
    return _fake_label_dict()


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Redirect da.FOLDER to an isolated temp directory per test."""
    folder = tmp_path / "cache"
    folder.mkdir()
    monkeypatch.setattr(da, "FOLDER", folder)
    return folder


# FastAPI test-client (shared; stateless between tests)
client = TestClient(app)


# ══════════════════════════════════════════════════════════════════
#  1.  SHARED INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════

class TestParseCsvStr:
    def test_splits_on_comma(self):
        assert da._parse_csv_str("a,b,c") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert da._parse_csv_str(" x , y ") == ["x", "y"]

    def test_none_returns_none(self):
        assert da._parse_csv_str(None) is None

    def test_empty_string_returns_none(self):
        assert da._parse_csv_str("") is None

    def test_single_value_list(self):
        assert da._parse_csv_str("solo") == ["solo"]

    def test_trailing_comma_ignored(self):
        result = da._parse_csv_str("a,b,")
        assert "" not in result


class TestParseCsvInt:
    def test_parses_integers(self):
        assert da._parse_csv_int("1,2,3") == [1, 2, 3]

    def test_none_returns_none(self):
        assert da._parse_csv_int(None) is None

    def test_empty_returns_none(self):
        assert da._parse_csv_int("") is None


class TestExtractSectors:
    """
    NOTE: These tests reflect the *intended* behaviour.
    They will fail on the current code due to BUG-1
    (_SECTOR_FIELDS = ("sectors") is a string, not a tuple).
    """

    def test_list_sector_field(self):
        assert da._extract_sectors({"sectors": ["J", "K"]}) == ["J", "K"]

    def test_string_sector_field(self):
        assert da._extract_sectors({"sectors": "J"}) == ["J"]

    def test_missing_key_returns_unknown(self):
        assert da._extract_sectors({}) == ["unknown"]

    def test_none_value_returns_unknown(self):
        assert da._extract_sectors({"sectors": None}) == ["unknown"]

    def test_empty_list_returns_unknown(self):
        assert da._extract_sectors({"sectors": []}) == ["unknown"]

    def test_string_stripped(self):
        assert da._extract_sectors({"sectors": " J "}) == ["J"]


class TestGroupItemsBySector:
    def test_single_sector_grouping(self):
        items = [{"sectors": ["J"]}, {"sectors": ["K"]}, {"sectors": ["J"]}]
        g = da._group_items_by_sector(items)
        assert len(g["J"]) == 2
        assert len(g["K"]) == 1

    def test_item_in_multiple_sectors(self):
        items = [{"sectors": ["J", "K"]}]
        g = da._group_items_by_sector(items)
        assert len(g["J"]) == 1
        assert len(g["K"]) == 1

    def test_empty_input_empty_dict(self):
        assert da._group_items_by_sector([]) == {}

    def test_missing_sector_field_goes_to_unknown(self):
        g = da._group_items_by_sector([{"other": True}])
        assert "unknown" in g


class TestCacheHelpers:
    def test_creates_in_progress_stub_when_missing(self, tmp_path):
        path = str(tmp_path / "stub.json")
        data, exists = da._load_or_init_cache(path)
        assert not exists
        assert data is None
        with open(path) as f:
            stub = json.load(f)
        assert stub["status"] == "in_progress"
        assert stub["result"] is None

    def test_returns_existing_data(self, tmp_path):
        path = str(tmp_path / "hit.json")
        payload = {"status": "ok", "result": [1, 2, 3]}
        with open(path, "w") as f:
            json.dump(payload, f)
        data, exists = da._load_or_init_cache(path)
        assert exists and data == payload

    def test_save_then_reload(self, tmp_path):
        path = str(tmp_path / "save.json")
        da._save_cache(path, {"k": "v"})
        with open(path) as f:
            assert json.load(f) == {"k": "v"}

    def test_error_cache_stores_message(self, tmp_path):
        path = str(tmp_path / "err.json")
        da._error_cache(path, ValueError("something broke"))
        with open(path) as f:
            d = json.load(f)
        assert d["status"] == "error"
        assert "something broke" in d["message"]

    def test_save_cache_unicode_safe(self, tmp_path):
        path = str(tmp_path / "unicode.json")
        da._save_cache(path, {"label": "Ανάλυση"})
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert d["label"] == "Ανάλυση"


class TestBuildCacheKey:
    def test_includes_all_parts(self):
        key = da._build_cache_key("prefix", "sector", "org")
        assert "prefix" in key and "sector" in key

    def test_none_parts_excluded(self):
        key = da._build_cache_key("a", None, "b")
        assert "None" not in key

    def test_slash_sanitized(self):
        assert "/" not in da._build_cache_key("sector/IT")

    def test_space_sanitized(self):
        assert " " not in da._build_cache_key("org name")

    def test_deterministic(self):
        k1 = da._build_cache_key("x", "y", "z")
        k2 = da._build_cache_key("x", "y", "z")
        assert k1 == k2


# ══════════════════════════════════════════════════════════════════
#  2.  SHORT-TERM ANALYSIS  —  DATE / QUARTER HELPERS
# ══════════════════════════════════════════════════════════════════

class TestToQuarterStr:
    @pytest.mark.parametrize("date_str,expected", [
        ("2024-01-01", "2024-Q1"),
        ("2024-04-15", "2024-Q2"),
        ("2023-08-30", "2023-Q3"),
        ("2022-12-31", "2022-Q4"),
        ("2023-03-01T12:00:00", "2023-Q1"),   # ISO with time component
    ])
    def test_date_to_quarter(self, date_str, expected):
        assert da._to_quarter_str(date_str) == expected

    def test_none_input_returns_none(self):
        assert da._to_quarter_str(None) is None

    def test_empty_string_returns_none(self):
        assert da._to_quarter_str("") is None

    def test_invalid_format_returns_none(self):
        assert da._to_quarter_str("not-a-date") is None

    def test_quarter_boundaries(self):
        # First and last day of each quarter
        assert da._to_quarter_str("2023-01-01")[-2:] == "Q1"
        assert da._to_quarter_str("2023-03-31")[-2:] == "Q1"
        assert da._to_quarter_str("2023-04-01")[-2:] == "Q2"
        assert da._to_quarter_str("2023-06-30")[-2:] == "Q2"


class TestQuartersBack:
    def test_correct_length(self):
        assert len(da._quarters_back(8)) == 8

    def test_strictly_ascending(self):
        labels = da._quarters_back(6)
        for a, b in zip(labels, labels[1:]):
            ya, qa = int(a[:4]), int(a[-1])
            yb, qb = int(b[:4]), int(b[-1])
            assert (ya, qa) < (yb, qb)

    def test_all_contain_q_marker(self):
        for lbl in da._quarters_back(4):
            assert "-Q" in lbl

    def test_ends_at_or_before_current_quarter(self):
        now = datetime.now()
        cur = (now.year, (now.month - 1) // 3 + 1)
        last = da._quarters_back(4)[-1]
        yr, q = int(last[:4]), int(last[-1])
        assert (yr, q) <= cur


class TestQuartersForward:
    def test_correct_length(self):
        assert len(da._quarters_forward(6)) == 6

    def test_all_strictly_future(self):
        now = datetime.now()
        curq = (now.year, (now.month - 1) // 3 + 1)
        for lbl in da._quarters_forward(4):
            yr, q = int(lbl[:4]), int(lbl[-1])
            assert (yr, q) > curq

    def test_strictly_ascending(self):
        labels = da._quarters_forward(4)
        for a, b in zip(labels, labels[1:]):
            ya, qa = int(a[:4]), int(a[-1])
            yb, qb = int(b[:4]), int(b[-1])
            assert (ya, qa) < (yb, qb)


# ══════════════════════════════════════════════════════════════════
#  2.  SHORT-TERM ANALYSIS  —  TIME-SERIES BUILDERS
# ══════════════════════════════════════════════════════════════════

class TestBuildSkillSeries:
    def test_counts_per_quarter(self):
        items = [
            {"upload_date": "2023-01-01", "skills": ["s1", "s2"]},
            {"upload_date": "2023-02-01", "skills": ["s1"]},
        ]
        series = da._build_skill_series(items)
        assert series["s1"]["2023-Q1"] == 2
        assert series["s2"]["2023-Q1"] == 1

    def test_multiple_quarters_tracked(self):
        items = [
            {"upload_date": "2022-03-01", "skills": ["s1"]},
            {"upload_date": "2022-09-01", "skills": ["s1"]},
            {"upload_date": "2023-01-01", "skills": ["s1"]},
        ]
        series = da._build_skill_series(items)
        assert len(series["s1"]) == 3

    def test_missing_date_skipped(self):
        items = [{"upload_date": None, "skills": ["s1"]}]
        assert da._build_skill_series(items) == {}

    def test_invalid_date_skipped(self):
        items = [{"upload_date": "bad", "skills": ["s1"]}]
        assert da._build_skill_series(items) == {}

    def test_empty_skills_skipped(self):
        items = [{"upload_date": "2023-01-01", "skills": []}]
        assert da._build_skill_series(items) == {}

    def test_custom_date_field(self):
        items = [{"posted_on": "2023-04-01", "skills": ["s1"]}]
        series = da._build_skill_series(items, date_field="posted_on")
        assert "s1" in series


class TestBuildOccupationSeries:
    def test_occupation_id_field(self):
        items = [
            {"upload_date": "2023-01-01", "occupation_id": "occ1"},
            {"upload_date": "2023-02-01", "occupation_id": "occ1"},
        ]
        assert da._build_occupation_series(items)["occ1"]["2023-Q1"] == 2

    def test_occupations_fallback_field(self):
        items = [{"upload_date": "2023-07-01", "occupations": "occ2"}]
        assert "occ2" in da._build_occupation_series(items)

    def test_no_occupation_skipped(self):
        items = [{"upload_date": "2023-01-01"}]
        assert da._build_occupation_series(items) == {}

    def test_missing_date_skipped(self):
        items = [{"occupation_id": "occ1"}]
        assert da._build_occupation_series(items) == {}


class TestFillSeries:
    def test_fills_zeros_for_missing(self):
        qd = {"2023-Q1": 5, "2023-Q3": 10}
        result = da._fill_series(qd, ["2023-Q1", "2023-Q2", "2023-Q3"])
        assert result == [5.0, 0.0, 10.0]

    def test_all_zeros_when_empty_dict(self):
        assert da._fill_series({}, ["2023-Q1", "2023-Q2"]) == [0.0, 0.0]

    def test_returns_floats(self):
        result = da._fill_series({"2023-Q1": 3}, ["2023-Q1"])
        assert all(isinstance(v, float) for v in result)


# ══════════════════════════════════════════════════════════════════
#  2.  SHORT-TERM ANALYSIS  —  FORECASTING
# ══════════════════════════════════════════════════════════════════

class TestLinearForecast:
    def test_correct_number_of_points(self):
        assert len(da._linear_forecast([1.0, 2.0, 3.0], 5)["forecast"]) == 5

    def test_method_label(self):
        assert da._linear_forecast([1.0, 2.0], 2)["method"] == "linear_trend"

    def test_increasing_series_yields_positive_forecast(self):
        fv = da._linear_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 1)["forecast"][0]["value"]
        assert fv > 5.0

    def test_ci_strictly_ordered(self):
        for pt in da._linear_forecast([1.0, 2.0, 3.0, 4.0], 4)["forecast"]:
            assert pt["ci_lower_95"] <= pt["ci_lower_80"]
            assert pt["ci_lower_80"] <= pt["value"]
            assert pt["value"]       <= pt["ci_upper_80"]
            assert pt["ci_upper_80"] <= pt["ci_upper_95"]

    def test_lower_bounds_non_negative(self):
        for pt in da._linear_forecast([10.0, 5.0, 2.0, 1.0], 6)["forecast"]:
            assert pt["ci_lower_95"] >= 0.0
            assert pt["ci_lower_80"] >= 0.0

    def test_empty_series_does_not_crash(self):
        result = da._linear_forecast([], 3)
        assert len(result["forecast"]) == 3

    def test_flat_series_forecast_near_mean(self):
        series = [5.0, 5.0, 5.0, 5.0]
        fv = da._linear_forecast(series, 1)["forecast"][0]["value"]
        assert abs(fv - 5.0) < 2.0

    def test_ci_widens_with_horizon(self):
        """Confidence interval should grow wider for more distant forecasts."""
        pts = da._linear_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 4)["forecast"]
        widths = [p["ci_upper_95"] - p["ci_lower_95"] for p in pts]
        assert widths[0] <= widths[-1]


class TestForecastSeries:
    def test_too_short_uses_linear(self):
        assert da.forecast_series([1.0, 2.0, 3.0], 3)["method"] == "linear_trend"

    def test_all_zero_uses_linear(self):
        assert da.forecast_series([0.0] * 8, 4)["method"] == "linear_trend"

    def test_returns_correct_n_points(self):
        assert len(da.forecast_series(list(range(12)), 6)["forecast"]) == 6

    def test_required_keys_present(self):
        result = da.forecast_series([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        assert {"method", "forecast"} <= set(result)


# ══════════════════════════════════════════════════════════════════
#  2.  SHORT-TERM ANALYSIS  —  ECONOMETRIC METRICS
# ══════════════════════════════════════════════════════════════════

class TestCagr:
    def test_zero_start_returns_none(self):
        assert da._cagr([0.0, 5.0], 1.0) is None

    def test_negative_start_returns_none(self):
        assert da._cagr([-1.0, 5.0], 1.0) is None

    def test_negative_end_returns_none(self):
        assert da._cagr([10.0, -5.0], 1.0) is None

    def test_single_element_returns_none(self):
        assert da._cagr([5.0], 1.0) is None

    def test_doubling_over_2_years(self):
        # 100 → 200 over 2 years = 2^(1/2) - 1 ≈ 41.42 %
        result = da._cagr([100.0, 200.0], 2.0)
        assert result is not None
        assert abs(result - 41.421) < 0.01

    def test_no_growth_zero_cagr(self):
        assert abs(da._cagr([100.0, 100.0], 2.0)) < 0.001

    def test_decline_negative_cagr(self):
        result = da._cagr([100.0, 50.0], 1.0)
        assert result is not None
        assert result < 0


class TestDemandVelocity:
    def test_too_short_returns_none(self):
        assert da._demand_velocity([1.0, 2.0]) is None

    def test_zero_denominator_returns_none(self):
        assert da._demand_velocity([0.0, 1.0, 2.0]) is None

    def test_exact_positive_growth(self):
        # (12 - 10) / 10 * 100 = 20
        assert da._demand_velocity([10.0, 11.0, 12.0]) == pytest.approx(20.0, abs=0.01)

    def test_exact_large_growth(self):
        # (100 - 1) / 1 * 100 = 9900
        assert da._demand_velocity([1.0, 50.0, 100.0]) == pytest.approx(9900.0, abs=1.0)

    def test_decline_negative(self):
        result = da._demand_velocity([10.0, 8.0, 5.0])
        assert result is not None and result < 0


class TestMpr:
    def test_zero_total_returns_none(self):
        assert da._mpr([5.0], [0.0]) is None

    def test_fifty_percent(self):
        assert da._mpr([5.0, 5.0], [10.0, 10.0]) == pytest.approx(50.0, abs=0.01)

    def test_full_penetration(self):
        assert da._mpr([10.0], [10.0]) == pytest.approx(100.0, abs=0.01)

    def test_zero_entity_zero_mpr(self):
        result = da._mpr([0.0, 0.0], [10.0, 10.0])
        assert result == pytest.approx(0.0, abs=0.01)


class TestVolatility:
    def test_too_short_returns_none(self):
        assert da._volatility([1.0, 2.0]) is None

    def test_flat_series_none_or_zero(self):
        # Mean growth rate is 0 → CV undefined → None
        result = da._volatility([5.0, 5.0, 5.0, 5.0])
        assert result is None or result == 0.0

    def test_volatile_series_positive(self):
        result = da._volatility([1.0, 10.0, 1.0, 10.0, 1.0])
        assert result is not None and result > 0

    def test_non_negative(self):
        result = da._volatility([1.0, 2.0, 3.0, 4.0, 5.0])
        if result is not None:
            assert result >= 0.0


class TestEmergenceIndex:
    def test_too_short_returns_none(self):
        assert da._emergence_index([1.0] * 4) is None

    def test_zero_series_returns_none(self):
        assert da._emergence_index([0.0] * 8) is None

    def test_bounded_0_1(self):
        for series in (
            [1.0] * 4 + [10.0] * 4,
            [10.0] * 4 + [1.0] * 4,
            [5.0] * 8,
        ):
            result = da._emergence_index(series)
            if result is not None:
                assert 0.0 <= result <= 1.0

    def test_recent_surge_higher_than_flat(self):
        flat    = [5.0] * 8
        surging = [1.0] * 4 + [5.0] * 4
        assert da._emergence_index(surging) > da._emergence_index(flat)

    def test_decline_lower_than_flat(self):
        flat     = [5.0] * 8
        dropping = [5.0] * 4 + [1.0] * 4
        ei_flat = da._emergence_index(flat)
        ei_drop = da._emergence_index(dropping)
        if ei_flat is not None and ei_drop is not None:
            assert ei_drop <= ei_flat


class TestRgi:
    def test_none_cagr_returns_none(self):
        assert da._rgi(None, 5.0) is None

    def test_zero_sector_returns_none(self):
        assert da._rgi(10.0, 0.0) is None

    def test_double_sector_average(self):
        assert da._rgi(10.0, 5.0) == pytest.approx(2.0, abs=0.001)

    def test_underperformance_below_one(self):
        assert da._rgi(3.0, 10.0) < 1.0

    def test_parity_is_one(self):
        assert da._rgi(7.0, 7.0) == pytest.approx(1.0, abs=0.001)


class TestMinmax:
    def test_at_minimum(self):
        assert da._minmax(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_at_maximum(self):
        assert da._minmax(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_midpoint(self):
        assert da._minmax(5.0, 0.0, 10.0) == pytest.approx(0.5)

    def test_none_returns_zero(self):
        assert da._minmax(None, 0.0, 10.0) == 0.0

    def test_equal_range_returns_half(self):
        assert da._minmax(5.0, 5.0, 5.0) == pytest.approx(0.5)

    def test_clips_above_max(self):
        assert da._minmax(999.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_clips_below_min(self):
        assert da._minmax(-999.0, 0.0, 10.0) == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════
#  2.  SHORT-TERM ANALYSIS  —  CPS & CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

_BASE_CPS_KWARGS = dict(
    hist_cagr=5.0, fore_cagr=8.0, rgi=1.5, ei=0.7, volatility=0.3,
    all_hist_cagrs=[3.0, 5.0, 7.0],
    all_fore_cagrs=[5.0, 8.0, 10.0],
    all_rgis=[0.8, 1.5, 2.0],
    all_vols=[0.1, 0.3, 0.8],
)


class TestComputeCps:
    def test_result_in_unit_interval(self):
        score = da.compute_cps(**_BASE_CPS_KWARGS)
        assert 0.0 <= score <= 1.0

    def test_strong_metrics_score_above_half(self):
        score = da.compute_cps(
            hist_cagr=20.0, fore_cagr=25.0, rgi=3.0, ei=0.9, volatility=0.05,
            all_hist_cagrs=[0.0, 5.0, 20.0],
            all_fore_cagrs=[0.0, 10.0, 25.0],
            all_rgis=[0.2, 1.0, 3.0],
            all_vols=[0.05, 0.3, 1.5],
        )
        assert score >= 0.5

    def test_weak_metrics_score_below_half(self):
        score = da.compute_cps(
            hist_cagr=-10.0, fore_cagr=-15.0, rgi=0.1, ei=0.05, volatility=3.0,
            all_hist_cagrs=[-10.0, 0.0, 5.0],
            all_fore_cagrs=[-15.0, 0.0, 8.0],
            all_rgis=[0.1, 1.0, 2.0],
            all_vols=[0.5, 1.5, 3.0],
        )
        assert score <= 0.5

    def test_all_none_inputs_handled_gracefully(self):
        score = da.compute_cps(
            hist_cagr=None, fore_cagr=None, rgi=None, ei=None, volatility=None,
            all_hist_cagrs=[5.0], all_fore_cagrs=[8.0],
            all_rgis=[1.0], all_vols=[0.3],
        )
        assert 0.0 <= score <= 1.0

    def test_single_element_norm_lists_no_crash(self):
        """vmin == vmax → _minmax returns 0.5; must not crash or produce NaN."""
        score = da.compute_cps(
            hist_cagr=5.0, fore_cagr=8.0, rgi=1.5, ei=0.7, volatility=0.3,
            all_hist_cagrs=[5.0], all_fore_cagrs=[8.0],
            all_rgis=[1.5], all_vols=[0.3],
        )
        assert 0.0 <= score <= 1.0
        assert not math.isnan(score)

    def test_empty_norm_lists_no_crash(self):
        score = da.compute_cps(
            hist_cagr=5.0, fore_cagr=8.0, rgi=1.5, ei=0.7, volatility=0.3,
            all_hist_cagrs=[], all_fore_cagrs=[], all_rgis=[], all_vols=[],
        )
        assert 0.0 <= score <= 1.0

    def test_custom_weights_sum_to_one_is_fine(self):
        weights = {"forecast_cagr": 0.40, "hist_cagr": 0.25,
                   "rgi": 0.20, "ei": 0.10, "stability": 0.05}
        score = da.compute_cps(**_BASE_CPS_KWARGS, weights=weights)
        assert 0.0 <= score <= 1.0


class TestClassifyPotential:
    @pytest.mark.parametrize("cps,expected_tier", [
        (0.00, "low"),
        (0.34, "low"),
        (0.35, "medium"),
        (0.50, "medium"),
        (0.64, "medium"),
        (0.65, "high"),
        (1.00, "high"),
    ])
    def test_tier_boundaries(self, cps, expected_tier):
        assert da.classify_potential(cps) == expected_tier


# ══════════════════════════════════════════════════════════════════
#  3.  LONG-TERM EMERGE FRAMEWORK
# ══════════════════════════════════════════════════════════════════

class TestIrtP:
    def test_at_difficulty_is_half(self):
        assert da._irt_p(0.5, a=1.0, b=0.5) == pytest.approx(0.5, abs=1e-6)

    def test_above_difficulty_above_half(self):
        assert da._irt_p(0.9, a=1.0, b=0.5) > 0.5

    def test_below_difficulty_below_half(self):
        assert da._irt_p(0.1, a=1.0, b=0.5) < 0.5

    def test_bounded_0_1(self):
        for theta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = da._irt_p(theta, a=2.0, b=0.5)
            assert 0.0 <= p <= 1.0

    def test_large_positive_theta_no_overflow(self):
        assert da._irt_p(1000.0, a=10.0, b=0.0) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_theta_no_overflow(self):
        assert da._irt_p(-1000.0, a=10.0, b=0.0) == pytest.approx(0.0, abs=1e-6)

    def test_higher_discrimination_steeper_curve(self):
        """Higher 'a' → bigger gap between theta 0.4 and 0.6."""
        gap_lo = da._irt_p(0.6, a=1.0, b=0.5) - da._irt_p(0.4, a=1.0, b=0.5)
        gap_hi = da._irt_p(0.6, a=5.0, b=0.5) - da._irt_p(0.4, a=5.0, b=0.5)
        assert gap_hi > gap_lo


class TestEstimateTheta:
    def test_zero_signals_low_theta(self):
        theta, _ = da.estimate_theta({k: 0.0 for k in da.IRT_PARAMS})
        assert theta < 0.5

    def test_high_signals_high_theta(self):
        theta, _ = da.estimate_theta({k: 1.0 for k in da.IRT_PARAMS})
        assert theta > 0.5

    def test_theta_in_unit_interval(self):
        for val in [0.0, 0.3, 0.7, 1.0]:
            theta, _ = da.estimate_theta({k: val for k in da.IRT_PARAMS})
            assert 0.0 <= theta <= 1.0

    def test_confidence_in_unit_interval(self):
        _, conf = da.estimate_theta({k: 0.5 for k in da.IRT_PARAMS})
        assert 0.0 <= conf <= 1.0

    def test_empty_signals_returns_fallback(self):
        theta, conf = da.estimate_theta({})
        assert isinstance(theta, float)
        assert isinstance(conf, float)
        assert 0.0 < theta < 1.0

    def test_all_zero_signals_returns_fallback(self):
        theta, conf = da.estimate_theta({k: 0.0 for k in da.IRT_PARAMS})
        assert 0.0 < theta < 1.0   # fallback, not 0 or crash

    def test_monotone_in_signal_strength(self):
        """Theta should increase as all signals increase."""
        theta_low,  _ = da.estimate_theta({k: 0.1 for k in da.IRT_PARAMS})
        theta_high, _ = da.estimate_theta({k: 0.9 for k in da.IRT_PARAMS})
        assert theta_high > theta_low


class TestFuzzyMemberships:
    def test_all_four_categories_present(self):
        m = da.fuzzy_memberships(0.5)
        assert set(m) == {"speculative", "niche", "emerging", "breakthrough"}

    def test_all_values_in_0_1(self):
        for theta in [0.05, 0.3, 0.5, 0.7, 0.95]:
            for v in da.fuzzy_memberships(theta).values():
                assert 0.0 <= v <= 1.0

    def test_theta_zero_speculative_dominant(self):
        m = da.fuzzy_memberships(0.0)
        assert m["speculative"] >= m["breakthrough"]
        assert m["breakthrough"] == pytest.approx(0.0, abs=1e-4)

    def test_theta_one_breakthrough_dominant(self):
        m = da.fuzzy_memberships(1.0)
        assert m["breakthrough"] >= m["speculative"]

    def test_theta_mid_has_nonzero_membership(self):
        m = da.fuzzy_memberships(0.5)
        assert sum(m.values()) > 0


class TestDominantCategory:
    def test_returns_highest_membership(self):
        m = {"speculative": 0.1, "niche": 0.9, "emerging": 0.3, "breakthrough": 0.2}
        assert da.dominant_category(m) == "niche"

    def test_single_nonzero(self):
        m = {"speculative": 0.0, "niche": 0.0, "emerging": 0.8, "breakthrough": 0.0}
        assert da.dominant_category(m) == "emerging"

    def test_returns_a_key_in_input(self):
        m = {k: 0.5 for k in ("speculative", "niche", "emerging", "breakthrough")}
        assert da.dominant_category(m) in m


class TestTimeToEmergence:
    def test_has_required_keys(self):
        tte = da.time_to_emergence(0.5, 0.8)
        assert {"point_estimate_years", "ci_lower_years", "ci_upper_years"} <= set(tte)

    def test_higher_theta_lower_tte(self):
        tte_hi = da.time_to_emergence(0.9, 0.9)
        tte_lo = da.time_to_emergence(0.1, 0.9)
        assert tte_hi["point_estimate_years"] < tte_lo["point_estimate_years"]

    def test_ci_correctly_ordered(self):
        tte = da.time_to_emergence(0.5, 0.7)
        assert tte["ci_lower_years"] <= tte["point_estimate_years"] <= tte["ci_upper_years"]

    def test_never_exceeds_t_max(self):
        for theta in [0.05, 0.5, 0.95]:
            tte = da.time_to_emergence(theta, 0.8)
            assert tte["point_estimate_years"] <= 5.0
            assert tte["ci_upper_years"]       <= 5.0

    def test_always_positive(self):
        for theta in [0.05, 0.5, 0.95]:
            tte = da.time_to_emergence(theta, 0.9)
            assert tte["point_estimate_years"] >= 0.2
            assert tte["ci_lower_years"]       >= 0.2

    def test_low_confidence_widens_ci(self):
        tte_conf  = da.time_to_emergence(0.5, 0.95)
        tte_unconf = da.time_to_emergence(0.5, 0.10)
        width_conf   = tte_conf["ci_upper_years"]   - tte_conf["ci_lower_years"]
        width_unconf = tte_unconf["ci_upper_years"] - tte_unconf["ci_lower_years"]
        assert width_unconf >= width_conf


class TestIsRecent:
    def test_within_window_true(self):
        d = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
        assert da._is_recent(d, years=1)

    def test_outside_window_false(self):
        d = (datetime.now() - timedelta(days=800)).strftime("%Y-%m-%d")
        assert not da._is_recent(d, years=1)

    def test_none_false(self):
        assert not da._is_recent(None)

    def test_empty_string_false(self):
        assert not da._is_recent("")

    def test_exactly_on_boundary(self):
        d = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        # Result may be True or False depending on exact time-of-day; just test no crash
        assert isinstance(da._is_recent(d, years=1), bool)


class TestYoyGrowthSignal:
    def test_empty_list_zero(self):
        assert da._yoy_growth_signal([]) == 0.0

    def test_result_in_unit_interval(self):
        dates = [(datetime.now() - timedelta(days=i * 60)).strftime("%Y-%m-%d")
                 for i in range(20)]
        assert 0.0 <= da._yoy_growth_signal(dates) <= 1.0

    def test_all_recent_high_value(self):
        recent = [(datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")] * 10
        assert da._yoy_growth_signal(recent) >= 0.5

    def test_all_none_dates_zero(self):
        assert da._yoy_growth_signal([None] * 5) == 0.0


class TestComputeJobSignals:
    def test_no_matching_skill_all_zeros(self):
        items = [{"skills": ["other"], "upload_date": "2023-01-01"}]
        sigs = da.compute_job_signals("missing_skill", items, 1)
        assert all(v == 0.0 for v in sigs.values())

    def test_returns_all_job_irt_param_keys(self):
        item = {
            "skills":        ["s1"],
            "upload_date":   (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "country":       "DE",
            "sectors":       ["J"],
            "occupation_id": "occ1",
        }
        sigs = da.compute_job_signals("s1", [item], 10)
        assert set(sigs) == set(da.JOB_IRT_PARAMS)

    def test_all_values_bounded_0_1(self):
        items = [{
            "skills":      ["s1"],
            "upload_date": (datetime.now() - timedelta(days=30 * i)).strftime("%Y-%m-%d"),
            "country":     ["DE", "FR", "IT", "ES", "PL"][i % 5],
            "sectors":     [["J", "K", "L", "M", "N"][i % 5]],
            "occupation_id": f"occ{i}",
        } for i in range(20)]
        sigs = da.compute_job_signals("s1", items, 20)
        for k, v in sigs.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v} out of bounds"

    def test_geo_spread_increases_with_country_diversity(self):
        same = [{"skills": ["s1"], "upload_date": "2023-01-01", "country": "DE"}] * 5
        diverse = [{"skills": ["s1"], "upload_date": "2023-01-01",
                    "country": ["DE", "FR", "IT", "ES", "PL"][i]} for i in range(5)]
        assert (da.compute_job_signals("s1", diverse, 5)["geo_spread"] >=
                da.compute_job_signals("s1", same,    5)["geo_spread"])

    def test_cross_sector_increases_with_sector_diversity(self):
        same = [{"skills": ["s1"], "upload_date": "2023-01-01", "sectors": ["J"]}] * 5
        div  = [{"skills": ["s1"], "upload_date": "2023-01-01",
                 "sectors": [["J", "K", "L", "M", "N"][i]]} for i in range(5)]
        assert (da.compute_job_signals("s1", div,  5)["cross_sector_adoption"] >=
                da.compute_job_signals("s1", same, 5)["cross_sector_adoption"])


class TestComputeSignals:
    def _doc(self, n=5, skill="s1"):
        return [{
            "skills":           [skill],
            "publication_date": (datetime.now() - timedelta(days=i * 60)).strftime("%Y-%m-%d"),
            "location_code":    ["DE", "FR"][i % 2],
            "nace_code":        ["J", "K"][i % 2],
        } for i in range(n)]

    def test_returns_all_irt_param_keys(self):
        src = {"policies": self._doc(), "projects": [], "white_papers": []}
        sigs = da.compute_signals("s1", src, {})
        assert set(sigs) == set(da.IRT_PARAMS)

    def test_all_values_bounded_0_1(self):
        src = {k: self._doc(10) for k in ("policies", "projects", "white_papers")}
        sigs = da.compute_signals("s1", src, {"s1": 30})
        for k, v in sigs.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v}"

    def test_no_matching_skill_all_zeros(self):
        src = {"policies": self._doc(skill="other"), "projects": [], "white_papers": []}
        sigs = da.compute_signals("s1", src, {})
        assert all(v == 0.0 for v in sigs.values())

    def test_empty_all_sources_all_zeros(self):
        sigs = da.compute_signals("s1", {}, {})
        for k in da.IRT_PARAMS:
            assert sigs.get(k, 0.0) == 0.0


# ══════════════════════════════════════════════════════════════════
#  4.  LLM HELPERS
# ══════════════════════════════════════════════════════════════════

class TestStripCodeFences:
    def test_json_fence_removed(self):
        assert da._strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'

    def test_plain_fence_removed(self):
        assert da._strip_code_fences('```\n{"a":1}\n```') == '{"a":1}'

    def test_no_fence_unchanged(self):
        s = '{"a":1}'
        assert da._strip_code_fences(s) == s

    def test_result_is_stripped(self):
        assert da._strip_code_fences('  {"a":1}  ') == '{"a":1}'


class TestParseLlmJson:
    def test_clean_json(self):
        assert da._parse_llm_json('{"k":"v"}') == {"k": "v"}

    def test_code_fenced_json(self):
        assert da._parse_llm_json('```json\n{"k":"v"}\n```') == {"k": "v"}

    def test_trailing_garbage(self):
        assert da._parse_llm_json('{"k":"v"} some trailing text') == {"k": "v"}

    def test_preamble_and_postamble(self):
        assert da._parse_llm_json('Preamble {"k":"v"} postamble') == {"k": "v"}

    def test_nested_object(self):
        assert da._parse_llm_json('{"a":{"b":[1,2]}}') == {"a": {"b": [1, 2]}}

    def test_empty_object(self):
        assert da._parse_llm_json("{}") == {}

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            da._parse_llm_json("totally not json!!!")

    def test_raises_on_empty_string(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            da._parse_llm_json("")


# ══════════════════════════════════════════════════════════════════
#  5.  PIPELINE INTEGRATION
# ══════════════════════════════════════════════════════════════════

class TestRunShortTermAnalysis:
    def test_skills_mode_structure(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis(sample_items, "skills", label_dict, top_n=5)
        assert {"skills", "metadata", "sector_summary"} <= set(r)

    def test_occupations_mode_structure(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis(sample_items, "occupations", label_dict, top_n=5)
        assert "occupations" in r

    def test_results_sorted_by_cps_descending(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis(sample_items, "skills", label_dict, top_n=10)
        scores = [e["metrics"]["composite_potential_score"] for e in r["skills"]]
        assert scores == sorted(scores, reverse=True)

    def test_empty_items_returns_empty_results(self, label_dict):
        r = da.run_short_term_analysis([], "skills", label_dict, top_n=5)
        assert r["skills"] == []

    def test_sector_summary_counts_consistent(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis(sample_items, "skills", label_dict, top_n=10)
        ss = r["sector_summary"]
        assert (ss["high_potential_count"] + ss["medium_potential_count"] +
                ss["low_potential_count"]) == ss["total_entities_analyzed"]

    def test_each_entity_has_recommendation_keys(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis(sample_items, "skills", label_dict, top_n=3)
        for entity in r["skills"]:
            recs = entity["recommendations"]
            assert {"talent_acquisition",
                    "training_and_development",
                    "compensation_and_retention"} <= set(recs)

    def test_metadata_analysis_type(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis(sample_items, "skills", label_dict, top_n=3)
        assert r["metadata"]["analysis_type"] == "short_term_skills"

    def test_top_n_respected(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis(sample_items, "skills", label_dict, top_n=2)
        assert len(r["skills"]) <= 2

    def test_each_entity_has_time_series(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis(sample_items, "skills", label_dict, top_n=3)
        for entity in r["skills"]:
            ts = entity["time_series"]
            assert "historical" in ts and "forecast" in ts and "forecast_method" in ts


class TestRunShortTermAnalysisBySector:
    def test_returns_dict_with_entries(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis_by_sector(
                sample_items, "skills", label_dict, top_n=5, user_sector="IT"
            )
        assert isinstance(r, dict) and len(r) > 0

    def test_sector_metadata_overridden_with_user_sector(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis_by_sector(
                sample_items, "skills", label_dict, top_n=5, user_sector="MyCustomSector"
            )
        for analysis in r.values():
            assert analysis["metadata"]["sector"] == "MyCustomSector"

    def test_sector_item_count_in_metadata(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_short_term_analysis_by_sector(
                sample_items, "skills", label_dict, top_n=5
            )
        for analysis in r.values():
            assert "sector_item_count" in analysis["metadata"]


class TestRunLongTermSkillsFromJobs:
    def test_structure(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=5)
        assert {"skills", "metadata", "sector_summary"} <= set(r)

    def test_empty_items_graceful(self, label_dict):
        r = da.run_long_term_skills_from_jobs([], label_dict)
        assert r["sector_summary"]["total_entities_analyzed"] == 0

    def test_sorted_by_theta_descending(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=10)
        thetas = [s["theta"] for s in r["skills"]]
        assert thetas == sorted(thetas, reverse=True)

    def test_theta_in_unit_interval(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=10)
        for s in r["skills"]:
            assert 0.0 <= s["theta"] <= 1.0

    def test_required_skill_fields_present(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=5)
        required = {
            "uri", "label", "emergence_quotient", "theta", "confidence",
            "time_to_emergence", "dominant_category", "fuzzy_memberships",
            "irt_signals", "total_job_mentions", "recommendations",
        }
        for s in r["skills"]:
            assert required <= set(s)

    def test_top_n_respected(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=2)
        assert len(r["skills"]) <= 2

    def test_metadata_framework_field(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=3)
        assert "EMERGE" in r["metadata"]["framework"]


class TestRunLongTermOccupationsFromJobs:
    def test_structure(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_occupations_from_jobs(
                sample_items, label_dict, top_n=5, sector="J"
            )
        assert {"occupations", "metadata", "sector_summary"} <= set(r)

    def test_top_n_respected(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_occupations_from_jobs(
                sample_items, label_dict, top_n=2
            )
        assert len(r["occupations"]) <= 2

    def test_sorted_by_theta_descending(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_occupations_from_jobs(
                sample_items, label_dict, top_n=10
            )
        thetas = [o["theta"] for o in r["occupations"]]
        assert thetas == sorted(thetas, reverse=True)

    def test_empty_items_graceful(self, label_dict):
        r = da.run_long_term_occupations_from_jobs([], label_dict)
        assert r["sector_summary"]["total_occupations_analyzed"] == 0

    def test_required_occupation_fields(self, sample_items, label_dict):
        with patch.object(da, "_chat_llm_json", return_value=None):
            r = da.run_long_term_occupations_from_jobs(
                sample_items, label_dict, top_n=5
            )
        for occ in r["occupations"]:
            for field in ("uri", "label", "theta", "dominant_category", "recommendations"):
                assert field in occ


# ══════════════════════════════════════════════════════════════════
#  6.  KNOWN BUGS — regression documentation
# ══════════════════════════════════════════════════════════════════

class TestKnownBugs:
    """
    These tests exercise code paths that are *currently broken*.
    They document existing bugs so they are not silently reintroduced.
    Expected to fail on the unpatched source; pass once fixed.
    """

    def test_bug1_sector_fields_is_string_not_tuple(self):
        """
        BUG-1: _SECTOR_FIELDS = ("sectors") creates a string, not a tuple.
        Iterating over it yields individual characters 's','e','c',…
        so _extract_sectors() can never find the 'sectors' key and always
        returns ["unknown"].

        Intended behaviour: extracting from {"sectors": ["J"]} returns ["J"].
        """
        result = da._extract_sectors({"sectors": ["J", "K"]})
        # This assertion should pass AFTER the fix:
        assert result == ["J", "K"], (
            "BUG-1 still present: _SECTOR_FIELDS is a string, not a tuple. "
            "Change to _SECTOR_FIELDS = ('sectors',) to fix."
        )

    def test_bug2_run_long_term_skills_sector_nameerror(self, label_dict):
        """
        BUG-2: run_long_term_skills() uses `sector` as a variable but it is not
        declared in the function's signature or body.
        Calling it with non-empty data raises NameError.

        Fix: add  sector: str = "Unknown"  to the function signature.
        """
        items_by_source = {
            "policies":    _make_items(5),
            "projects":    [],
            "white_papers": [],
        }
        with patch.object(da, "_chat_llm_json", return_value=None):
            with pytest.raises(NameError):
                da.run_long_term_skills(items_by_source, label_dict, top_n=2)


# ══════════════════════════════════════════════════════════════════
#  7.  FASTAPI ENDPOINTS
# ══════════════════════════════════════════════════════════════════

_ESCO_DF   = _fake_esco_df()
_LABEL_DICT = _fake_label_dict()


class TestShortTermSkillsEndpoint:
    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_200_with_data(self, _llm, _pag, _esco, tmp_cache):
        assert client.get("/shorttermanalysis/skills").status_code == 200

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_response_contains_metadata_and_results(self, _llm, _pag, _esco, tmp_cache):
        body = client.get("/shorttermanalysis/skills").json()
        assert "metadata" in body and "results_by_sector" in body

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=[])
    def test_no_data_returns_no_data_status(self, _pag, _esco, tmp_cache):
        assert client.get("/shorttermanalysis/skills").json()["status"] == "no_data"

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_sector_query_param_forwarded(self, _llm, mock_pag, _esco, tmp_cache):
        client.get("/shorttermanalysis/skills?sector=J")
        assert mock_pag.call_args[0][0].get("sectors") == "J"

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_organization_query_param_forwarded(self, _llm, mock_pag, _esco, tmp_cache):
        client.get("/shorttermanalysis/skills?organization=AcmeCorp")
        assert mock_pag.call_args[0][0].get("organization_names") == "AcmeCorp"

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_second_identical_request_uses_cache(self, _llm, mock_pag, _esco, tmp_cache):
        client.get("/shorttermanalysis/skills?sector=J")
        mock_pag.reset_mock()
        client.get("/shorttermanalysis/skills?sector=J")
        mock_pag.assert_not_called()

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_different_sector_hits_api_again(self, _llm, mock_pag, _esco, tmp_cache):
        client.get("/shorttermanalysis/skills?sector=J")
        mock_pag.reset_mock()
        client.get("/shorttermanalysis/skills?sector=K")  # different sector
        mock_pag.assert_called_once()

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_filters_applied_field_in_output(self, _llm, _pag, _esco, tmp_cache):
        body = client.get("/shorttermanalysis/skills?sector=J&organization=Acme").json()
        assert body.get("filters_applied", {}).get("sector") == "J"


class TestShortTermOccupationsEndpoint:
    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_200_with_data(self, _llm, _pag, _esco, tmp_cache):
        assert client.get("/shorttermanalysis/occupations").status_code == 200

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=[])
    def test_no_data_status(self, _pag, _esco, tmp_cache):
        assert client.get("/shorttermanalysis/occupations").json()["status"] == "no_data"

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_sector_forwarded_to_paginate(self, _llm, mock_pag, _esco, tmp_cache):
        client.get("/shorttermanalysis/occupations?sector=K")
        assert mock_pag.call_args[0][0].get("sectors") == "K"

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_result_cached(self, _llm, mock_pag, _esco, tmp_cache):
        client.get("/shorttermanalysis/occupations?sector=K")
        mock_pag.reset_mock()
        client.get("/shorttermanalysis/occupations?sector=K")
        mock_pag.assert_not_called()


class TestLongTermSkillsEndpoint:
    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=[])
    def test_no_data_status(self, _pag, _esco, tmp_cache):
        assert client.get("/longtermanalysis/skills").json()["status"] == "no_data"

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(8))
    @patch.object(da, "run_long_term_skills",
                  side_effect=lambda *a, **k: _make_lt_skills_output())
    def test_200_with_mocked_pipeline(self, mock_lt, _pag, _esco, tmp_cache):
        """
        run_long_term_skills is mocked to bypass BUG-2 (NameError on `sector`).
        Remove the mock once BUG-2 is fixed.
        """
        resp = client.get("/longtermanalysis/skills")
        assert resp.status_code == 200
        mock_lt.assert_called_once()

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(8))
    @patch.object(da, "run_long_term_skills",
                  side_effect=lambda *a, **k: _make_lt_skills_output())
    def test_keywords_added_to_metadata(self, _lt, _pag, _esco, tmp_cache):
        body = client.get("/longtermanalysis/skills?keywords=python,ml").json()
        assert body["metadata"].get("keywords") is not None

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(8))
    @patch.object(da, "run_long_term_skills",
                  side_effect=lambda *a, **k: _make_lt_skills_output())
    def test_result_cached(self, _lt, mock_pag, _esco, tmp_cache):
        client.get("/longtermanalysis/skills?keywords=ai")
        mock_pag.reset_mock()
        client.get("/longtermanalysis/skills?keywords=ai")
        mock_pag.assert_not_called()


class TestLongTermOccupationsEndpoint:
    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_200_with_data(self, _llm, _pag, _esco, tmp_cache):
        assert client.get("/longtermanalysis/occupations").status_code == 200

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=[])
    def test_no_data_status(self, _pag, _esco, tmp_cache):
        assert client.get("/longtermanalysis/occupations").json()["status"] == "no_data"

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_top_n_query_param_accepted(self, _llm, _pag, _esco, tmp_cache):
        assert client.get("/longtermanalysis/occupations?top_n=5").status_code == 200

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_sector_forwarded_to_paginate(self, _llm, mock_pag, _esco, tmp_cache):
        client.get("/longtermanalysis/occupations?sector=J")
        assert mock_pag.call_args[0][0].get("sectors") == "J"

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_result_cached(self, _llm, mock_pag, _esco, tmp_cache):
        client.get("/longtermanalysis/occupations?sector=J")
        mock_pag.reset_mock()
        client.get("/longtermanalysis/occupations?sector=J")
        mock_pag.assert_not_called()

    @patch.object(da, "load_esco_mapping", return_value=(_ESCO_DF, _LABEL_DICT))
    @patch.object(da, "paginate_all",      return_value=_make_items(12))
    @patch.object(da, "_chat_llm_json",    return_value=None)
    def test_output_has_occupations_key(self, _llm, _pag, _esco, tmp_cache):
        body = client.get("/longtermanalysis/occupations").json()
        assert "occupations" in body or body.get("status") == "no_data"


# ══════════════════════════════════════════════════════════════════
#  8.  EDGE CASES & REGRESSION
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_forecast_single_point_no_crash(self):
        result = da.forecast_series([42.0], 4)
        assert len(result["forecast"]) == 4

    def test_quarters_back_cross_year_boundary(self):
        labels = da._quarters_back(4)
        years = {lbl[:4] for lbl in labels}
        assert len(years) <= 2

    def test_build_skill_series_year_boundary(self):
        items = [
            {"upload_date": "2022-12-31", "skills": ["s1"]},
            {"upload_date": "2023-01-01", "skills": ["s1"]},
        ]
        series = da._build_skill_series(items)
        assert "2022-Q4" in series["s1"]
        assert "2023-Q1" in series["s1"]

    def test_emergence_index_all_zeros_returns_none(self):
        assert da._emergence_index([0.0] * 8) is None

    def test_tte_minimum_clamp(self):
        tte = da.time_to_emergence(0.99, 0.99)
        assert tte["point_estimate_years"] >= 0.2

    def test_irt_p_exactly_at_difficulty(self):
        for b in [0.2, 0.5, 0.8]:
            assert da._irt_p(b, a=2.0, b=b) == pytest.approx(0.5, abs=1e-6)

    def test_fuzzy_at_theta_zero_speculative_is_one(self):
        m = da.fuzzy_memberships(0.0)
        assert m["speculative"] == pytest.approx(1.0, abs=1e-4)
        assert m["breakthrough"] == pytest.approx(0.0, abs=1e-4)

    def test_cps_no_nan_with_empty_norm_lists(self):
        score = da.compute_cps(
            hist_cagr=5.0, fore_cagr=8.0, rgi=1.5, ei=0.7, volatility=0.3,
            all_hist_cagrs=[], all_fore_cagrs=[], all_rgis=[], all_vols=[],
        )
        assert not math.isnan(score)
        assert 0.0 <= score <= 1.0

    def test_parse_llm_json_empty_curly(self):
        assert da._parse_llm_json("{}") == {}

    def test_demand_velocity_exactly_three_elements(self):
        result = da._demand_velocity([5.0, 6.0, 7.0])
        assert result is not None

    def test_cagr_equal_start_end_zero(self):
        assert abs(da._cagr([10.0, 10.0], 1.0)) < 0.001

    def test_linear_forecast_ci_monotone_widening(self):
        pts = da._linear_forecast([2.0, 4.0, 6.0, 8.0, 10.0], 5)["forecast"]
        widths = [p["ci_upper_95"] - p["ci_lower_95"] for p in pts]
        for w1, w2 in zip(widths, widths[1:]):
            assert w2 >= w1

    def test_fill_series_empty_labels(self):
        result = da._fill_series({"2023-Q1": 5}, [])
        assert result == []

    def test_compute_job_signals_empty_item_list(self):
        sigs = da.compute_job_signals("s1", [], 0)
        assert all(v == 0.0 for v in sigs.values())

    def test_yoy_growth_signal_none_dates_in_list(self):
        dates = [None, None, None]
        assert da._yoy_growth_signal(dates) == 0.0

    def test_short_term_recommendations_static_fallback(self, label_dict):
        """When LLM fails, static recommendations must cover all three keys."""
        with patch.object(da, "_chat_llm_json", return_value=None):
            recs = da._short_term_recommendations(
                tier="high",
                entity_label="Python",
                metrics={
                    "historical_cagr_pct": 12.0,
                    "forecast_cagr_pct": 15.0,
                    "demand_velocity_pct": 5.0,
                    "market_penetration_rate_pct": 8.0,
                    "demand_volatility": 0.4,
                    "relative_growth_index": 1.8,
                    "emergence_index": 0.7,
                    "composite_potential_score": 0.8,
                },
            )
        assert {"talent_acquisition",
                "training_and_development",
                "compensation_and_retention"} <= set(recs)

    def test_long_term_recommendations_static_fallback(self):
        """When LLM fails, static LT recommendations must cover all three keys."""
        tte = da.time_to_emergence(0.6, 0.8)
        sigs = {k: 0.3 for k in da.IRT_PARAMS}
        with patch.object(da, "_chat_llm_json", return_value=None):
            recs = da._long_term_recommendations(
                "emerging", "Machine Learning", 0.6, tte, sigs
            )
        assert {"strategic_workforce_planning",
                "partnerships_and_pipeline",
                "regulatory_and_compliance"} <= set(recs)