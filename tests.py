"""
test_demand_analysis.py
=====================================================================
Pytest suite for the SKILLAB Demand Analysis service
=====================================================================

Every endpoint takes the X-User-Organization header, runs in the background
(first call -> 202 "started", then "in_progress", then the result) and returns two analyses:
  organization_analysis — job ads from the Hiring Management API
  sector_analysis       — sectors from the Employee Management API,
                          then job postings per sector from the SKILLAB Tracker

No test touches the network: the portal APIs, the Tracker, the ESCO file
and the LLM are all replaced by in-memory fakes.

Run with:
    pytest test_demand_analysis.py -v
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── module under test ────────────────────────────────────────────
# Adjust the import if your module has a different name.
import service_new as da
from service_new import app

client = TestClient(app)

# Fail fast with a clear message if `service` is an older version of the module
_REQUIRED = ["_run_combined", "fetch_org_profile", "fetch_org_jobs", "fetch_sector_jobs", "_portal_get",
             "_clean_sectors", "_available_signals", "_occupations_of", "_long_term_job_recommendations",
             "HIRING_API", "EMPLOYEE_API", "PORTAL_API_TOKEN"]
_missing = [name for name in _REQUIRED if not hasattr(da, name)]
if _missing:
    pytest.exit(f"`service` is not the current demand-analysis module; missing: {', '.join(_missing)}",
                returncode=2)

# Fixed URLs so the tests do not depend on the .env of the machine running them
TEST_HIRING_API   = "https://test.local/hiring-management"
TEST_EMPLOYEE_API = "https://test.local/employee-management"


# ══════════════════════════════════════════════════════════════════
#  SHARED HELPERS & FIXTURES
# ══════════════════════════════════════════════════════════════════

SKILL_URIS = [f"http://data.europa.eu/esco/skill/{i}" for i in range(6)]
OCC_URIS   = [f"http://data.europa.eu/esco/occupation/{i}" for i in range(3)]


def _days_ago(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


def _make_items(n: int = 15) -> list[dict]:
    """Tracker-style jobs spread over the last ~2.5 years (inside the 12-quarter window)."""
    return [{
        "upload_date": _days_ago(i * 60),
        "skills":      SKILL_URIS[: (i % 3) + 1],
        "occupations": [OCC_URIS[i % 3]],
        "sectors":     [["J", "K"][i % 2]],
        "country":     ["DE", "FR", "IT"][i % 3],
    } for i in range(n)]


def _label_dict() -> dict[str, str]:
    d = {u: f"Skill {i}" for i, u in enumerate(SKILL_URIS)}
    d.update({u: f"Occupation {i}" for i, u in enumerate(OCC_URIS)})
    return d


_ESCO_DF = pd.DataFrame({"conceptUri": list(_label_dict()), "preferredLabel": list(_label_dict().values())})


class _Resp:
    def __init__(self, data=None, status=200):
        self._data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


class FakePortal:
    """
    In-memory Hiring Management + Employee Management APIs.
    Records every GET so tests can check URLs and headers.
    """

    def __init__(self, n_ads: int = 12):
        self.organizations = [
            {"id": 7, "name": "Eclipse Foundation", "location": "Brussels",
             "sectors": ['"Computer programming activities"', '"Research"'], "objectives": []},
            {"id": 8, "name": "Eclipse", "location": "Athens",
             "sectors": ["Exact match sector"], "objectives": []},
            {"id": 9, "name": "No Sector Org", "location": "Rome", "sectors": [], "objectives": []},
        ]
        occs = ["software developer", "data analyst", "ict project manager"]
        self.ads = [{"id": i, "jobTitle": f"Job {i}", "occupationName": occs[i % 3],
                     "status": "PUBLISHED", "departmentName": "IT"} for i in range(1, n_ads + 1)]
        self.details = {i: {"id": i, "publishDate": _days_ago(i * 70)} for i in range(1, n_ads + 1)}
        self.skills = {i: [{"id": s, "title": f"skill {s}"} for s in range(1, (i % 4) + 3)]
                       for i in range(1, n_ads + 1)}
        self.fail_hiring = False
        self.fail_employee = False
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, dict(headers or {})))
        if url == f"{TEST_EMPLOYEE_API}/organizations":
            return _Resp(status=500) if self.fail_employee else _Resp(self.organizations)
        if url.startswith(f"{TEST_HIRING_API}/"):
            if self.fail_hiring:
                return _Resp(status=500)
            path = url[len(TEST_HIRING_API):]
            if path == "/api/v1/jobAds":
                return _Resp(self.ads)
            job_id = int(path.split("/")[4])
            if path.endswith("/interview-skills"):
                return _Resp(self.skills.get(job_id, []))
            return _Resp(self.details.get(job_id, {}))
        return _Resp(status=404)


@pytest.fixture(autouse=True)
def fixed_env(monkeypatch):
    """Ignore whatever HIRING_API / EMPLOYEE_API / PORTAL_API_TOKEN the local .env sets."""
    monkeypatch.setattr(da, "HIRING_API", TEST_HIRING_API)
    monkeypatch.setattr(da, "EMPLOYEE_API", TEST_EMPLOYEE_API)
    monkeypatch.setattr(da, "PORTAL_API_TOKEN", None)
    monkeypatch.setattr(da, "TRACKER_YEARS_BACK", 3)


@pytest.fixture
def label_dict():
    return _label_dict()


@pytest.fixture
def sample_items():
    return _make_items(15)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    folder.mkdir()
    monkeypatch.setattr(da, "FOLDER", folder)
    return folder


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(da.time, "sleep", lambda *_: None)


@pytest.fixture
def no_llm(monkeypatch):
    llm = MagicMock(return_value=None)
    monkeypatch.setattr(da, "_chat_llm_json", llm)
    return llm


@pytest.fixture
def portal(monkeypatch, no_sleep):
    fake = FakePortal()
    monkeypatch.setattr(da.req, "get", fake.get)
    return fake


@pytest.fixture
def services(portal, tmp_cache, no_llm, monkeypatch):
    """Everything an endpoint needs, faked. Returns (portal, tracker_mock, llm_mock)."""
    tracker = MagicMock(side_effect=lambda body, endpoint: _make_items(12))
    monkeypatch.setattr(da, "paginate_all", tracker)
    monkeypatch.setattr(da, "load_esco_mapping", lambda: (_ESCO_DF, _label_dict()))
    return portal, tracker, no_llm


ENDPOINTS = [
    ("/shorttermanalysis/skills",       "skills"),
    ("/shorttermanalysis/occupations",  "occupations"),
    ("/longtermanalysis/skills",        "skills"),
    ("/longtermanalysis/occupations",   "occupations"),
]

ORG_HEADER = "X-User-Organization"


def _get(path, org, top_n=3):
    return client.get(f"{path}?top_n={top_n}", headers={ORG_HEADER: org})


def _analyse(path, org, top_n=3):
    """First call starts the job (TestClient runs background tasks before returning);
    the second call returns what the job produced."""
    first = _get(path, org, top_n)
    assert first.status_code == 202 and first.json()["status"] == "started"
    return _get(path, org, top_n).json()


@pytest.fixture(autouse=True)
def _reset_running_jobs():
    da._RUNNING.clear()
    yield
    da._RUNNING.clear()


# ══════════════════════════════════════════════════════════════════
#  1.  SHARED INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════

class TestCacheHelpers:
    def test_in_progress_stub(self):
        assert da._in_progress_stub() == {"status": "in_progress",
                                          "message": "Analysis is being computed", "result": None}

    def test_save_cache_unicode_safe(self, tmp_path):
        path = str(tmp_path / "unicode.json")
        da._save_cache(path, {"label": "Ανάλυση"})
        with open(path, encoding="utf-8") as f:
            assert json.load(f)["label"] == "Ανάλυση"


class TestBuildCacheKey:
    def test_none_parts_excluded(self):
        assert "None" not in da._build_cache_key("a", None, "b")

    @pytest.mark.parametrize("raw", ["sector/IT", "org name", 'with "quotes"'])
    def test_sanitized(self, raw):
        key = da._build_cache_key(raw)
        assert not any(c in key for c in '/ "')

    def test_deterministic(self):
        assert da._build_cache_key("x", "y") == da._build_cache_key("x", "y")


class TestPaginateAll:
    def test_logs_in_once_for_all_pages(self, monkeypatch):
        token = MagicMock(return_value="tok")
        monkeypatch.setattr(da, "get_token", token)
        monkeypatch.setattr(da.time, "sleep", lambda *_: None)
        pages = {1: {"count": 700, "items": [1] * 300}, 2: {"items": [2] * 300}, 3: {"items": [3] * 100}}
        extract = MagicMock(side_effect=lambda body, page, endpoint, token: pages[page])
        monkeypatch.setattr(da, "api_extract", extract)

        items = da.paginate_all({"sectors": "J"}, "jobs")

        assert len(items) == 700
        token.assert_called_once()
        assert all(c.kwargs["token"] == "tok" for c in extract.call_args_list)

    def test_single_page(self, monkeypatch):
        monkeypatch.setattr(da, "get_token", lambda: "tok")
        monkeypatch.setattr(da, "api_extract", lambda *a, **k: {"count": 2, "items": ["a", "b"]})
        assert da.paginate_all({}, "jobs") == ["a", "b"]


class TestOccupationsOf:
    @pytest.mark.parametrize("item,expected", [
        ({"occupations": "data analyst"},          ["data analyst"]),
        ({"occupations": ["u1", "u2"]},            ["u1", "u2"]),
        ({"occupation_id": "occ1"},                ["occ1"]),
        ({"occupations": [None, "", "u1"]},        ["u1"]),
        ({},                                       []),
        ({"occupations": None},                    []),
    ])
    def test_cases(self, item, expected):
        assert da._occupations_of(item) == expected


# ══════════════════════════════════════════════════════════════════
#  2.  SHORT-TERM ANALYSIS — HELPERS AND METRICS
# ══════════════════════════════════════════════════════════════════

class TestToQuarterStr:
    @pytest.mark.parametrize("date_str,expected", [
        ("2024-01-01", "2024-Q1"),
        ("2024-04-15", "2024-Q2"),
        ("2023-08-30", "2023-Q3"),
        ("2022-12-31", "2022-Q4"),
        ("2023-03-01T12:00:00", "2023-Q1"),
    ])
    def test_date_to_quarter(self, date_str, expected):
        assert da._to_quarter_str(date_str) == expected

    @pytest.mark.parametrize("bad", [None, "", "not-a-date"])
    def test_invalid_returns_none(self, bad):
        assert da._to_quarter_str(bad) is None


class TestQuarterLabels:
    def test_back_length_and_order(self):
        labels = da._quarters_back(8)
        assert len(labels) == 8
        keys = [(int(l[:4]), int(l[-1])) for l in labels]
        assert keys == sorted(keys)

    def test_back_ends_at_current_quarter(self):
        now = datetime.now()
        assert da._quarters_back(4)[-1] == f"{now.year}-Q{(now.month - 1) // 3 + 1}"

    def test_forward_all_future(self):
        now = datetime.now()
        cur = (now.year, (now.month - 1) // 3 + 1)
        for lbl in da._quarters_forward(4):
            assert (int(lbl[:4]), int(lbl[-1])) > cur


class TestSeriesBuilders:
    def test_skill_counts_per_quarter(self):
        items = [{"upload_date": "2023-01-01", "skills": ["s1", "s2"]},
                 {"upload_date": "2023-02-01", "skills": ["s1"]}]
        s = da._build_skill_series(items)
        assert s["s1"]["2023-Q1"] == 2 and s["s2"]["2023-Q1"] == 1

    def test_skill_missing_or_bad_date_skipped(self):
        assert da._build_skill_series([{"upload_date": None, "skills": ["s1"]},
                                       {"upload_date": "bad", "skills": ["s1"]}]) == {}

    def test_occupation_name_string(self):
        items = [{"upload_date": "2023-07-01", "occupations": "data analyst"}]
        assert da._build_occupation_series(items) == {"data analyst": {"2023-Q3": 1}}

    def test_occupation_list_counts_each_uri(self):
        """Tracker jobs carry a list of URIs; each must be its own key, not str(list)."""
        items = [{"upload_date": "2023-07-01", "occupations": ["u1", "u2"]}]
        s = da._build_occupation_series(items)
        assert set(s) == {"u1", "u2"}

    def test_fill_series_zeros_for_missing(self):
        assert da._fill_series({"2023-Q1": 5, "2023-Q3": 10},
                               ["2023-Q1", "2023-Q2", "2023-Q3"]) == [5.0, 0.0, 10.0]


class TestForecast:
    def test_linear_points_and_method(self):
        r = da._linear_forecast([1.0, 2.0, 3.0], 5)
        assert r["method"] == "linear_trend" and len(r["forecast"]) == 5

    def test_linear_ci_ordered_and_non_negative(self):
        for pt in da._linear_forecast([10.0, 5.0, 2.0, 1.0], 6)["forecast"]:
            assert 0.0 <= pt["ci_lower_95"] <= pt["ci_lower_80"] <= pt["value"]
            assert pt["value"] <= pt["ci_upper_80"] <= pt["ci_upper_95"]

    def test_linear_ci_widens_with_horizon(self):
        pts = da._linear_forecast([2.0, 4.0, 6.0, 8.0, 10.0], 5)["forecast"]
        widths = [p["ci_upper_95"] - p["ci_lower_95"] for p in pts]
        assert widths == sorted(widths)

    def test_short_or_zero_series_use_linear(self):
        assert da.forecast_series([1.0, 2.0, 3.0], 3)["method"] == "linear_trend"
        assert da.forecast_series([0.0] * 8, 4)["method"] == "linear_trend"

    def test_single_point_no_crash(self):
        assert len(da.forecast_series([42.0], 4)["forecast"]) == 4


class TestMetrics:
    def test_cagr_doubling_over_two_years(self):
        assert da._cagr([100.0, 200.0], 2.0) == pytest.approx(41.421, abs=0.01)

    @pytest.mark.parametrize("series", [[0.0, 5.0], [-1.0, 5.0], [10.0, -5.0], [5.0]])
    def test_cagr_undefined_returns_none(self, series):
        assert da._cagr(series, 1.0) is None

    def test_demand_velocity(self):
        assert da._demand_velocity([10.0, 11.0, 12.0]) == pytest.approx(20.0, abs=0.01)
        assert da._demand_velocity([1.0, 2.0]) is None
        assert da._demand_velocity([0.0, 1.0, 2.0]) is None

    def test_mpr(self):
        assert da._mpr([5.0, 5.0], [10.0, 10.0]) == pytest.approx(50.0, abs=0.01)
        assert da._mpr([5.0], [0.0]) is None

    def test_volatility(self):
        assert da._volatility([1.0, 2.0]) is None
        assert da._volatility([1.0, 10.0, 1.0, 10.0, 1.0]) > 0

    def test_emergence_index(self):
        assert da._emergence_index([1.0] * 4) is None
        assert da._emergence_index([0.0] * 8) is None
        assert da._emergence_index([1.0] * 4 + [5.0] * 4) > da._emergence_index([5.0] * 8)

    def test_rgi(self):
        assert da._rgi(10.0, 5.0) == pytest.approx(2.0)
        assert da._rgi(None, 5.0) is None
        assert da._rgi(10.0, 0.0) is None

    @pytest.mark.parametrize("val,lo,hi,expected", [
        (0.0, 0.0, 10.0, 0.0), (10.0, 0.0, 10.0, 1.0), (5.0, 0.0, 10.0, 0.5),
        (None, 0.0, 10.0, 0.0), (5.0, 5.0, 5.0, 0.5), (999.0, 0.0, 10.0, 1.0), (-999.0, 0.0, 10.0, 0.0),
    ])
    def test_minmax(self, val, lo, hi, expected):
        assert da._minmax(val, lo, hi) == pytest.approx(expected)


class TestCps:
    def test_bounded_and_not_nan_with_empty_lists(self):
        s = da.compute_cps(hist_cagr=5.0, fore_cagr=8.0, rgi=1.5, ei=0.7, volatility=0.3,
                           all_hist_cagrs=[], all_fore_cagrs=[], all_rgis=[], all_vols=[])
        assert 0.0 <= s <= 1.0 and not math.isnan(s)

    def test_all_none_inputs(self):
        s = da.compute_cps(hist_cagr=None, fore_cagr=None, rgi=None, ei=None, volatility=None,
                           all_hist_cagrs=[5.0], all_fore_cagrs=[8.0], all_rgis=[1.0], all_vols=[0.3])
        assert 0.0 <= s <= 1.0

    def test_strong_beats_weak(self):
        common = dict(all_hist_cagrs=[-10.0, 0.0, 20.0], all_fore_cagrs=[-15.0, 0.0, 25.0],
                      all_rgis=[0.1, 1.0, 3.0], all_vols=[0.05, 0.5, 3.0])
        strong = da.compute_cps(hist_cagr=20.0, fore_cagr=25.0, rgi=3.0, ei=0.9, volatility=0.05, **common)
        weak   = da.compute_cps(hist_cagr=-10.0, fore_cagr=-15.0, rgi=0.1, ei=0.05, volatility=3.0, **common)
        assert strong > 0.5 > weak

    @pytest.mark.parametrize("cps,tier", [
        (0.00, "low"), (0.34, "low"), (0.35, "medium"), (0.64, "medium"), (0.65, "high"), (1.00, "high"),
    ])
    def test_tiers(self, cps, tier):
        assert da.classify_potential(cps) == tier


# ══════════════════════════════════════════════════════════════════
#  3.  LONG-TERM EMERGE FRAMEWORK
# ══════════════════════════════════════════════════════════════════

ALL_KEYS = list(da.JOB_IRT_PARAMS)


class TestIrtP:
    def test_at_difficulty_is_half(self):
        assert da._irt_p(0.5, a=1.0, b=0.5) == pytest.approx(0.5)

    def test_no_overflow(self):
        assert da._irt_p(1000.0, a=10.0, b=0.0) == pytest.approx(1.0)
        assert da._irt_p(-1000.0, a=10.0, b=0.0) == pytest.approx(0.0)

    def test_higher_discrimination_is_steeper(self):
        gap = lambda a: da._irt_p(0.6, a, 0.5) - da._irt_p(0.4, a, 0.5)
        assert gap(5.0) > gap(1.0)


class TestEstimateTheta:
    def test_uses_every_job_signal(self):
        """Regression: θ used to ignore posting_density, recency_intensity and occupation_breadth."""
        base = {k: 0.4 for k in ALL_KEYS}   # mid-range, so θ is not clipped at its 0.05 floor
        for key in ("posting_density", "recency_intensity", "occupation_breadth"):
            boosted = {**base, key: 1.0}
            assert da.estimate_theta(boosted)[0] > da.estimate_theta(base)[0], key

    def test_monotone_in_signal_strength(self):
        assert (da.estimate_theta({k: 0.9 for k in ALL_KEYS})[0] >
                da.estimate_theta({k: 0.1 for k in ALL_KEYS})[0])

    def test_works_on_a_subset_of_signals(self):
        theta, conf = da.estimate_theta({"posting_density": 0.8, "yoy_growth_rate": 0.8})
        assert 0.0 < theta < 1.0 and 0.0 < conf < 1.0

    @pytest.mark.parametrize("sig", [{}, {k: 0.0 for k in ALL_KEYS}])
    def test_empty_or_zero_returns_fallback(self, sig):
        assert da.estimate_theta(sig) == (0.10, 0.15)

    def test_unknown_keys_ignored(self):
        assert da.estimate_theta({"posting_density": 0.7, "made_up": 1.0}) == \
               da.estimate_theta({"posting_density": 0.7})


class TestFuzzyAndTte:
    def test_memberships_bounded(self):
        for theta in (0.05, 0.3, 0.5, 0.7, 0.95):
            m = da.fuzzy_memberships(theta)
            assert set(m) == {"speculative", "niche", "emerging", "breakthrough"}
            assert all(0.0 <= v <= 1.0 for v in m.values())

    def test_extremes(self):
        assert da.dominant_category(da.fuzzy_memberships(0.0)) == "speculative"
        assert da.dominant_category(da.fuzzy_memberships(1.0)) == "breakthrough"

    def test_tte_ordering_and_bounds(self):
        for theta in (0.05, 0.5, 0.95):
            t = da.time_to_emergence(theta, 0.7)
            assert 0.2 <= t["ci_lower_years"] <= t["point_estimate_years"] <= t["ci_upper_years"] <= 5.0

    def test_tte_higher_theta_sooner(self):
        assert (da.time_to_emergence(0.9, 0.9)["point_estimate_years"] <
                da.time_to_emergence(0.1, 0.9)["point_estimate_years"])

    def test_low_confidence_wider_ci(self):
        w = lambda c: (lambda t: t["ci_upper_years"] - t["ci_lower_years"])(da.time_to_emergence(0.5, c))
        assert w(0.10) >= w(0.95)


class TestRecencyAndGrowth:
    def test_is_recent(self):
        assert da._is_recent(_days_ago(100), years=1)
        assert not da._is_recent(_days_ago(800), years=1)
        assert not da._is_recent(None) and not da._is_recent("")

    def test_yoy(self):
        assert da._yoy_growth_signal([]) == 0.0
        assert da._yoy_growth_signal([None] * 5) == 0.0
        assert da._yoy_growth_signal([_days_ago(10)] * 10) >= 0.5
        assert 0.0 <= da._yoy_growth_signal([_days_ago(i * 60) for i in range(20)]) <= 1.0


class TestAvailableSignals:
    def test_hiring_data_has_no_geo_or_sector(self):
        items = [{"skills": ["1"], "upload_date": _days_ago(5), "occupations": "dev"}]
        keys = da._available_signals(items)
        assert "geo_spread" not in keys and "cross_sector_adoption" not in keys
        assert {"posting_density", "recency_intensity", "yoy_growth_rate", "occupation_breadth"} <= set(keys)

    def test_tracker_data_has_all_six(self, sample_items):
        assert set(da._available_signals(sample_items)) == set(ALL_KEYS)


class TestComputeJobSignals:
    def test_returns_exactly_requested_keys(self, sample_items):
        keys = ["posting_density", "yoy_growth_rate"]
        assert set(da.compute_job_signals(SKILL_URIS[0], sample_items, 15, keys)) == set(keys)

    def test_no_match_all_zero(self):
        sig = da.compute_job_signals("missing", [{"skills": ["other"]}], 1, ALL_KEYS)
        assert sig == {k: 0.0 for k in ALL_KEYS}

    def test_empty_items(self):
        assert all(v == 0.0 for v in da.compute_job_signals("s1", [], 0, ALL_KEYS).values())

    def test_bounded(self):
        items = [{"skills": ["s1"], "upload_date": _days_ago(30 * i),
                  "country": ["DE", "FR", "IT", "ES", "PL"][i % 5],
                  "sectors": [["J", "K", "L", "M", "N"][i % 5]],
                  "occupations": [f"occ{i}"]} for i in range(20)]
        for k, v in da.compute_job_signals("s1", items, 20, ALL_KEYS).items():
            assert 0.0 <= v <= 1.0, k

    def test_diversity_raises_geo_and_sector(self):
        same = [{"skills": ["s1"], "upload_date": "2023-01-01", "country": "DE", "sectors": ["J"]}] * 5
        div = [{"skills": ["s1"], "upload_date": "2023-01-01", "country": c, "sectors": [s]}
               for c, s in zip(["DE", "FR", "IT", "ES", "PL"], ["J", "K", "L", "M", "N"])]
        a = da.compute_job_signals("s1", div, 5, ALL_KEYS)
        b = da.compute_job_signals("s1", same, 5, ALL_KEYS)
        assert a["geo_spread"] > b["geo_spread"]
        assert a["cross_sector_adoption"] > b["cross_sector_adoption"]

    def test_occupation_breadth_counts_list_occupations(self):
        items = [{"skills": ["s1"], "occupations": [f"u{i}"]} for i in range(5)]
        assert da.compute_job_signals("s1", items, 5, ["occupation_breadth"])["occupation_breadth"] == 0.5


# ══════════════════════════════════════════════════════════════════
#  4.  LLM HELPERS & RECOMMENDATION FALLBACKS
# ══════════════════════════════════════════════════════════════════

class TestLlmParsing:
    @pytest.mark.parametrize("raw,expected", [
        ('{"k":"v"}', {"k": "v"}),
        ('```json\n{"k":"v"}\n```', {"k": "v"}),
        ('{"k":"v"} trailing text', {"k": "v"}),
        ('Preamble {"k":"v"} postamble', {"k": "v"}),
        ('{"a":{"b":[1,2]}}', {"a": {"b": [1, 2]}}),
        ("{}", {}),
    ])
    def test_parses(self, raw, expected):
        assert da._parse_llm_json(raw) == expected

    @pytest.mark.parametrize("raw", ["totally not json!!!", ""])
    def test_raises(self, raw):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            da._parse_llm_json(raw)

    def test_strip_code_fences(self):
        assert da._strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'
        assert da._strip_code_fences('  {"a":1}  ') == '{"a":1}'


class TestRecommendationFallbacks:
    def test_short_term_static(self, no_llm):
        recs = da._short_term_recommendations(
            tier="high", entity_label="Python",
            metrics={"historical_cagr_pct": 12.0, "forecast_cagr_pct": 15.0, "demand_velocity_pct": 5.0,
                     "market_penetration_rate_pct": 8.0, "demand_volatility": 0.4,
                     "relative_growth_index": 1.8, "emergence_index": 0.7, "composite_potential_score": 0.8},
        )
        assert {"talent_acquisition", "training_and_development", "compensation_and_retention"} <= set(recs)

    @pytest.mark.parametrize("category", ["breakthrough", "emerging", "niche", "speculative"])
    def test_long_term_static_three_distinct(self, no_llm, category):
        tte = da.time_to_emergence(0.6, 0.8)
        recs = da._long_term_job_recommendations(category, "ML", 0.6, tte, {k: 0.3 for k in ALL_KEYS})
        assert set(recs) == {"strategic_workforce_planning", "partnerships_and_pipeline",
                             "regulatory_and_compliance"}
        assert len(set(recs.values())) == 3

    def test_long_term_prompt_only_lists_given_signals(self, monkeypatch):
        llm = MagicMock(return_value={"x": "y"})
        monkeypatch.setattr(da, "_chat_llm_json", llm)
        da._long_term_job_recommendations("niche", "ML", 0.4, da.time_to_emergence(0.4, 0.5),
                                          {"posting_density": 0.3}, sector="Research",
                                          data_scope="organization")
        prompt = llm.call_args[0][1]
        assert "Skill frequency across job postings" in prompt
        assert "Geographic spread" not in prompt
        assert "hiring-management" in prompt and "Research" in prompt


# ══════════════════════════════════════════════════════════════════
#  5.  PIPELINES
# ══════════════════════════════════════════════════════════════════

class TestRunShortTermAnalysis:
    @pytest.mark.parametrize("mode", ["skills", "occupations"])
    def test_structure(self, sample_items, label_dict, no_llm, mode):
        r = da.run_short_term_analysis(sample_items, mode, label_dict, top_n=5)
        assert {mode, "metadata", "sector_summary"} <= set(r)
        assert r["metadata"]["analysis_type"] == f"short_term_{mode}"
        assert r[mode], "expected non-empty results"

    def test_sorted_by_cps(self, sample_items, label_dict, no_llm):
        r = da.run_short_term_analysis(sample_items, "skills", label_dict, top_n=10)
        scores = [e["metrics"]["composite_potential_score"] for e in r["skills"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_and_counts(self, sample_items, label_dict, no_llm):
        r = da.run_short_term_analysis(sample_items, "skills", label_dict, top_n=2)
        ss = r["sector_summary"]
        assert len(r["skills"]) <= 2
        assert ss["high_potential_count"] + ss["medium_potential_count"] + ss["low_potential_count"] \
               == ss["total_entities_analyzed"]

    def test_labels_resolved(self, sample_items, label_dict, no_llm):
        r = da.run_short_term_analysis(sample_items, "occupations", label_dict, top_n=3)
        assert all(e["label"].startswith("Occupation") for e in r["occupations"])

    def test_empty_items(self, label_dict):
        assert da.run_short_term_analysis([], "skills", label_dict, top_n=5)["skills"] == []


class TestRunLongTermSkills:
    def test_structure_and_order(self, sample_items, label_dict, no_llm):
        r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=10)
        assert {"skills", "metadata", "sector_summary"} <= set(r)
        thetas = [s["theta"] for s in r["skills"]]
        assert thetas == sorted(thetas, reverse=True)
        assert all(0.0 <= t <= 1.0 for t in thetas)

    def test_required_fields(self, sample_items, label_dict, no_llm):
        r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=3)
        required = {"uri", "label", "emergence_quotient", "theta", "confidence", "time_to_emergence",
                    "dominant_category", "fuzzy_memberships", "irt_signals", "total_job_mentions",
                    "recommendations"}
        for s in r["skills"]:
            assert required <= set(s)

    def test_one_llm_call_per_skill(self, sample_items, label_dict, no_llm):
        r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=2)
        assert no_llm.call_count == len(r["skills"]) == 2

    def test_without_recommendations_no_llm(self, sample_items, label_dict, no_llm):
        r = da.run_long_term_skills_from_jobs(sample_items, label_dict, top_n=5, with_recommendations=False)
        no_llm.assert_not_called()
        assert all(s["recommendations"] is None for s in r["skills"])

    def test_signals_reported_in_metadata(self, label_dict, no_llm):
        hiring_like = [{"skills": ["1", "2"], "upload_date": _days_ago(i * 40), "occupations": "dev"}
                       for i in range(10)]
        r = da.run_long_term_skills_from_jobs(hiring_like, label_dict, top_n=2)
        used = r["metadata"]["irt_signals_used"]
        assert "geo_spread" not in used
        for s in r["skills"]:
            assert set(s["irt_signals"]) == set(used)

    def test_empty(self, label_dict):
        assert da.run_long_term_skills_from_jobs([], label_dict)["sector_summary"]["total_entities_analyzed"] == 0


class TestRunLongTermOccupations:
    def test_structure_and_order(self, sample_items, label_dict, no_llm):
        r = da.run_long_term_occupations_from_jobs(sample_items, label_dict, top_n=5, sector="J")
        assert {"occupations", "metadata", "sector_summary"} <= set(r)
        thetas = [o["theta"] for o in r["occupations"]]
        assert thetas == sorted(thetas, reverse=True)
        for o in r["occupations"]:
            assert {"uri", "label", "theta", "dominant_category", "recommendations"} <= set(o)

    def test_llm_only_for_returned_occupations(self, sample_items, label_dict, no_llm):
        """Regression: used to call the LLM for up to 500 skills plus every occupation."""
        r = da.run_long_term_occupations_from_jobs(sample_items, label_dict, top_n=2)
        assert len(r["occupations"]) == 2
        assert no_llm.call_count == 2

    def test_list_occupations_split(self, sample_items, label_dict, no_llm):
        r = da.run_long_term_occupations_from_jobs(sample_items, label_dict, top_n=10)
        assert {o["uri"] for o in r["occupations"]} == set(OCC_URIS)

    def test_named_occupations_from_hiring(self, label_dict, no_llm):
        items = [{"skills": [str(i % 4)], "upload_date": _days_ago(i * 30),
                  "occupations": ["dev", "analyst"][i % 2]} for i in range(10)]
        r = da.run_long_term_occupations_from_jobs(items, label_dict, top_n=5)
        assert {o["label"] for o in r["occupations"]} == {"dev", "analyst"}

    def test_empty(self, label_dict):
        assert da.run_long_term_occupations_from_jobs([], label_dict)["sector_summary"][
                   "total_occupations_analyzed"] == 0


# ══════════════════════════════════════════════════════════════════
#  6.  DATA SOURCES
# ══════════════════════════════════════════════════════════════════

class TestPortalGet:
    def test_sends_org_header(self, portal):
        da._portal_get(f"{da.HIRING_API}/api/v1/jobAds", "Acme")
        assert portal.calls[-1][1]["X-User-Organization"] == "Acme"

    def test_no_org_header_when_not_given(self, portal):
        da._portal_get(f"{da.EMPLOYEE_API}/organizations")
        assert "X-User-Organization" not in portal.calls[-1][1]

    def test_bearer_token_when_configured(self, portal, monkeypatch):
        monkeypatch.setattr(da, "PORTAL_API_TOKEN", "secret")
        da._portal_get(f"{da.EMPLOYEE_API}/organizations")
        assert portal.calls[-1][1]["Authorization"] == "Bearer secret"

    def test_retries_three_times_then_none(self, portal):
        portal.fail_hiring = True
        assert da._portal_get(f"{da.HIRING_API}/api/v1/jobAds", "Acme") is None
        assert len(portal.calls) == 3

    def test_required_raises(self, portal):
        portal.fail_hiring = True
        with pytest.raises(RuntimeError):
            da._portal_get(f"{da.HIRING_API}/api/v1/jobAds", "Acme", required=True)


class TestFetchOrgJobs:
    def test_reshapes_ads(self, portal):
        items, labels = da.fetch_org_jobs("Eclipse")
        assert len(items) == len(portal.ads)
        first = items[0]
        assert first["upload_date"] == portal.details[1]["publishDate"]
        assert first["occupations"] == portal.ads[0]["occupationName"]
        assert first["skills"] == sorted(str(s["id"]) for s in portal.skills[1])
        assert labels["1"] == "skill 1"

    def test_uses_ids_from_ads_for_skill_calls(self, portal):
        da.fetch_org_jobs("Eclipse")
        skill_urls = [u for u, _ in portal.calls if u.endswith("/interview-skills")]
        assert len(skill_urls) == len(portal.ads)
        assert f"{da.HIRING_API}/api/v1/jobAds/3/interview-skills" in skill_urls

    def test_missing_publish_date_kept_as_none(self, portal):
        portal.details[2] = {"id": 2}
        items, _ = da.fetch_org_jobs("Eclipse")
        assert items[1]["upload_date"] is None

    def test_occupation_falls_back_to_detail(self, portal):
        portal.ads[0]["occupationName"] = None
        portal.details[1]["occupation"] = {"title": "from detail"}
        items, _ = da.fetch_org_jobs("Eclipse")
        assert items[0]["occupations"] == "from detail"

    def test_ad_without_id_skipped(self, portal):
        portal.ads.append({"jobTitle": "no id"})
        items, _ = da.fetch_org_jobs("Eclipse")
        assert len(items) == len(portal.ads) - 1

    def test_list_failure_raises(self, portal):
        portal.fail_hiring = True
        with pytest.raises(RuntimeError):
            da.fetch_org_jobs("Eclipse")


class TestCleanSectors:
    @pytest.mark.parametrize("raw,expected", [
        (['"Computer programming activities"'], ["Computer programming activities"]),
        ([' "A" ', "'B'", "C"],                 ["A", "B", "C"]),
        (['"A"', "A", '""', "", None],          ["A"]),
        ('"Single"',                            ["Single"]),
        (None,                                  []),
    ])
    def test_cases(self, raw, expected):
        assert da._clean_sectors(raw) == expected


class TestFetchOrgProfile:
    def test_exact_match_preferred(self, portal):
        assert da.fetch_org_profile("eclipse")["id"] == 8

    def test_partial_match(self, portal):
        assert da.fetch_org_profile("foundation")["id"] == 7

    def test_case_and_whitespace_insensitive(self, portal):
        assert da.fetch_org_profile("  ECLIPSE FOUNDATION ")["id"] == 7

    def test_sectors_cleaned(self, portal):
        assert da.fetch_org_profile("Eclipse Foundation")["sectors"] == \
               ["Computer programming activities", "Research"]

    def test_does_not_mutate_source(self, portal):
        da.fetch_org_profile("Eclipse Foundation")
        assert portal.organizations[0]["sectors"][0] == '"Computer programming activities"'

    def test_not_found(self, portal):
        assert da.fetch_org_profile("nobody") is None

    def test_failure_raises(self, portal):
        portal.fail_employee = True
        with pytest.raises(RuntimeError):
            da.fetch_org_profile("eclipse")


class TestFetchSectorJobs:
    def test_body(self, monkeypatch):
        pag = MagicMock(return_value=[])
        monkeypatch.setattr(da, "paginate_all", pag)
        da.fetch_sector_jobs("Computer programming activities")
        body = pag.call_args.args[0]
        endpoint = pag.call_args.kwargs.get("endpoint", pag.call_args.args[1:2] and pag.call_args.args[1])
        start, end = da._default_dates(da.TRACKER_YEARS_BACK)
        assert endpoint == "jobs"
        assert body == {"sectors": "Computer programming activities",
                        "min_upload_date": start, "max_upload_date": end}


# ══════════════════════════════════════════════════════════════════
#  7.  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("path,key", ENDPOINTS)
class TestEveryEndpoint:
    def test_organization_required(self, path, key, services):
        assert client.get(path).status_code == 422

    def test_organization_query_param_not_accepted(self, path, key, services):
        assert client.get(f"{path}?organization=Eclipse&top_n=3").status_code == 422

    def test_both_analyses_returned(self, path, key, services):
        body = _analyse(path, "Eclipse Foundation", 3)
        assert body["organization_profile"]["id"] == 7
        org, sec = body["organization_analysis"], body["sector_analysis"]
        assert org["status"] == "ok" and org[key]
        assert "Hiring Management" in org["data_source"]
        assert sec["status"] == "ok"
        assert sec["sectors"] == ["Computer programming activities", "Research"]
        assert set(sec["results_by_sector"]) == set(sec["sectors"])
        for r in sec["results_by_sector"].values():
            assert r["status"] == "ok" and r[key] and r["jobs_retrieved"] == 12

    def test_tracker_queried_once_per_clean_sector(self, path, key, services):
        _, tracker, _ = services
        _analyse(path, "Eclipse Foundation", 3)
        assert sorted(c.args[0]["sectors"] for c in tracker.call_args_list) == \
               ["Computer programming activities", "Research"]

    def test_hiring_called_with_org_header(self, path, key, services):
        portal, _, _ = services
        _analyse(path, "Eclipse Foundation", 3)
        hiring = [h for u, h in portal.calls if u.startswith(TEST_HIRING_API)]
        assert hiring and all(h["X-User-Organization"] == "Eclipse Foundation" for h in hiring)

    def test_top_n_respected(self, path, key, services):
        body = _analyse(path, "Eclipse Foundation", 2)
        assert len(body["organization_analysis"][key]) <= 2
        for r in body["sector_analysis"]["results_by_sector"].values():
            assert len(r[key]) <= 2

    def test_second_call_served_from_cache(self, path, key, services):
        portal, tracker, _ = services
        first = _analyse(path, "Eclipse Foundation", 3)
        n_portal, n_tracker = len(portal.calls), tracker.call_count
        second = _get(path, "Eclipse Foundation").json()
        assert second == first
        assert len(portal.calls) == n_portal and tracker.call_count == n_tracker

    def test_hiring_failure_keeps_sector_and_skips_cache(self, path, key, services, tmp_cache):
        portal, _, _ = services
        portal.fail_hiring = True
        failed = _analyse(path, "Eclipse Foundation", 3)
        assert failed["status"] == "failed"
        body = failed["result"]
        assert body["organization_analysis"]["status"] == "error"
        assert body["sector_analysis"]["status"] == "ok"
        assert list(tmp_cache.iterdir()) == []
        assert _get(path, "Eclipse Foundation").json()["status"] == "started"   # retried

    def test_unknown_org(self, path, key, services):
        body = _analyse(path, "nobody", 3)
        assert body["organization_profile"] is None
        assert body["sector_analysis"]["status"] == "no_data"
        assert body["organization_analysis"]["status"] == "ok"   # hiring still answers for the name

    def test_first_call_starts_in_background(self, path, key, services, tmp_cache):
        r = _get(path, "Eclipse Foundation")
        assert r.status_code == 202
        assert r.json()["status"] == "started" and r.json()["result"] is None
        assert len(list(tmp_cache.iterdir())) == 1

    def test_in_progress_while_running(self, path, key, services):
        portal, tracker, _ = services
        da._RUNNING.add(da._cache_path(path, "Eclipse Foundation", 3))
        r = _get(path, "Eclipse Foundation")
        assert r.status_code == 202 and r.json()["status"] == "in_progress"
        assert portal.calls == [] and tracker.call_count == 0

    def test_finished_result_returned_with_200(self, path, key, services):
        _analyse(path, "Eclipse Foundation", 3)
        r = _get(path, "Eclipse Foundation")
        assert r.status_code == 200 and r.json()["organization_analysis"]["status"] == "ok"


class TestBackgroundJobs:
    def test_stale_in_progress_file_is_restarted(self, services):
        path = "/shorttermanalysis/skills"
        da._ensure_folder()
        da._save_cache(da._cache_path(path, "Eclipse", 3), da._in_progress_stub())
        r = _get(path, "Eclipse")
        assert r.status_code == 202 and r.json()["status"] == "started"
        assert _get(path, "Eclipse").json()["organization_analysis"]["status"] == "ok"

    def test_unexpected_exception_reported_once(self, services, tmp_cache, monkeypatch):
        path = "/longtermanalysis/skills"
        monkeypatch.setattr(da, "_compute_combined", MagicMock(side_effect=RuntimeError("boom")))
        failed = _analyse(path, "Eclipse", 3)
        assert failed["status"] == "failed" and "boom" in failed["message"] and failed["result"] is None
        assert list(tmp_cache.iterdir()) == []
        assert da._RUNNING == set()

    def test_job_removed_from_running_when_done(self, services):
        _analyse("/shorttermanalysis/skills", "Eclipse", 3)
        assert da._RUNNING == set()

    def test_different_organizations_run_separately(self, services):
        a = _analyse("/shorttermanalysis/skills", "Eclipse", 3)
        assert _get("/shorttermanalysis/skills", "Other Org").json()["status"] == "started"
        assert a["metadata"]["organization"] == "Eclipse"


class TestRerun:
    """?rerun=true recomputes an existing analysis (same organization + top_n)."""

    @pytest.mark.parametrize("path,key", ENDPOINTS)
    def test_rerun_recomputes_finished_analysis(self, path, key, services):
        portal, tracker, _ = services
        first = _analyse(path, "Eclipse Foundation", 3)
        n_portal, n_tracker = len(portal.calls), tracker.call_count
        r = client.get(f"{path}?top_n=3&rerun=true", headers={ORG_HEADER: "Eclipse Foundation"})
        assert r.status_code == 202 and r.json()["status"] == "started"
        assert len(portal.calls) > n_portal and tracker.call_count > n_tracker   # really ran again
        second = _get(path, "Eclipse Foundation").json()
        assert second["organization_analysis"]["status"] == "ok"
        assert second["metadata"]["analysis_date"] >= first["metadata"]["analysis_date"]

    def test_rerun_while_running_does_not_start_twice(self, services):
        path = "/shorttermanalysis/skills"
        portal, tracker, _ = services
        da._RUNNING.add(da._cache_path(path, "Eclipse", 3))
        r = client.get(f"{path}?top_n=3&rerun=true", headers={ORG_HEADER: "Eclipse"})
        assert r.status_code == 202 and r.json()["status"] == "in_progress"
        assert portal.calls == [] and tracker.call_count == 0

    def test_rerun_without_previous_result_just_starts(self, services):
        path = "/longtermanalysis/skills"
        r = client.get(f"{path}?top_n=3&rerun=true", headers={ORG_HEADER: "Eclipse"})
        assert r.status_code == 202 and r.json()["status"] == "started"
        assert _get(path, "Eclipse").json()["organization_analysis"]["status"] == "ok"

    def test_rerun_only_affects_its_own_top_n(self, services):
        path = "/shorttermanalysis/skills"
        _analyse(path, "Eclipse", 3)
        _analyse(path, "Eclipse", 2)
        client.get(f"{path}?top_n=3&rerun=true", headers={ORG_HEADER: "Eclipse"})
        assert _get(path, "Eclipse", 2).status_code == 200   # other top_n still cached

    def test_rerun_false_uses_cache(self, services):
        path = "/shorttermanalysis/skills"
        portal, _, _ = services
        first = _analyse(path, "Eclipse", 3)
        n = len(portal.calls)
        r = client.get(f"{path}?top_n=3&rerun=false", headers={ORG_HEADER: "Eclipse"})
        assert r.status_code == 200 and r.json() == first and len(portal.calls) == n


class TestEndpointEdgeCases:
    def test_org_without_sectors(self, services):
        _, tracker, _ = services
        body = _analyse("/shorttermanalysis/skills", "No Sector Org", 3)
        assert body["sector_analysis"]["status"] == "no_data"
        tracker.assert_not_called()

    def test_employee_failure_reported_and_not_cached(self, services, tmp_cache):
        portal, _, _ = services
        portal.fail_employee = True
        body = _analyse("/shorttermanalysis/skills", "Eclipse", 3)["result"]
        assert body["sector_analysis"]["status"] == "error"
        assert "Employee Management" in body["sector_analysis"]["message"]
        assert body["organization_analysis"]["status"] == "ok"
        assert list(tmp_cache.iterdir()) == []

    def test_one_sector_failing_does_not_hide_the_other(self, services, tmp_cache):
        _, tracker, _ = services

        def flaky(body, endpoint):
            if body["sectors"] == "Research":
                raise RuntimeError("tracker down")
            return _make_items(12)
        tracker.side_effect = flaky

        body = _analyse("/shorttermanalysis/skills", "Eclipse Foundation", 3)["result"]
        res = body["sector_analysis"]["results_by_sector"]
        assert res["Computer programming activities"]["status"] == "ok"
        assert res["Research"]["status"] == "error"
        assert list(tmp_cache.iterdir()) == []

    def test_sector_with_no_jobs(self, services):
        _, tracker, _ = services
        tracker.side_effect = lambda body, endpoint: []
        body = _analyse("/longtermanalysis/skills", "Eclipse", 3)
        assert body["sector_analysis"]["results_by_sector"]["Exact match sector"]["status"] == "no_data"
        assert body["sector_analysis"]["status"] == "no_data"

    def test_org_with_only_undated_ads(self, services):
        portal, _, _ = services
        portal.details = {i: {"id": i} for i in portal.details}
        body = _analyse("/shorttermanalysis/skills", "Eclipse", 3)
        org = body["organization_analysis"]
        assert org["status"] == "no_data"
        assert org["jobs_without_publish_date"] == len(portal.ads)

    def test_different_top_n_not_shared_in_cache(self, services):
        _, tracker, _ = services
        _analyse("/shorttermanalysis/skills", "Eclipse", 3)
        n = tracker.call_count
        _analyse("/shorttermanalysis/skills", "Eclipse", 4)
        assert tracker.call_count > n

    def test_org_sector_used_in_org_recommendations(self, services):
        _, _, llm = services
        _analyse("/longtermanalysis/skills", "Eclipse", 1)
        prompts = [c.args[1] for c in llm.call_args_list]
        assert any("hiring-management" in p and "Exact match sector" in p for p in prompts)