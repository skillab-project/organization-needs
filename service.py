"""
demand_analysis.py
==================
SKILLAB — Demand Analysis Service
Endpoints:
  GET /shorttermanalysis/skills       — US #23  Short-term skill demand (jobs only)
  GET /shorttermanalysis/occupations  — US #24  Short-term occupation demand (jobs only)
  GET /longtermanalysis/skills        — US #23  Long-term skill emergence (policies + projects + white papers)
  GET /longtermanalysis/occupations   — US #24  Long-term occupation emergence (aggregated from skills via ESCO)
"""

import time
import re
import math
import json
import logging
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, Query
import pandas as pd
import requests as req
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demand_analysis")

# ── Optional heavy dependencies ──────────────────────────────────
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    log.warning("statsmodels not found — linear trend fallback will be used for forecasting.")

try:
    from scipy.optimize import minimize_scalar
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    log.warning("scipy not found — grid-search theta estimation will be used.")

# ── App & environment ────────────────────────────────────────────
load_dotenv()
API      = os.getenv("TRACKER_API")
USERNAME = os.getenv("TRACKER_USERNAME")
PASSWORD = os.getenv("TRACKER_PASSWORD")

# ── LLM configuration ────────────────────────────────────────────
LLM_API_URL   = os.getenv("LLM_API_URL")
LLM_API_TOKEN = os.getenv("LLM_API_TOKEN")
LLM_MODEL     = os.getenv("LLM_MODEL", "mistral:latest")
LLM_TIMEOUT   = int(os.getenv("LLM_TIMEOUT", "120"))

LLM_HEADERS = {
    "Authorization": f"Bearer {LLM_API_TOKEN}",
    "Accept":        "application/json",
    "Content-Type":  "application/json",
}

def _strip_code_fences(s: str) -> str:
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s, count=1, flags=re.MULTILINE)
        s = re.sub(r"\n?```$",             "", s, count=1, flags=re.MULTILINE)
    return s.strip()

def _parse_llm_json(content: str) -> Dict:
    """Try progressively looser parsing strategies."""
    content = _strip_code_fences(content).strip()

    # 1. Direct parse (ideal case)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. raw_decode: parses the first valid JSON value and ignores anything after it
    try:
        obj, _ = json.JSONDecoder().raw_decode(content)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 3. Regex extraction: find the outermost {...} block
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON object found in LLM response: {content[:300]}")

def _chat_llm_json(system: str, user: str, schema: dict) -> Optional[Dict]:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.1,
        "seed":        42,
        "response_format": {"type": "json_object"},   # ← replaces "format"
    }
    url = f"{LLM_API_URL}/api/chat/completions"
    for attempt in range(3):
        try:
            resp = req.post(url, headers=LLM_HEADERS, json=payload, timeout=LLM_TIMEOUT)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            log.info(f"[LLM] raw content: {repr(content)}")
            return _parse_llm_json(content or "{}")
        except Exception as exc:
            log.warning(f"[LLM] attempt {attempt+1} failed: {exc}")
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    log.error("[LLM] all retries exhausted — falling back to static recommendations")
    return None
app = FastAPI(
    title="SKILLAB Demand Analysis API",
    description="Short-term and long-term skill/occupation demand analysis for US #23 and US #24.",
    version="1.0.0",
    # root_path="/demand_analysis",  # uncomment when running behind a proxy
)

FOLDER = Path("completed_anlyses5")


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — SHARED INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════

def get_token() -> str:
    res = req.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD})
    return res.text.replace('"', "")


def _parse_csv_str(v: Optional[str]) -> Optional[List[str]]:
    if not v:
        return None
    return [x.strip() for x in v.split(",") if x.strip()]


def _parse_csv_int(v: Optional[str]) -> Optional[List[int]]:
    if not v:
        return None
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def api_extract(request_body: dict, page: int, endpoint: str) -> dict:
    """Single-page retrieval."""
    page_size = 300
    params = {"page": page, "page_size": page_size}
    data = req.post(
        f"{API}/{endpoint}",
        headers={"Authorization": f"Bearer {get_token()}"},
        params=params,
        data=request_body,
    )
    return data.json()


def paginate_all(request_body: dict, endpoint: str) -> List[dict]:
    """
    Retrieve every page from an endpoint and return a flat list of items.
    """
    log.info(f"[{endpoint}] fetching page 1...")
    first = api_extract(request_body, page=1, endpoint=endpoint)
    count = first.get("count", 0)
    n_pages = max(1, math.ceil(count / 300))
    log.info(f"[{endpoint}] {count} records found — {n_pages} page(s) total")

    items: List[dict] = list(first.get("items", []))
    for page in range(2, n_pages + 1):
        log.info(f"[{endpoint}] fetching page {page}/{n_pages}...")
        chunk = api_extract(request_body, page=page, endpoint=endpoint)
        items.extend(chunk.get("items", []))
        time.sleep(0.3)  # polite rate-limiting

    log.info(f"[{endpoint}] DONE — {len(items)} items retrieved")
    return items


def load_esco_mapping() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load new_ESCO_mapping.xlsx and return (DataFrame, uri→label dict)."""
    log.info("Loading ESCO mapping file...")
    skills_df = pd.read_excel("mapping_of_ESCO_skills.xlsx")
    list_cols = [
        "skills_levels", "knowledge_levels", "traversal_levels",
        "skills_ancestors", "knowledge_ancestors", "traversal_ancestors", "children",
    ]
    for col in list_cols:
        if col in skills_df.columns:
            skills_df[col] = skills_df[col].apply(eval)
    label_dict: Dict[str, str] = {
        row["conceptUri"]: row["preferredLabel"]
        for _, row in skills_df.iterrows()
    }
    log.info(f"ESCO mapping loaded: {len(label_dict)} URI→label entries")
    return skills_df, label_dict


def _ensure_folder() -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)


def _load_or_init_cache(file_path: str) -> Tuple[Optional[Any], bool]:
    """
    Returns (cached_data, already_exists).
    If file does not exist, writes an in-progress stub and returns (None, False).
    """
    if os.path.exists(file_path):
        log.info(f"Cache hit: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f), True
    log.info(f"Cache miss: {file_path} — initializing stub")
    stub = {"status": "in_progress", "message": "Analysis is being computed", "result": None}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(stub, f, indent=4, ensure_ascii=False)
    return None, False


def _save_cache(file_path: str, data: Any) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)
    log.info(f"Saved cache: {file_path}")


def _error_cache(file_path: str, exc: Exception) -> None:
    log.error(f"Error in pipeline: {exc}")
    _save_cache(file_path, {"status": "error", "message": str(exc), "result": None})


# ── Sector extraction ────────────────────────────────────────────

_SECTOR_FIELDS = ("sectors")

def _extract_sectors(item: dict) -> List[str]:
    """Return the list of sector codes attached to a job item."""
    for field in _SECTOR_FIELDS:
        val = item.get(field)
        if val is None:
            continue
        if isinstance(val, list):
            cleaned = [str(s).strip() for s in val if s]
            return cleaned if cleaned else ["unknown"]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
    return ["unknown"]


def _group_items_by_sector(items: List[dict]) -> Dict[str, List[dict]]:
    """Partition job items by sector."""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        for sector in _extract_sectors(item):
            groups[sector].append(item)
    return dict(groups)


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — SHORT-TERM ANALYSIS
# ══════════════════════════════════════════════════════════════════

# ── 2.1  Date / quarter helpers ──────────────────────────────────

def _to_quarter_str(date_val: Optional[str]) -> Optional[str]:
    if not date_val:
        return None
    try:
        d = datetime.fromisoformat(str(date_val)[:10])
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}"
    except Exception:
        return None


def _quarters_back(n: int) -> List[str]:
    today = datetime.now()
    cq = (today.month - 1) // 3
    cy = today.year
    labels = []
    for i in range(n - 1, -1, -1):
        tq = cq - i
        labels.append(f"{cy + tq // 4}-Q{tq % 4 + 1}")
    return labels


def _quarters_forward(n: int) -> List[str]:
    today = datetime.now()
    cq = (today.month - 1) // 3
    cy = today.year
    labels = []
    for i in range(1, n + 1):
        tq = cq + i
        labels.append(f"{cy + tq // 4}-Q{tq % 4 + 1}")
    return labels


# ── 2.2  Time-series builders ────────────────────────────────────

def _build_skill_series(items: List[dict], date_field: str = "upload_date") -> Dict[str, Dict[str, int]]:
    acc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        q = _to_quarter_str(item.get(date_field))
        if not q:
            continue
        for skill in item.get("skills", []):
            acc[skill][q] += 1
    return {k: dict(v) for k, v in acc.items()}


def _build_occupation_series(items: List[dict], date_field: str = "upload_date") -> Dict[str, Dict[str, int]]:
    acc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        q = _to_quarter_str(item.get(date_field))
        if not q:
            continue
        occ = item.get("occupation_id") or item.get("occupations")
        if occ:
            acc[str(occ)][q] += 1
    return {k: dict(v) for k, v in acc.items()}


def _fill_series(quarter_dict: Dict[str, int], labels: List[str]) -> List[float]:
    return [float(quarter_dict.get(q, 0)) for q in labels]


# ── 2.3  Forecasting ─────────────────────────────────────────────

def _linear_forecast(series: List[float], n: int) -> Dict:
    arr = np.array(series, dtype=float)
    x = np.arange(len(arr))
    if len(arr) >= 2 and arr.std() > 0:
        slope, intercept = np.polyfit(x, arr, 1)
        residual_std = float(np.std(arr - (intercept + slope * x)))
    else:
        slope, intercept = 0.0, float(arr.mean()) if len(arr) else 0.0
        residual_std = float(arr.mean() * 0.15) if len(arr) else 0.0

    forecast = []
    last_x = len(arr) - 1
    for i in range(1, n + 1):
        fv = max(0.0, intercept + slope * (last_x + i))
        m95 = 1.96 * residual_std * math.sqrt(i)
        m80 = 1.28 * residual_std * math.sqrt(i)
        forecast.append({
            "value":       round(fv, 2),
            "ci_lower_95": round(max(0.0, fv - m95), 2),
            "ci_upper_95": round(fv + m95, 2),
            "ci_lower_80": round(max(0.0, fv - m80), 2),
            "ci_upper_80": round(fv + m80, 2),
        })
    return {"method": "linear_trend", "forecast": forecast}


def forecast_series(series: List[float], n_forecast: int = 12) -> Dict:
    arr = np.array(series, dtype=float)
    n = len(arr)

    if n < 4 or arr.sum() == 0:
        return _linear_forecast(list(arr), n_forecast)

    if HAS_STATSMODELS:
        try:
            use_seasonal = n >= 12
            model = ExponentialSmoothing(
                arr,
                trend="add",
                seasonal="add" if use_seasonal else None,
                seasonal_periods=4 if use_seasonal else None,
                damped_trend=True,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            raw_fc = fit.forecast(n_forecast)
            sigma = max(float(np.std(fit.resid)), float(arr.mean()) * 0.05)

            forecast = []
            for i, fv in enumerate(raw_fc):
                fv = max(0.0, float(fv))
                m95 = 1.96 * sigma * math.sqrt(i + 1)
                m80 = 1.28 * sigma * math.sqrt(i + 1)
                forecast.append({
                    "value":       round(fv, 2),
                    "ci_lower_95": round(max(0.0, fv - m95), 2),
                    "ci_upper_95": round(fv + m95, 2),
                    "ci_lower_80": round(max(0.0, fv - m80), 2),
                    "ci_upper_80": round(fv + m80, 2),
                })
            return {"method": "holt_winters", "forecast": forecast}
        except Exception as hw_err:
            log.debug(f"[HW] Error ({hw_err}) — falling back to linear trend")

    return _linear_forecast(list(arr), n_forecast)


# ── 2.4  Econometric metrics ─────────────────────────────────────

def _cagr(series: List[float], n_years: float) -> Optional[float]:
    if len(series) < 2 or series[0] <= 0 or series[-1] < 0:
        return None
    try:
        return round(((series[-1] / series[0]) ** (1.0 / n_years) - 1.0) * 100, 3)
    except Exception:
        return None


def _demand_velocity(series: List[float]) -> Optional[float]:
    if len(series) < 3 or series[-3] == 0:
        return None
    return round((series[-1] - series[-3]) / series[-3] * 100, 3)


def _mpr(entity_series: List[float], total_series: List[float]) -> Optional[float]:
    avg_e = float(np.mean(entity_series)) if entity_series else 0.0
    avg_t = float(np.mean(total_series)) if total_series else 0.0
    if avg_t == 0:
        return None
    return round(avg_e / avg_t * 100, 3)


def _volatility(series: List[float]) -> Optional[float]:
    if len(series) < 3:
        return None
    rates = [
        (series[i] - series[i - 1]) / series[i - 1]
        for i in range(1, len(series))
        if series[i - 1] > 0
    ]
    if len(rates) < 2:
        return None
    mu = float(np.mean(rates))
    if mu == 0:
        return None
    return round(abs(float(np.std(rates)) / mu), 4)


def _emergence_index(series: List[float]) -> Optional[float]:
    if len(series) < 5:
        return None
    recent  = float(np.mean(series[-4:]))
    overall = float(np.mean(series))
    if overall == 0:
        return None
    ei_raw = recent / overall - 1.0
    return round(max(0.0, min(1.0, (ei_raw + 1.0) / 2.0)), 4)


def _rgi(entity_cagr: Optional[float], sector_mean_cagr: float) -> Optional[float]:
    if entity_cagr is None or sector_mean_cagr == 0:
        return None
    return round(entity_cagr / sector_mean_cagr, 4)


def _minmax(val: Optional[float], vmin: float, vmax: float) -> float:
    if val is None:
        return 0.0
    if vmax == vmin:
        return 0.5
    return float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))


# ── 2.5  Composite Potential Score & classification ───────────────

DEFAULT_CPS_WEIGHTS = {
    "forecast_cagr": 0.30,
    "hist_cagr":     0.20,
    "rgi":           0.20,
    "ei":            0.18,
    "stability":     0.12,
}


def compute_cps(
    hist_cagr: Optional[float],
    fore_cagr: Optional[float],
    rgi:       Optional[float],
    ei:        Optional[float],
    volatility: Optional[float],
    all_hist_cagrs:  List[float],
    all_fore_cagrs:  List[float],
    all_rgis:        List[float],
    all_vols:        List[float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    w = weights or DEFAULT_CPS_WEIGHTS
    s = lambda lst: (min((x for x in lst if x is not None), default=0.0),
                     max((x for x in lst if x is not None), default=1.0))

    mn_hc, mx_hc = s(all_hist_cagrs)
    mn_fc, mx_fc = s(all_fore_cagrs)
    mn_rg, mx_rg = s(all_rgis)
    mn_vl, mx_vl = s(all_vols)

    f_hc   = _minmax(hist_cagr,  mn_hc, mx_hc)
    f_fc   = _minmax(fore_cagr,  mn_fc, mx_fc)
    f_rgi  = _minmax(rgi,        mn_rg, mx_rg)
    f_ei   = ei if ei is not None else 0.0
    f_stab = 1.0 - _minmax(volatility, mn_vl, mx_vl) if volatility is not None else 0.5

    cps = (
        w["forecast_cagr"] * f_fc
        + w["hist_cagr"]   * f_hc
        + w["rgi"]         * f_rgi
        + w["ei"]          * f_ei
        + w["stability"]   * f_stab
    )
    return round(float(np.clip(cps, 0.0, 1.0)), 4)


def classify_potential(cps: float) -> str:
    if cps >= 0.65:
        return "high"
    elif cps >= 0.35:
        return "medium"
    return "low"


# ── 2.6  Short-term recommendations ──────────────────────────────

_ST_RECS: Dict[str, List[Dict]] = {
    "high": [
        {
            "action":  "Prioritize talent acquisition",
            "detail":  "Demand is accelerating. Launch targeted recruitment and fast-track internal upskilling pipelines immediately. Define dedicated career paths to retain scarce talent ahead of market competition.",
            "owner":   "HR Manager / Recruiter",
            "urgency": "Immediate",
        },
        {
            "action":  "Build a structured training pipeline",
            "detail":  "Establish certifications and structured learning pathways. Partnering with external training providers now provides lead time before trainer supply tightens.",
            "owner":   "L&D Lead",
            "urgency": "Within 3 months",
        },
        {
            "action":  "Benchmark compensation against market",
            "detail":  "Rising demand signals supply scarcity. Conduct a salary benchmarking exercise to stay competitive and prevent attrition to higher-paying competitors.",
            "owner":   "HR Manager",
            "urgency": "Within 6 months",
        },
    ],
    "medium": [
        {
            "action":  "Build a proactive candidate pipeline",
            "detail":  "Steady growth warrants preparatory action. Map internal talent and develop a warm pipeline for future gaps before demand accelerates.",
            "owner":   "Recruiter",
            "urgency": "Within 6 months",
        },
        {
            "action":  "Cross-skill bundling strategy",
            "detail":  "Bundle this skill or role with high-potential areas to amplify strategic value. Identify adjacent roles where skill combinations create multiplied organizational impact.",
            "owner":   "HR Strategy",
            "urgency": "Within 12 months",
        },
    ],
    "low": [
        {
            "action":  "Redirect training investment",
            "detail":  "Low or declining demand signals reduced strategic value. Redirect L&D budgets toward high-potential skill areas with stronger forecasted return on investment.",
            "owner":   "HR Manager / L&D Lead",
            "urgency": "Next planning cycle",
        },
        {
            "action":  "Reassess standalone role definitions",
            "detail":  "Consider integrating this skill within broader, in-demand role profiles rather than maintaining standalone job descriptions centred on it.",
            "owner":   "HR Strategy",
            "urgency": "Next planning cycle",
        },
    ],
}


def _short_term_recommendations(
    tier:         str,
    entity_label: str,
    metrics:      Dict,
    sector:       str = "Unknown",
    organization: str = "Unknown",
    entity_type:  str = "skill",
) -> Dict:

    # ── Derive interpretive context from raw metrics ──────────────
    hist_cagr   = metrics.get("historical_cagr_pct")
    fore_cagr   = metrics.get("forecast_cagr_pct")
    dv          = metrics.get("demand_velocity_pct")
    mpr         = metrics.get("market_penetration_rate_pct")
    rgi         = metrics.get("relative_growth_index")
    ei          = metrics.get("emergence_index")
    vol         = metrics.get("demand_volatility")
    cps         = metrics.get("composite_potential_score")

    # Momentum quadrant (from deliverable Section 3.1.3.2)
    if hist_cagr is not None and dv is not None:
        if hist_cagr > 0 and dv > 0:
            momentum = "sustained accelerating growth — the strongest signal of strategic priority"
        elif hist_cagr > 0 and dv <= 0:
            momentum = "historical growth that is currently losing momentum — warrants a cautious stance"
        elif hist_cagr <= 0 and dv > 0:
            momentum = "potential recovery from prior contraction — a possible early-entry opportunity"
        else:
            momentum = "structural decline regardless of any historical periods of apparent recovery"
    else:
        momentum = "insufficient data to determine momentum direction"

    # RGI interpretation (thresholds from deliverable)
    if rgi is not None:
        if rgi > 1.2:
            rgi_interp = f"strong outperformance — growing at {rgi:.2f}x the sector average"
        elif rgi >= 0.8:
            rgi_interp = f"broadly in line with the sector average ({rgi:.2f}x)"
        else:
            rgi_interp = f"meaningful underperformance relative to the sector ({rgi:.2f}x)"
    else:
        rgi_interp = "not computable (flat or zero sector baseline)"

    # EI interpretation
    if ei is not None:
        if ei > 0.65:
            ei_interp = "strong recent acceleration above historical baseline — canonical emerging signal"
        elif ei > 0.5:
            ei_interp = "mild recent acceleration above historical mean"
        else:
            ei_interp = "recent demand below or at historical mean — no acceleration signal"
    else:
        ei_interp = "not available"

    # Maturity signal from MPR + EI combination (from deliverable)
    if mpr is not None and ei is not None:
        if mpr < 10 and ei > 0.6:
            maturity_signal = "low market penetration combined with strong recent acceleration — canonical emerging skill signature; early investment creates genuine competitive advantage"
        elif mpr >= 20 and (fore_cagr or 0) > 5:
            maturity_signal = "broadly embedded in sector hiring vocabulary and still growing — sustained broad-based demand rather than niche emergence"
        elif mpr >= 20 and (fore_cagr or 0) <= 0:
            maturity_signal = "widely listed in job descriptions but demand is declining — possible structural obsolescence driven by template inertia"
        else:
            maturity_signal = "moderate market penetration with mixed growth signals"
    else:
        maturity_signal = "market positioning signals not fully available"

    system = (
        "You are a workforce strategy advisor generating evidence-grounded recommendations "
        "for an HR manager using the SKILLAB platform. "
        "You MUST respond with valid JSON only — no prose, no numbered lists, no markdown. "
        "Your entire response must be a single JSON object matching the required structure."
    )

    user = f"""You are preparing a demand briefing for the HR manager at {organization}, 
operating in the {sector} sector.

The analysis concerns the {entity_type} "{entity_label}" within the {sector} sector, 
derived from job posting data over the past 3 years with a 3-year forward forecast.

== METRIC PROFILE ==

Growth:
  Historical CAGR (past 3 years):    {hist_cagr}%
  Forecast CAGR (next 3 years):      {fore_cagr}%
  Demand Velocity (6-month momentum):{dv}%
  Momentum interpretation:           {momentum}

Market Positioning:
  Market Penetration Rate:           {mpr}% of sector job postings
  Relative Growth Index:             {rgi_interp}
  Emergence Index (0–1):             {ei} — {ei_interp}
  Maturity signal:                   {maturity_signal}

Risk:
  Demand Volatility (coeff. of var.):{vol} — {"high signal uncertainty; staged investment advised" if (vol or 0) > 1.0 else "acceptable signal stability"}

Composite Potential Score:           {cps} / 1.00
Potential Tier:                      {tier.upper()}

== ANALYTICAL INTERPRETATION GUIDE ==

Use the following rules when reasoning about what to recommend:

- High CAGR + positive Demand Velocity → sustained accelerating growth; immediate action required
- High CAGR + negative Demand Velocity → past growth losing momentum; cautious preparedness stance
- Low/negative CAGR + positive Demand Velocity → potential recovery; early-entry opportunity
- Low CAGR + negative Demand Velocity → structural decline; redirect investment
- RGI > 1.2 → skill is a genuine sector outperformer, not just riding macro growth
- RGI < 0.8 → growth driven by sector-wide tide, not skill-specific demand
- MPR < 10% + high EI → canonical emerging skill; early investment advantage still available
- MPR > 20% + negative CAGR → commoditized or declining despite widespread listing
- High Volatility (> 1.0) → forecast uncertainty is elevated; recommend staged rather than immediate large commitment
- Tier HIGH → convey urgency; specific timelines required (e.g. "within the next two quarters")
- Tier MEDIUM → convey preparedness; build pipeline ahead of expected acceleration
- Tier LOW → convey strategic redirection; avoid new investment; consider redeployment

== YOUR TASK ==

Generate three recommendations, one per dimension:
1. talent_acquisition — hiring strategy and external pipeline
2. training_and_development — internal upskilling and L&D investment
3. compensation_and_retention — pay benchmarking and retention tactics

Each recommendation MUST:
- Be 3–4 sentences long
- Address the HR manager in second person ("you should...")
- Cite at least two specific metric values by number (e.g. "a forecast CAGR of X%" or "an RGI of Y")
- Reference "{entity_label}" by name at least once
- Reference the "{sector}" sector context explicitly
- Apply the interpretation guide above — do not give generic advice
- Match the urgency of the {tier.upper()} tier
- If Demand Volatility > 1.0, acknowledge forecast uncertainty and recommend staged investment
"""

    schema = {
        "type": "object",
        "properties": {
            "talent_acquisition":         {"type": "string"},
            "training_and_development":   {"type": "string"},
            "compensation_and_retention": {"type": "string"},
        },
        "required": ["talent_acquisition", "training_and_development", "compensation_and_retention"],
        "additionalProperties": False,
    }

    result = _chat_llm_json(system, user, schema)
    if result:
        return result

    log.warning(f"[Recs] LLM failed for '{entity_label}' — using static fallback")
    static = _ST_RECS.get(tier, _ST_RECS["medium"])
    return {
        "talent_acquisition":         static[0]["detail"] if len(static) > 0 else "",
        "training_and_development":   static[1]["detail"] if len(static) > 1 else "",
        "compensation_and_retention": static[0]["detail"] if len(static) > 0 else "",
    }


# ── 2.7  Core short-term pipeline ────────────────────────────────

def run_short_term_analysis(
    items:        List[dict],
    mode:         str,
    label_dict:   Dict[str, str],
    date_field:   str = "upload_date",
    top_n:        int = 50,
    sector:       str = "Unknown",        # ← add
    organization: str = "Unknown",        # ← add
) -> Dict:
    log.info(f"  [Analysis/{mode}] starting on {len(items)} items, top_n={top_n}")
    N_HIST = 12
    N_FORE = 12
    N_HIST_YRS = 3.0

    hist_labels = _quarters_back(N_HIST)
    fore_labels = _quarters_forward(N_FORE)

    log.info(f"  [Analysis/{mode}] building total time series...")
    total_q: Dict[str, int] = defaultdict(int)
    for item in items:
        q = _to_quarter_str(item.get(date_field))
        if q:
            total_q[q] += 1
    total_series = _fill_series(dict(total_q), hist_labels)

    log.info(f"  [Analysis/{mode}] building per-entity time series...")
    if mode == "skills":
        raw = _build_skill_series(items, date_field)
    else:
        raw = _build_occupation_series(items, date_field)

    totals = {uri: sum(v.values()) for uri, v in raw.items()}
    top_entities = sorted(totals, key=totals.get, reverse=True)[:top_n]
    log.info(f"  [Analysis/{mode}] {len(raw)} unique entities — analyzing top {len(top_entities)}")

    log.info(f"  [Analysis/{mode}] Pass 1: forecasting + raw metrics...")
    pass1: Dict[str, Dict] = {}
    for idx, uri in enumerate(top_entities, 1):
        if idx % 10 == 0 or idx == len(top_entities):
            log.info(f"    ...Pass 1 progress: {idx}/{len(top_entities)}")
        hist = _fill_series(raw[uri], hist_labels)
        fc_data = forecast_series(hist, n_forecast=N_FORE)
        fore_vals = [p["value"] for p in fc_data["forecast"]]
        pass1[uri] = {
            "hist":        hist,
            "fore_data":   fc_data,
            "fore_vals":   fore_vals,
            "hist_cagr":   _cagr(hist, N_HIST_YRS),
            "fore_cagr":   _cagr(fore_vals, N_HIST_YRS),
            "dv":          _demand_velocity(hist),
            "mpr":         _mpr(hist, total_series),
            "vol":         _volatility(hist),
            "ei":          _emergence_index(hist),
        }

    log.info(f"  [Analysis/{mode}] Pass 1 done. Computing normalization vectors...")
    all_hc  = [p["hist_cagr"] for p in pass1.values() if p["hist_cagr"] is not None]
    all_fc  = [p["fore_cagr"] for p in pass1.values() if p["fore_cagr"] is not None]
    all_vol = [p["vol"]       for p in pass1.values() if p["vol"] is not None]
    all_ei  = [p["ei"]        for p in pass1.values() if p["ei"] is not None]
    sector_mean_hcagr = float(np.mean(all_hc)) if all_hc else 0.0

    log.info(f"  [Analysis/{mode}] Pass 2: RGI, CPS, tier, recommendations...")
    results = []
    for idx, uri in enumerate(top_entities, 1):
        if idx % 10 == 0 or idx == len(top_entities):
            log.info(f"    ...Pass 2 progress: {idx}/{len(top_entities)}")
        p = pass1[uri]
        rgi = _rgi(p["hist_cagr"], sector_mean_hcagr)
        all_rgi = [
            _rgi(p2["hist_cagr"], sector_mean_hcagr)
            for p2 in pass1.values()
            if p2["hist_cagr"] is not None
        ]
        all_rgi = [r for r in all_rgi if r is not None]

        cps  = compute_cps(p["hist_cagr"], p["fore_cagr"], rgi, p["ei"], p["vol"],
                           all_hc, all_fc, all_rgi, all_vol)
        tier = classify_potential(cps)
        label = label_dict.get(uri, uri)

        metrics_out = {
            "historical_cagr_pct":       p["hist_cagr"],
            "forecast_cagr_pct":         p["fore_cagr"],
            "demand_velocity_pct":        p["dv"],
            "market_penetration_rate_pct": p["mpr"],
            "demand_volatility":          p["vol"],
            "relative_growth_index":      round(rgi, 4) if rgi is not None else None,
            "emergence_index":            p["ei"],
            "composite_potential_score":  cps,
        }

        results.append({
            "uri":   uri,
            "label": label,
            "time_series": {
                "historical": [
                    {"quarter": q, "count": int(c)}
                    for q, c in zip(hist_labels, p["hist"])
                ],
                "forecast": [
                    {"quarter": q, **fp}
                    for q, fp in zip(fore_labels, p["fore_data"]["forecast"])
                ],
                "forecast_method": p["fore_data"]["method"],
            },
            "metrics":        metrics_out,
            "potential_tier": tier,
            "recommendations": _short_term_recommendations(
    tier, label, metrics_out,
    sector=sector,
    organization=organization,
    entity_type=mode[:-1]   # "skills"→"skill", "occupations"→"occupation"
),
    })

    results.sort(key=lambda x: x["metrics"]["composite_potential_score"], reverse=True)

    tiers = [r["potential_tier"] for r in results]
    sector_summary = {
        "total_entities_analyzed":    len(results),
        "high_potential_count":       tiers.count("high"),
        "medium_potential_count":     tiers.count("medium"),
        "low_potential_count":        tiers.count("low"),
        "sector_mean_historical_cagr_pct":  round(sector_mean_hcagr, 3),
        "sector_mean_forecast_cagr_pct":    round(float(np.mean(all_fc)), 3) if all_fc else None,
        "top_high_potential": [r["label"] for r in results if r["potential_tier"] == "high"][:5],
        "top_low_potential":  [r["label"] for r in results if r["potential_tier"] == "low"][:5],
    }

    log.info(f"  [Analysis/{mode}] COMPLETE: {len(results)} entities "
             f"(high={tiers.count('high')}, med={tiers.count('medium')}, low={tiers.count('low')})")

    return {
        "metadata": {
            "analysis_type":            f"short_term_{mode}",
            "forecast_horizon_years":   3,
            "historical_window_years":  3,
            "forecast_model":           "holt_winters_with_linear_fallback",
            "total_records_retrieved":  len(items),
            "analysis_date":            datetime.now().isoformat(),
        },
        mode:             results,
        "sector_summary": sector_summary,
    }


# ── 2.8  Per-sector dispatcher ────────────────────────────────────

def run_short_term_analysis_by_sector(
    items:        List[dict],
    mode:         str,
    label_dict:   Dict[str, str],
    top_n:        int = 50,
    organization: str = "Unknown",
    user_sector:  str = "Unknown",        # ← add
) -> Dict[str, Dict]:
    sector_groups = _group_items_by_sector(items)
    log.info(f"[BySector] {len(sector_groups)} sector(s) found: {sorted(sector_groups)}")

    results: Dict[str, Dict] = {}
    n_sectors = len(sector_groups)
    for i, (sector_code, sector_items) in enumerate(sorted(sector_groups.items()), 1):
        log.info(f"[BySector {i}/{n_sectors}] '{sector_code}' — {len(sector_items)} items — starting {mode} analysis")
        analysis = run_short_term_analysis(
            sector_items, mode=mode, label_dict=label_dict, top_n=top_n,
            sector=user_sector,           # ← use user's value instead of sector_code
            organization=organization,
        )
        analysis["metadata"]["sector"] = user_sector    # ← same here
        analysis["metadata"]["sector_item_count"] = len(sector_items)
        results[sector_code] = analysis
        log.info(f"[BySector {i}/{n_sectors}] '{sector_code}' DONE")

    log.info(f"[BySector] All {n_sectors} sectors processed.")
    return results


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — LONG-TERM ANALYSIS  (EMERGE framework)
# ══════════════════════════════════════════════════════════════════

IRT_PARAMS: Dict[str, Dict] = {
    "whitepaper_density":     {"a": 0.8, "b": 0.20, "desc": "White paper publication density"},
    "project_density":        {"a": 1.4, "b": 0.35, "desc": "R&D project funding signal"},
    "policy_density":         {"a": 1.6, "b": 0.45, "desc": "Policy document mention density"},
    "policy_intensity":       {"a": 2.0, "b": 0.52, "desc": "Policy language intensity (recent vs. all)"},
    "geo_spread":             {"a": 1.3, "b": 0.40, "desc": "Geographic spread across EU"},
    "cross_sector_adoption":  {"a": 1.5, "b": 0.55, "desc": "Cross-sector adoption signals (NACE diversity)"},
    "yoy_growth_rate":        {"a": 1.8, "b": 0.38, "desc": "Year-over-year mention growth rate"},
}

JOB_IRT_PARAMS: Dict[str, Dict] = {
    "posting_density":       {"a": 1.2, "b": 0.25, "desc": "Skill frequency across job postings"},
    "recency_intensity":     {"a": 2.0, "b": 0.50, "desc": "Share of postings in last 12 months vs all"},
    "geo_spread":            {"a": 1.3, "b": 0.40, "desc": "Geographic spread of postings"},
    "cross_sector_adoption": {"a": 1.5, "b": 0.55, "desc": "Sector diversity of postings"},
    "yoy_growth_rate":       {"a": 1.8, "b": 0.38, "desc": "Year-over-year posting growth rate"},
    "occupation_breadth":    {"a": 1.1, "b": 0.30, "desc": "Distinct occupations requiring this skill"},
}
def _irt_p(theta: float, a: float, b: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-a * (theta - b)))
    except OverflowError:
        return 0.0 if theta < b else 1.0


def _log_likelihood(theta: float, signal_vec: Dict[str, float]) -> float:
    ll = 0.0
    for key, val in signal_vec.items():
        if key not in IRT_PARAMS:
            continue
        p_params = IRT_PARAMS[key]
        prob = float(np.clip(_irt_p(theta, p_params["a"], p_params["b"]), 1e-9, 1 - 1e-9))
        ll += val * math.log(prob) + (1 - val) * math.log(1 - prob)
    return ll


def estimate_theta(signal_vec: Dict[str, float]) -> Tuple[float, float]:
    if not signal_vec or all(v == 0.0 for v in signal_vec.values()):
        return 0.10, 0.15

    if HAS_SCIPY:
        try:
            result = minimize_scalar(
                lambda th: -_log_likelihood(th, signal_vec),
                bounds=(0.01, 0.99),
                method="bounded",
            )
            theta_hat = float(np.clip(result.x, 0.05, 0.95))
        except Exception:
            theta_hat = _grid_theta(signal_vec)
    else:
        theta_hat = _grid_theta(signal_vec)

    h = 1e-3
    ll_c = _log_likelihood(theta_hat, signal_vec)
    ll_p = _log_likelihood(min(0.99, theta_hat + h), signal_vec)
    ll_m = _log_likelihood(max(0.01, theta_hat - h), signal_vec)
    fisher = max(1e-9, -(ll_p - 2 * ll_c + ll_m) / h ** 2)
    se = 1.0 / math.sqrt(fisher)
    confidence = float(np.clip(1.0 - se, 0.10, 0.98))

    return round(theta_hat, 4), round(confidence, 4)


def _grid_theta(signal_vec: Dict[str, float]) -> float:
    grid = [i / 100.0 for i in range(1, 100)]
    return max(grid, key=lambda th: _log_likelihood(th, signal_vec))


FUZZY_SETS: Dict[str, Dict[str, float]] = {
    "speculative":  {"a": -0.10, "b": -0.10, "c": 0.22, "d": 0.45},
    "niche":        {"a":  0.30, "b":  0.42, "c": 0.54, "d": 0.66},
    "emerging":     {"a":  0.54, "b":  0.64, "c": 0.76, "d": 0.87},
    "breakthrough": {"a":  0.74, "b":  0.86, "c": 1.10, "d": 1.10},
}


def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    return (x - a) / (b - a) if x < b else (d - x) / (d - c)


def fuzzy_memberships(theta: float) -> Dict[str, float]:
    return {
        label: round(_trap(theta, **p), 4)
        for label, p in FUZZY_SETS.items()
    }


def dominant_category(memberships: Dict[str, float]) -> str:
    return max(memberships, key=memberships.get)


def time_to_emergence(theta: float, confidence: float) -> Dict:
    T_MAX = 5.0
    ALPHA = 1.8
    tte = float(np.clip(T_MAX * (1.0 - theta ** ALPHA), 0.2, T_MAX))
    half_ci = (1.0 - confidence) * 1.5
    return {
        "point_estimate_years": round(tte, 2),
        "ci_lower_years":       round(max(0.2, tte - half_ci), 2),
        "ci_upper_years":       round(min(T_MAX, tte + half_ci), 2),
    }


def _is_recent(date_val: Optional[str], years: int = 2) -> bool:
    if not date_val:
        return False
    try:
        d = datetime.fromisoformat(str(date_val)[:10])
        return d >= datetime.now().replace(year=datetime.now().year - years)
    except Exception:
        return False


def _yoy_growth_signal(date_strings: List[Optional[str]]) -> float:
    this_year = sum(1 for d in date_strings if _is_recent(d, years=1))
    last_year = sum(1 for d in date_strings
                    if d and not _is_recent(d, years=1) and _is_recent(d, years=2))
    if last_year == 0:
        return min(1.0, this_year / 10.0)
    growth = (this_year - last_year) / last_year
    return float(np.clip((growth + 1.0) / 2.0, 0.0, 1.0))


def compute_job_signals(
    entity_uri:        str,
    job_items:         List[dict],
    total_job_count:   int,
) -> Dict[str, float]:
    docs = [it for it in job_items if entity_uri in it.get("skills", [])]
    if not docs:
        return {k: 0.0 for k in JOB_IRT_PARAMS}

    posting_density = min(1.0, len(docs) / max(total_job_count * 0.3, 1))

    recent = [it for it in docs if _is_recent(it.get("upload_date"), years=1)]
    recency_intensity = min(1.0, len(recent) / len(docs))

    geo_set = set()
    for it in docs:
        loc = it.get("country") or it.get("location_code") or ""
        if loc:
            geo_set.add(str(loc)[:2].upper())
    geo_spread = min(1.0, len(geo_set) / 10.0)

    nace_set = set()
    for it in docs:
        sectors = it.get("sectors")
        if isinstance(sectors, list):
            for s in sectors:
                if s:
                    nace_set.add(str(s)[:2].upper())
        elif sectors:
            nace_set.add(str(sectors)[:2].upper())
    cross_sector = min(1.0, len(nace_set) / 5.0)

    yoy = _yoy_growth_signal([it.get("upload_date") for it in docs])

    occ_set = set()
    for it in docs:
        occ = it.get("occupation_id") or it.get("occupations")
        if occ:
            occ_set.add(str(occ))
    occ_breadth = min(1.0, len(occ_set) / 10.0)

    return {
        "posting_density":       round(posting_density,   4),
        "recency_intensity":     round(recency_intensity, 4),
        "geo_spread":            round(geo_spread,        4),
        "cross_sector_adoption": round(cross_sector,      4),
        "yoy_growth_rate":       round(yoy,               4),
        "occupation_breadth":    round(occ_breadth,       4),
    }


def compute_signals(
    entity_uri:        str,
    items_by_source:   Dict[str, List[dict]],
    entity_doc_counts: Dict[str, int],
) -> Dict[str, float]:
    pol_items = items_by_source.get("policies",    [])
    prj_items = items_by_source.get("projects",    [])
    wp_items  = items_by_source.get("white_papers", [])
    all_items = pol_items + prj_items + wp_items

    def in_entity(lst: List[dict]) -> List[dict]:
        return [it for it in lst if entity_uri in it.get("skills", [])]

    pol_docs = in_entity(pol_items)
    prj_docs = in_entity(prj_items)
    wp_docs  = in_entity(wp_items)
    all_docs = pol_docs + prj_docs + wp_docs

    wp_density  = min(1.0, len(wp_docs)  / max(len(wp_items)  * 0.3, 1))
    prj_density = min(1.0, len(prj_docs) / max(len(prj_items) * 0.3, 1))
    pol_density = min(1.0, len(pol_docs) / max(len(pol_items) * 0.3, 1))

    pol_intensity = (
        min(1.0, len([it for it in pol_docs if _is_recent(
            it.get("publication_date") or it.get("date"), years=2
        )]) / len(pol_docs))
        if pol_docs else 0.0
    )

    geo_set = set()
    for it in all_docs:
        loc = it.get("location_code") or it.get("country") or ""
        if loc:
            geo_set.add(str(loc)[:2].upper())
    geo_spread = min(1.0, len(geo_set) / 10.0)

    nace_set = set()
    for it in all_docs:
        nace = it.get("nace_code") or it.get("sector") or ""
        if nace:
            nace_set.add(str(nace)[:2].upper())
    cross_sector = min(1.0, len(nace_set) / 5.0)

    dates = [
        it.get("publication_date") or it.get("start_date") or it.get("date")
        for it in all_docs
    ]
    yoy = _yoy_growth_signal(dates)

    return {
        "whitepaper_density":    round(wp_density,    4),
        "project_density":       round(prj_density,   4),
        "policy_density":        round(pol_density,   4),
        "policy_intensity":      round(pol_intensity, 4),
        "geo_spread":            round(geo_spread,    4),
        "cross_sector_adoption": round(cross_sector,  4),
        "yoy_growth_rate":       round(yoy,           4),
    }


def _source_mix(entity_uri: str, items_by_source: Dict[str, List[dict]]) -> Dict[str, int]:
    counts = {src: sum(1 for it in items if entity_uri in it.get("skills", []))
              for src, items in items_by_source.items()}
    total = sum(counts.values())
    if total == 0:
        return {src: 0 for src in items_by_source}
    return {src: round(c / total * 100) for src, c in counts.items()}


_LT_RECS: Dict[str, List[Dict]] = {
    "breakthrough": [
        {
            "action":  "Begin strategic workforce transformation now",
            "detail":  "Emergence is imminent with high confidence. Begin workforce transformation immediately: identify capability gaps, establish dedicated competency centres, and initiate senior hires before the market tightens.",
            "owner":   "C-Suite / Workforce Planning",
            "urgency": "Immediate — within 6 months",
        },
        {
            "action":  "Establish R&D and academic partnerships",
            "detail":  "Co-fund or collaborate with the R&D projects already driving this technology. Early institutional partnerships provide preferential access to talent and proprietary know-how.",
            "owner":   "Innovation Lead",
            "urgency": "Within 12 months",
        },
        {
            "action":  "Develop future-ready job description frameworks",
            "detail":  "Define and publish role profiles and competency requirements before the market converges on standards. Early clarity accelerates targeted recruitment and creates an employer brand advantage.",
            "owner":   "HR Strategy / L&D",
            "urgency": "Within 12 months",
        },
    ],
    "emerging": [
        {
            "action":  "Launch exploratory internal capability programmes",
            "detail":  "Establish pilot training to develop internal expertise. Identify 2–3 internal champions who can lead capability building as the technology approaches mainstream adoption.",
            "owner":   "L&D Lead",
            "urgency": "Within 12 months",
        },
        {
            "action":  "Track regulatory trajectory",
            "detail":  "Growing policy signals indicate increasing institutional attention. Monitor relevant EU regulatory developments (e.g. AI Act, Green Deal, ENISA guidelines) and pre-align internal capabilities with likely compliance requirements.",
            "owner":   "Risk & Compliance",
            "urgency": "Ongoing — quarterly review",
        },
    ],
    "niche": [
        {
            "action":  "Assess strategic relevance before investing",
            "detail":  "Niche signals exist but mainstream adoption remains uncertain. Conduct an internal strategic alignment review before committing training or recruitment budgets.",
            "owner":   "HR Strategy",
            "urgency": "At next strategy review cycle",
        },
    ],
    "speculative": [
        {
            "action":  "Horizon scanning only — do not invest yet",
            "detail":  "Signals are predominantly from early-stage research. Maintain awareness through a horizon scanning programme (quarterly reviews of R&D publications and policy consultations) without committing organizational resources.",
            "owner":   "Innovation / Strategy",
            "urgency": "Horizon monitoring only",
        },
    ],
}


def _long_term_recommendations(
        category: str,
        entity_label: str,
        theta: float,
        tte: Dict,
        signals: Dict[str, float],
        source_mix: Optional[Dict[str, int]] = None,
        sector: str = "Unknown",
        entity_type: str = "skill",
) -> Dict:
    ci_width = tte["ci_upper_years"] - tte["ci_lower_years"]
    ci_note = (
        f" Note: the confidence interval spans {ci_width:.1f} years, indicating meaningful estimation uncertainty — "
        "staged investment and quarterly signal reassessment are advised before committing large resources."
        if ci_width > 2.0 else
        f" The confidence interval is narrow ({ci_width:.1f} years), indicating a reliable projection."
    )

    # Dominant signal interpretation
    pol = signals.get("policy_density", 0)
    rd = signals.get("project_density", 0)
    wp = signals.get("whitepaper_density", 0)
    yoy = signals.get("yoy_growth_rate", 0)
    geo = signals.get("geo_spread", 0)
    cross = signals.get("cross_sector_adoption", 0)

    strongest = max([("policy attention", pol), ("R&D funding", rd),
                     ("industry white paper discourse", wp), ("year-on-year growth", yoy)],
                    key=lambda x: x[1])

    geo_interp = (
        "broad EU-wide geographic spread" if geo > 0.6
        else "moderate cross-country diffusion" if geo > 0.3
        else "geographically concentrated — limited EU-wide diffusion so far"
    )
    cross_interp = (
        "strong cross-sector adoption — resilient across multiple NACE contexts" if cross > 0.6
        else "moderate cross-sector presence" if cross > 0.3
        else "sector-concentrated — adoption not yet diffusing beyond primary sectors"
    )

    src = source_mix or {}

    system = (
        "You are a workforce strategy advisor generating long-term capability briefings "
        "for a senior HR strategist using the SKILLAB EMERGE framework. "
        "You MUST respond with valid JSON only — no prose, no numbered lists, no markdown. "
        "Your entire response must be a single JSON object matching the required structure."
    )

    user = f"""You are preparing a 5-year capability briefing for the HR strategist 
operating in the {sector} sector, based on the EMERGE framework applied to European 
policy documents, R&D projects, and industry white papers.

The analysis concerns the {entity_type} "{entity_label}".

== EMERGE PROFILE ==

Emergence Category:          {category.upper()}
Emergence Quotient (EQ):     {round(theta * 100)} / 100
Theta (latent maturity 0–1): {theta}

Time-to-Emergence Projection:
  Point estimate:             {tte.get("point_estimate_years")} years
  Confidence interval:        {tte.get("ci_lower_years")} – {tte.get("ci_upper_years")} years
  Interval assessment:       {ci_note}

IRT Signal Profile (0–1 scale):
  Policy document signal:     {pol} — {"strong institutional attention" if pol > 0.5 else "limited policy discourse so far"}
  R&D project signal:         {rd} — {"active funded research pipeline" if rd > 0.5 else "limited R&D investment signal"}
  White paper signal:         {wp} — {"industry has crystallised attention" if wp > 0.5 else "early-stage industry discourse"}
  Year-on-year growth:        {yoy} — {"accelerating mentions" if yoy > 0.5 else "stable or slowing mentions"}
  Geographic spread (EU):     {geo} — {geo_interp}
  Cross-sector adoption:      {cross} — {cross_interp}
  Strongest signal:           {strongest[0]} ({strongest[1]:.2f})

Document Source Mix:
  Policy documents:           {src.get("policies", "N/A")}%
  R&D projects:               {src.get("projects", "N/A")}%
  White papers:               {src.get("white_papers", "N/A")}%

== EMERGENCE CATEGORY GUIDANCE ==

BREAKTHROUGH → Technology is on the verge of mainstream labour market diffusion. 
  Convey immediacy. Recommend decisive action within 6–12 months. 
  Reference the short TTE and the strongest signal as the primary talent pipeline.

EMERGING → Clear multi-signal confirmation but not yet mainstream. 
  Convey active preparation. Recommend pilot programmes, partnership exploration, 
  and regulatory monitoring. Reference the TTE range as the planning window.

NICHE → Some signal exists but diffusion is narrow and uncertain. 
  Convey cautious assessment. Recommend internal review before committing budgets. 
  Reference geographic or cross-sector concentration as the risk factor.

SPECULATIVE → Early-stage discourse only. 
  Convey horizon monitoring. No resource commitment warranted. 
  Reference low signal values and wide CI as the basis for restraint.

== YOUR TASK ==

Generate three recommendations, one per dimension:
1. strategic_workforce_planning — 3–5 year capability investment and role pipeline
2. partnerships_and_pipeline — university, R&D, and institutional partnerships
3. regulatory_and_compliance — regulatory trajectory and compliance readiness

Each recommendation MUST:
- Be 3–4 sentences long
- Address the HR strategist in second person ("you should...")
- Cite at least two specific signal or projection values by number
- Reference "{entity_label}" by name at least once
- Reference the "{sector}" sector context
- Apply the category guidance above — do not give generic foresight advice
- Acknowledge uncertainty if the CI width exceeds 2 years
- Avoid referencing any external knowledge about "{entity_label}" beyond what the metrics show
"""

    schema = {
        "type": "object",
        "properties": {
            "strategic_workforce_planning": {"type": "string"},
            "partnerships_and_pipeline": {"type": "string"},
            "regulatory_and_compliance": {"type": "string"},
        },
        "required": ["strategic_workforce_planning", "partnerships_and_pipeline", "regulatory_and_compliance"],
        "additionalProperties": False,
    }

    result = _chat_llm_json(system, user, schema)
    if result:
        return result

    log.warning(f"[Recs] LLM failed for '{entity_label}' — using static fallback")
    static = _LT_RECS.get(category, _LT_RECS["niche"])
    return {
        "strategic_workforce_planning": static[0]["detail"] if len(static) > 0 else "",
        "partnerships_and_pipeline": static[1]["detail"] if len(static) > 1 else static[0]["detail"],
        "regulatory_and_compliance": static[-1]["detail"],
    }


def _long_term_job_recommendations(
        category: str,
        entity_label: str,
        theta: float,
        tte: Dict,
        signals: Dict[str, float],
        sector: str = "Unknown",
        entity_type: str = "occupation",
) -> Dict:
    ci_width = tte["ci_upper_years"] - tte["ci_lower_years"]
    ci_note = (
        f" Note: the confidence interval spans {ci_width:.1f} years — "
        "staged commitment and quarterly reassessment are advised."
        if ci_width > 2.0 else
        f" The projection confidence interval is narrow ({ci_width:.1f} years), supporting reliable planning."
    )

    pd_ = signals.get("posting_density", 0)
    ri = signals.get("recency_intensity", 0)
    geo = signals.get("geo_spread", 0)
    cross = signals.get("cross_sector_adoption", 0)
    yoy = signals.get("yoy_growth_rate", 0)
    occ = signals.get("occupation_breadth", 0)

    pd_interp = "high volume of job postings" if pd_ > 0.5 else "limited posting volume so far"
    ri_interp = "recent postings dominating — strong recency signal" if ri > 0.6 else "postings distributed across historical window — no strong recency surge"
    occ_interp = "required across a broad range of occupational roles" if occ > 0.6 else "concentrated within a narrow occupational cluster"

    system = (
        "You are a workforce strategy advisor generating long-term occupation capability briefings "
        "for a senior HR strategist using the SKILLAB EMERGE framework applied to job posting data. "
        "You MUST respond with valid JSON only — no prose, no numbered lists, no markdown. "
        "Your entire response must be a single JSON object matching the required structure."
    )

    user = f"""You are preparing a 5-year capability briefing for the HR strategist 
operating in the {sector} sector. This analysis is derived from job posting data 
(not policy or research documents) using the EMERGE framework.

The analysis concerns the {entity_type} "{entity_label}" in the {sector} sector.

== EMERGE PROFILE (JOB-BASED) ==

Emergence Category:          {category.upper()}
Emergence Quotient (EQ):     {round(theta * 100)} / 100
Theta (latent maturity 0–1): {theta}

Time-to-Emergence Projection:
  Point estimate:             {tte.get("point_estimate_years")} years
  Confidence interval:        {tte.get("ci_lower_years")} – {tte.get("ci_upper_years")} years
  Interval assessment:       {ci_note}

Job Market Signal Profile (0–1 scale):
  Posting density:            {pd_} — {pd_interp}
  Recency intensity:          {ri} — {ri_interp}
  Geographic spread:          {geo} — {"broad multi-country presence" if geo > 0.5 else "geographically concentrated postings"}
  Cross-sector adoption:      {cross} — {"demanded across multiple NACE sectors" if cross > 0.5 else "sector-concentrated demand"}
  Year-on-year growth:        {yoy} — {"accelerating posting volume" if yoy > 0.5 else "stable or slowing posting volume"}
  Occupation breadth:         {occ} — {occ_interp}

== EMERGENCE CATEGORY GUIDANCE ==

BREAKTHROUGH → Posting signals confirm imminent mainstream labour market diffusion.
  Convey immediacy. Decisive action within 6–12 months. 
  Reference TTE and the strongest job market signal.

EMERGING → Multiple job market signals confirm trajectory but mainstream not yet reached.
  Convey active preparation. Pilot programmes, pipeline building, partnership exploration.
  Reference TTE range as the planning window.

NICHE → Some signal exists but posting volume and geographic spread remain narrow.
  Convey cautious assessment before committing L&D or recruitment budgets.
  Reference concentration risk (geographic or cross-sector).

SPECULATIVE → Weak signals across all dimensions.
  Convey horizon monitoring only. No resource commitment warranted.
  Reference low signal values as the basis for restraint.

== YOUR TASK ==

Generate three recommendations, one per dimension:
1. strategic_workforce_planning — 3–5 year role pipeline and capability investment
2. partnerships_and_pipeline — talent sourcing partnerships and pipeline development
3. regulatory_and_compliance — compliance readiness relevant to this occupation in {sector}

Each recommendation MUST:
- Be 3–4 sentences long
- Address the HR strategist in second person ("you should...")
- Cite at least two specific job market signal values by number
- Reference "{entity_label}" by name at least once
- Reference the "{sector}" sector context explicitly
- Apply the category guidance above — do not give generic workforce advice
- Acknowledge uncertainty if the CI width exceeds 2 years
- Ground every recommendation strictly in the job market signals — no external knowledge about "{entity_label}"
"""

    schema = {
        "type": "object",
        "properties": {
            "strategic_workforce_planning": {"type": "string"},
            "partnerships_and_pipeline": {"type": "string"},
            "regulatory_and_compliance": {"type": "string"},
        },
        "required": ["strategic_workforce_planning", "partnerships_and_pipeline", "regulatory_and_compliance"],
        "additionalProperties": False,
    }

    result = _chat_llm_json(system, user, schema)
    if result:
        return result

    log.warning(f"[Recs] LLM failed for '{entity_label}' — using static fallback")
    static = _LT_RECS.get(category, _LT_RECS["niche"])
    return {
        "strategic_workforce_planning": static[0]["detail"] if len(static) > 0 else "",
        "partnerships_and_pipeline": static[1]["detail"] if len(static) > 1 else static[0]["detail"],
        "regulatory_and_compliance": static[-1]["detail"],
    }

def run_long_term_skills_from_jobs(
    job_items:  List[dict],
    label_dict: Dict[str, str],
    top_n:      int = 50,
) -> Dict:
    log.info("[LT/Skills/Jobs] starting long-term skill emergence pipeline from jobs")

    skill_counts: Dict[str, int] = defaultdict(int)
    for item in job_items:
        for skill in item.get("skills", []):
            skill_counts[skill] += 1

    if not skill_counts:
        log.warning("[LT/Skills/Jobs] no skills found — returning empty result")
        return {"metadata": {}, "skills": [], "sector_summary": {"total_entities_analyzed": 0}}

    top_skills = sorted(skill_counts, key=skill_counts.get, reverse=True)[:top_n]
    log.info(f"[LT/Skills/Jobs] {len(skill_counts)} unique skills, analyzing top {len(top_skills)}")

    results = []
    for idx, uri in enumerate(top_skills, 1):
        if idx % 10 == 0 or idx == len(top_skills):
            log.info(f"  ...EMERGE/Jobs progress: {idx}/{len(top_skills)}")

        signals     = compute_job_signals(uri, job_items, len(job_items))
        theta, conf = estimate_theta(signals)
        membs       = fuzzy_memberships(theta)
        dom_cat     = dominant_category(membs)
        tte         = time_to_emergence(theta, conf)
        label       = label_dict.get(uri, uri)
        recs        = _long_term_recommendations(dom_cat, label, theta, tte, signals)

        results.append({
            "uri":                   uri,
            "label":                 label,
            "emergence_quotient":    round(theta * 100),
            "theta":                 theta,
            "confidence":            conf,
            "time_to_emergence":     tte,
            "dominant_category":     dom_cat,
            "fuzzy_memberships":     membs,
            "irt_signals":           {
                k: {"value": v, "description": JOB_IRT_PARAMS[k]["desc"]}
                for k, v in signals.items()
            },
            "total_job_mentions":    skill_counts.get(uri, 0),
            "recommendations":       recs,
        })

    results.sort(key=lambda x: x["theta"], reverse=True)

    cat_dist: Dict[str, int] = defaultdict(int)
    for r in results:
        cat_dist[r["dominant_category"]] += 1

    log.info(f"[LT/Skills/Jobs] COMPLETE: {len(results)} skills — {dict(cat_dist)}")

    return {
        "metadata": {
            "analysis_type":           "long_term_skills_from_jobs",
            "framework":               "EMERGE (IRT 2PL + Trapezoidal Fuzzy Logic)",
            "forecast_horizon_years":  5,
            "n_irt_signals":           len(JOB_IRT_PARAMS),
            "fuzzy_sets":              list(FUZZY_SETS.keys()),
            "total_records_retrieved": len(job_items),
            "analysis_date":           datetime.now().isoformat(),
        },
        "skills": results,
        "sector_summary": {
            "total_entities_analyzed":  len(results),
            "total_jobs_ingested":      len(job_items),
            "category_distribution":    dict(cat_dist),
            "top_breakthrough":         [r["label"] for r in results if r["dominant_category"] == "breakthrough"][:3],
            "top_emerging":             [r["label"] for r in results if r["dominant_category"] == "emerging"][:3],
            "top_speculative":          [r["label"] for r in results if r["dominant_category"] == "speculative"][:3],
        },
    }

def run_long_term_skills(
    items_by_source: Dict[str, List[dict]],
    label_dict:      Dict[str, str],
    top_n:           int = 50,
) -> Dict:
    log.info("[LT/Skills] starting long-term skill emergence pipeline")
    all_items = [it for src in items_by_source.values() for it in src]

    log.info("[LT/Skills] counting skill appearances across the corpus...")
    skill_counts: Dict[str, int] = defaultdict(int)
    for item in all_items:
        for skill in item.get("skills", []):
            skill_counts[skill] += 1

    if not skill_counts:
        log.warning("[LT/Skills] no skills found in corpus — returning empty result")
        return {"metadata": {}, "skills": [], "sector_summary": {"total_entities_analyzed": 0}}

    top_skills = sorted(skill_counts, key=skill_counts.get, reverse=True)[:top_n]
    log.info(f"[LT/Skills] {len(skill_counts)} unique skills found, analyzing top {len(top_skills)}")

    results = []
    for idx, uri in enumerate(top_skills, 1):
        if idx % 10 == 0 or idx == len(top_skills):
            log.info(f"  ...EMERGE progress: {idx}/{len(top_skills)}")
        signals    = compute_signals(uri, items_by_source, dict(skill_counts))
        theta, conf = estimate_theta(signals)
        membs      = fuzzy_memberships(theta)
        dom_cat    = dominant_category(membs)
        tte        = time_to_emergence(theta, conf)
        src_mix    = _source_mix(uri, items_by_source)
        label      = label_dict.get(uri, uri)
        recs = _long_term_recommendations(
            dom_cat, label, theta, tte, signals,
            source_mix=_source_mix(uri, items_by_source),
            entity_type="skill",
        )

        results.append({
            "uri":                    uri,
            "label":                  label,
            "emergence_quotient":     round(theta * 100),
            "theta":                  theta,
            "confidence":             conf,
            "time_to_emergence":      tte,
            "dominant_category":      dom_cat,
            "fuzzy_memberships":      membs,
            "irt_signals":            {
                k: {"value": v, "description": IRT_PARAMS[k]["desc"]}
                for k, v in signals.items()
            },
            "source_mix_pct":         src_mix,
            "total_document_mentions": skill_counts.get(uri, 0),
            "recommendations":        recs,
        })

    results.sort(key=lambda x: x["theta"], reverse=True)

    cat_dist: Dict[str, int] = defaultdict(int)
    for r in results:
        cat_dist[r["dominant_category"]] += 1

    log.info(f"[LT/Skills] COMPLETE: {len(results)} skills classified — {dict(cat_dist)}")

    return {
        "metadata": {
            "analysis_type":           "long_term_skills",
            "framework":               "EMERGE (IRT 2PL + Trapezoidal Fuzzy Logic)",
            "forecast_horizon_years":  5,
            "n_irt_signals":           len(IRT_PARAMS),
            "fuzzy_sets":              list(FUZZY_SETS.keys()),
            "total_records_retrieved": len(all_items),
            "analysis_date":           datetime.now().isoformat(),
        },
        "skills": results,
        "sector_summary": {
            "total_entities_analyzed":   len(results),
            "total_documents_ingested":  len(all_items),
            "documents_by_source":       {src: len(items) for src, items in items_by_source.items()},
            "category_distribution":     dict(cat_dist),
            "top_breakthrough":          [r["label"] for r in results if r["dominant_category"] == "breakthrough"][:3],
            "top_emerging":              [r["label"] for r in results if r["dominant_category"] == "emerging"][:3],
            "top_speculative":           [r["label"] for r in results if r["dominant_category"] == "speculative"][:3],
        },
    }

def run_long_term_occupations_from_jobs(
    job_items:  List[dict],
    label_dict: Dict[str, str],
    top_n:      int = 50,
    sector: str = "Unknown"
) -> Dict:
    log.info("[LT/Occupations/Jobs] starting long-term occupation emergence pipeline from jobs")
    total_jobs = len(job_items)

    # Step 1: compute skill-level thetas from jobs
    log.info("[LT/Occupations/Jobs] Step 1: computing skill thetas from jobs...")
    skills_output = run_long_term_skills_from_jobs(job_items, label_dict, top_n=500)
    skill_theta_map: Dict[str, Tuple[float, float]] = {
        s["uri"]: (s["theta"], s["confidence"])
        for s in skills_output.get("skills", [])
    }
    log.info(f"[LT/Occupations/Jobs] Step 1 done: {len(skill_theta_map)} skill thetas computed")

    if not skill_theta_map:
        log.warning("[LT/Occupations/Jobs] no skill thetas — returning empty result")
        return {"metadata": {}, "occupations": [], "sector_summary": {"total_occupations_analyzed": 0}}

    # Step 2: build occupation→skill map directly from job co-occurrence
    log.info("[LT/Occupations/Jobs] Step 2: building occupation→skill map from jobs...")
    occ_skills_map: Dict[str, List[str]] = defaultdict(list)
    for item in job_items:
        occ = item.get("occupation_id") or item.get("occupations")
        if occ:
            # handle both single URI and list of URIs
            occ_list = occ if isinstance(occ, list) else [occ]
            for single_occ in occ_list:
                if single_occ:
                    for skill in item.get("skills", []):
                        occ_skills_map[str(single_occ)].append(skill)

    if not occ_skills_map:
        log.warning("[LT/Occupations/Jobs] no occupation-skill mapping found — returning empty result")
        return {
            "metadata":       {},
            "occupations":    [],
            "sector_summary": {"total_occupations_analyzed": 0, "reason": "No occupation-skill mapping found."},
        }

    # Step 3: aggregate skill thetas to occupation level
    log.info(f"[LT/Occupations/Jobs] Step 3: aggregating across {len(occ_skills_map)} occupations...")
    occ_results = []
    for idx, (occ_uri, assoc_skills) in enumerate(occ_skills_map.items(), 1):
        if idx % 25 == 0:
            log.info(f"  ...aggregation progress: {idx}/{len(occ_skills_map)}")

        theta_conf_pairs = [
            skill_theta_map[s] for s in assoc_skills if s in skill_theta_map
        ]
        if not theta_conf_pairs:
            continue

        thetas    = [tc[0] for tc in theta_conf_pairs]
        confs     = [tc[1] for tc in theta_conf_pairs]
        weights   = np.array(thetas) + 0.01
        agg_theta = float(np.average(thetas, weights=weights))
        agg_conf  = float(np.mean(confs))

        membs   = fuzzy_memberships(agg_theta)
        dom_cat = dominant_category(membs)
        tte     = time_to_emergence(agg_theta, agg_conf)
        label   = label_dict.get(occ_uri, occ_uri)

        # aggregate job signals across constituent skills
        agg_sigs: Dict[str, List[float]] = defaultdict(list)
        for s_uri in assoc_skills:
            if s_uri in skill_theta_map:
                s_signals = compute_job_signals(s_uri, job_items, total_jobs)
                for k, v in s_signals.items():
                    agg_sigs[k].append(v)
        final_sigs = {k: round(float(np.mean(v)), 4) for k, v in agg_sigs.items()}

        recs = _long_term_job_recommendations(
            dom_cat, label, theta, tte, signals,
            sector=sector,
            entity_type="skill",
        )

        occ_results.append({
            "uri":                  occ_uri,
            "label":                label,
            "emergence_quotient":   round(agg_theta * 100),
            "theta":                round(agg_theta, 4),
            "confidence":           round(agg_conf, 4),
            "time_to_emergence":    tte,
            "dominant_category":    dom_cat,
            "fuzzy_memberships":    membs,
            "irt_signals":          {
                k: {"value": v, "description": JOB_IRT_PARAMS.get(k, {}).get("desc", k)}
                for k, v in final_sigs.items()
            },
            "n_associated_skills":  len(assoc_skills),
            "recommendations":      recs,
        })

    occ_results.sort(key=lambda x: x["theta"], reverse=True)
    occ_results = occ_results[:top_n]

    cat_dist: Dict[str, int] = defaultdict(int)
    for r in occ_results:
        cat_dist[r["dominant_category"]] += 1

    log.info(f"[LT/Occupations/Jobs] COMPLETE: {len(occ_results)} occupations — {dict(cat_dist)}")

    return {
        "metadata": {
            "analysis_type":           "long_term_occupations_from_jobs",
            "framework":               "EMERGE (IRT 2PL + Fuzzy Logic) — Job-based Skill Aggregation",
            "forecast_horizon_years":  5,
            "aggregation_method":      "theta-weighted mean of constituent skill scores",
            "total_records_retrieved": total_jobs,
            "analysis_date":           datetime.now().isoformat(),
        },
        "occupations": occ_results,
        "sector_summary": {
            "total_occupations_analyzed": len(occ_results),
            "total_jobs_ingested":        total_jobs,
            "category_distribution":      dict(cat_dist),
            "top_breakthrough":           [r["label"] for r in occ_results if r["dominant_category"] == "breakthrough"][:3],
            "top_emerging":               [r["label"] for r in occ_results if r["dominant_category"] == "emerging"][:3],
        },
    }


def run_long_term_occupations(
    items_by_source: Dict[str, List[dict]],
    label_dict:      Dict[str, str],
    esco_df:         pd.DataFrame,
    sector:          str,
    top_n:           int = 50,
) -> Dict:
    log.info("[LT/Occupations] starting long-term occupation emergence pipeline")
    all_items = [it for src in items_by_source.values() for it in src]
    total_docs = len(all_items)

    log.info("[LT/Occupations] Step 1: running skill-level EMERGE analysis...")
    skills_output = run_long_term_skills(items_by_source, label_dict, top_n=500)
    skill_theta_map: Dict[str, Tuple[float, float]] = {
        s["uri"]: (s["theta"], s["confidence"])
        for s in skills_output.get("skills", [])
    }
    log.info(f"[LT/Occupations] Step 1 done: {len(skill_theta_map)} skill thetas computed")

    if not skill_theta_map:
        log.warning("[LT/Occupations] no skill thetas available — returning empty result")
        return {"metadata": {}, "occupations": [], "sector_summary": {"total_occupations_analyzed": 0}}

    log.info("[LT/Occupations] Step 2: building occupation→skill mapping...")
    occ_skills_map: Dict[str, List[str]] = defaultdict(list)

    if "occupation_uri" in esco_df.columns and "conceptUri" in esco_df.columns:
        log.info("[LT/Occupations] Using explicit ESCO occupation_uri mapping.")
        for _, row in esco_df.iterrows():
            occ = row.get("occupation_uri")
            skill = row.get("conceptUri")
            if occ and skill and not pd.isna(occ) and not pd.isna(skill):
                occ_skills_map[str(occ)].append(str(skill))
    else:
        log.info("[LT/Occupations] No occupation_uri column — using job co-occurrence fallback.")
        job_body = {
            "sectors":        sector,
            "min_upload_date":  (datetime.now().replace(year=datetime.now().year - 5)).strftime("%Y-%m-%d"),
            "max_upload_date":  datetime.now().strftime("%Y-%m-%d"),
        }
        job_items = paginate_all(job_body, endpoint="jobs")
        for item in job_items:
            occ = item.get("occupation_id") or item.get("occupation")
            if occ:
                for skill in item.get("skills", []):
                    occ_skills_map[str(occ)].append(skill)

    if not occ_skills_map:
        log.warning("[LT/Occupations] no occupation-skill mapping found — returning empty result")
        return {
            "metadata": {},
            "occupations": [],
            "sector_summary": {"total_occupations_analyzed": 0, "reason": "No occupation-skill mapping found."},
        }

    log.info(f"[LT/Occupations] Step 3: aggregating theta across {len(occ_skills_map)} occupations...")
    occ_results = []
    for idx, (occ_uri, assoc_skills) in enumerate(occ_skills_map.items(), 1):
        if idx % 25 == 0:
            log.info(f"  ...aggregation progress: {idx}/{len(occ_skills_map)}")
        theta_conf_pairs = [
            skill_theta_map[s] for s in assoc_skills if s in skill_theta_map
        ]
        if not theta_conf_pairs:
            continue

        thetas = [tc[0] for tc in theta_conf_pairs]
        confs  = [tc[1] for tc in theta_conf_pairs]

        weights  = np.array(thetas) + 0.01
        agg_theta = float(np.average(thetas, weights=weights))
        agg_conf  = float(np.mean(confs))

        membs   = fuzzy_memberships(agg_theta)
        dom_cat = dominant_category(membs)
        tte     = time_to_emergence(agg_theta, agg_conf)
        label   = label_dict.get(occ_uri, occ_uri)

        agg_sigs: Dict[str, List[float]] = defaultdict(list)
        for s_uri in assoc_skills:
            if s_uri in skill_theta_map:
                s_signals = compute_signals(s_uri, items_by_source,
                                            {u: 1 for u in assoc_skills})
                for k, v in s_signals.items():
                    agg_sigs[k].append(v)
        final_sigs = {k: round(float(np.mean(v)), 4) for k, v in agg_sigs.items()}

        recs = _long_term_recommendations(dom_cat, label, agg_theta, tte, final_sigs)

        occ_results.append({
            "uri":                  occ_uri,
            "label":                label,
            "emergence_quotient":   round(agg_theta * 100),
            "theta":                round(agg_theta, 4),
            "confidence":           round(agg_conf, 4),
            "time_to_emergence":    tte,
            "dominant_category":    dom_cat,
            "fuzzy_memberships":    membs,
            "irt_signals":          {
                k: {"value": v, "description": IRT_PARAMS.get(k, {}).get("desc", k)}
                for k, v in final_sigs.items()
            },
            "n_associated_skills":  len(assoc_skills),
            "recommendations":      recs,
        })

    occ_results.sort(key=lambda x: x["theta"], reverse=True)
    occ_results = occ_results[:top_n]

    cat_dist: Dict[str, int] = defaultdict(int)
    for r in occ_results:
        cat_dist[r["dominant_category"]] += 1

    log.info(f"[LT/Occupations] COMPLETE: {len(occ_results)} occupations — {dict(cat_dist)}")

    return {
        "metadata": {
            "analysis_type":           "long_term_occupations",
            "framework":               "EMERGE (IRT 2PL + Fuzzy Logic) — Skill Aggregation",
            "forecast_horizon_years":  5,
            "aggregation_method":      "theta-weighted mean of constituent skill scores",
            "total_records_retrieved": total_docs,
            "analysis_date":           datetime.now().isoformat(),
        },
        "occupations": occ_results,
        "sector_summary": {
            "total_occupations_analyzed": len(occ_results),
            "total_documents_ingested":   total_docs,
            "documents_by_source":        {src: len(items) for src, items in items_by_source.items()},
            "category_distribution":      dict(cat_dist),
            "top_breakthrough":           [r["label"] for r in occ_results if r["dominant_category"] == "breakthrough"][:3],
            "top_emerging":               [r["label"] for r in occ_results if r["dominant_category"] == "emerging"][:3],
        },
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — ENDPOINTS
# ══════════════════════════════════════════════════════════════════

def _default_dates(years_back: int = 3) -> Tuple[str, str]:
    today = datetime.now()
    start = today.replace(year=today.year - years_back).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    return start, end


def _build_cache_key(*parts: Optional[str]) -> str:
    return "_".join(
        re.sub(r"[^\w]", "-", str(p))[:40]
        for p in parts
        if p is not None
    )


# ── 4.1  SHORT-TERM SKILLS ────────────────────────────────────────

@app.get("/shorttermanalysis/skills")
def short_term_skills(
    organization: Optional[str] = Query(None, description="Organization / company name (display label)."),
    sector: Optional[str] = Query(None, description="NACE sector code to filter job postings."),
    top_n: int = Query(50, ge=1, le=200, description="Max entities per sector."),
):
    log.info("=" * 70)
    log.info(f"[ENDPOINT] /shorttermanalysis/skills — "
             f"sector={sector}, organization={organization}")
    log.info("=" * 70)

    _ensure_folder()

    cache_key = _build_cache_key(
        "sts_skills", sector, organization
    )
    file_path = os.path.join(FOLDER, cache_key)

    cached, exists = _load_or_init_cache(file_path)
    if exists:
        return cached

    try:
        _, label_dict = load_esco_mapping()

        request_body: dict = {}
        if sector:
            request_body["sectors"] = sector
        if organization:
            request_body["organization_names"] = organization


        log.info(f"[ShortTerm/Skills] fetching jobs with filters: {request_body}")
        items = paginate_all(request_body, endpoint="jobs")
        log.info(f"[ShortTerm/Skills] {len(items)} job records retrieved")

        if not items:
            result = {
                "status":  "no_data",
                "message": "No job records found for the given filters.",
                "result":  [],
            }
            _save_cache(file_path, result)
            return result

        results_by_sector = run_short_term_analysis_by_sector(
            items, mode="skills", label_dict=label_dict, top_n=top_n,
            organization=organization or "Unknown",
            user_sector=sector or "Unknown",  # ← add this
        )

        output = {
            "metadata": {
                "analysis_type":           "short_term_skills_by_sector",
                "forecast_horizon_years":  3,
                "historical_window_years": 3,
                "forecast_model":          "holt_winters_with_linear_fallback",
                "total_records_retrieved": len(items),
                "sectors_analyzed":        sorted(results_by_sector.keys()),
                "analysis_date":           datetime.now().isoformat(),
            },
            "filters_applied": {
                "organization": organization,
                "sector":       sector
            },
            "results_by_sector": results_by_sector,
        }

        _save_cache(file_path, output)
        log.info("[ENDPOINT] /shorttermanalysis/skills — DONE")
        return output

    except Exception as e:
        _error_cache(file_path, e)
        raise e


# ── 4.2  SHORT-TERM OCCUPATIONS ───────────────────────────────────

@app.get("/shorttermanalysis/occupations")
def short_term_occupations(
    organization: Optional[str] = Query(None, description="Organization name."),
    sector: Optional[str] = Query(None, description="NACE sector code."),
    top_n: int = Query(50, ge=1, le=200, description="Max occupations per sector."),
):
    log.info("=" * 70)
    log.info(f"[ENDPOINT] /shorttermanalysis/occupations — "
             f"sector={sector}, organization={organization}")
    log.info("=" * 70)

    _ensure_folder()

    cache_key = _build_cache_key(
        "sts_occupations", sector, organization
    )
    file_path = os.path.join(FOLDER, cache_key)

    cached, exists = _load_or_init_cache(file_path)
    if exists:
        return cached

    try:
        _, label_dict = load_esco_mapping()

        request_body: dict = {}
        if sector:
            request_body["sectors"] = sector
        if organization:
            request_body["organization_names"] = organization

        log.info(f"[ShortTerm/Occupations] fetching jobs with filters: {request_body}")
        items = paginate_all(request_body, endpoint="jobs")
        log.info(f"[ShortTerm/Occupations] {len(items)} job records retrieved")

        if not items:
            result = {
                "status":  "no_data",
                "message": "No job records found for the given filters.",
                "result":  [],
            }
            _save_cache(file_path, result)
            return result

        results_by_sector = run_short_term_analysis_by_sector(
            items, mode="occupations", label_dict=label_dict, top_n=top_n,
            organization=organization or "Unknown",
            user_sector=sector or "Unknown",  # ← add this
        )
        output = {
            "metadata": {
                "analysis_type":           "short_term_occupations_by_sector",
                "forecast_horizon_years":  3,
                "historical_window_years": 3,
                "forecast_model":          "holt_winters_with_linear_fallback",
                "total_records_retrieved": len(items),
                "sectors_analyzed":        sorted(results_by_sector.keys()),
                "analysis_date":           datetime.now().isoformat(),
            },
            "filters_applied": {
                "organization": organization,
                "sector":       sector
            },
            "results_by_sector": results_by_sector,
        }

        _save_cache(file_path, output)
        log.info("[ENDPOINT] /shorttermanalysis/occupations — DONE")
        return output

    except Exception as e:
        _error_cache(file_path, e)
        raise e


# ── 4.3  LONG-TERM SKILLS ─────────────────────────────────────────

@app.get("/longtermanalysis/skills")
def long_term_skills(
    keywords: Optional[str] = Query(None, description="Comma-separated keywords to filter documents."),
    keyword_logic: str = Query("or", description="Combine keywords with 'and' or 'or'."),
    top_n: int = Query(50, ge=1, le=200, description="Max skills in output"),
):
    log.info("=" * 70)
    log.info("=" * 70)

    _ensure_folder()

    filename = f"longtermanalysis_skills_{keywords}"
    file_path = os.path.join(FOLDER, filename)

    cached, exists = _load_or_init_cache(file_path)
    if exists:
        return cached

    try:
        _, label_dict = load_esco_mapping()

        keyword_list = _parse_csv_str(keywords)  # already defined in Section 1

        base = {}
        if keyword_list:
            base["keywords"] = keyword_list
            base["keywords_logic"] = keyword_logic.lower()

        pol_body = {**base}
        prj_body = {**base}

        policy_items  = paginate_all(pol_body, endpoint="law-policies")
        project_items = paginate_all(prj_body, endpoint="projects")

        items_by_source = {
            "policies":    policy_items,
            "projects":    project_items,
        }
        total = sum(len(v) for v in items_by_source.values())
        log.info(f"[LongTerm/Skills] Total documents: {total} "
                 f"(policies={len(policy_items)}, projects={len(project_items)}")

        if total == 0:
            result = {
                "status":  "no_data",
                "message": "No policy, project, or white paper records found for the given sector.",
                "result":  [],
            }
            _save_cache(file_path, result)
            return result

        output = run_long_term_skills(items_by_source, label_dict, top_n=top_n)
        output["metadata"].update({
            "keywords": keyword_list,
            "keyword_logic": keyword_logic,
        })

        _save_cache(file_path, output)
        log.info("[ENDPOINT] /longtermanalysis/skills — DONE")
        return output

    except Exception as e:
        _error_cache(file_path, e)
        raise e


# ── 4.4  LONG-TERM OCCUPATIONS ────────────────────────────────────


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 (continued) — LONG-TERM FROM JOBS
# ══════════════════════════════════════════════════════════════════

JOB_IRT_PARAMS: Dict[str, Dict] = {
    "posting_density":       {"a": 1.2, "b": 0.25, "desc": "Skill frequency across job postings"},
    "recency_intensity":     {"a": 2.0, "b": 0.50, "desc": "Share of postings in last 12 months vs all"},
    "geo_spread":            {"a": 1.3, "b": 0.40, "desc": "Geographic spread of postings"},
    "cross_sector_adoption": {"a": 1.5, "b": 0.55, "desc": "Sector diversity of postings"},
    "yoy_growth_rate":       {"a": 1.8, "b": 0.38, "desc": "Year-over-year posting growth rate"},
    "occupation_breadth":    {"a": 1.1, "b": 0.30, "desc": "Distinct occupations requiring this skill"},
}


def compute_job_signals(
    entity_uri:        str,
    job_items:         List[dict],
    total_job_count:   int,
) -> Dict[str, float]:
    docs = [it for it in job_items if entity_uri in it.get("skills", [])]
    if not docs:
        return {k: 0.0 for k in JOB_IRT_PARAMS}

    posting_density = min(1.0, len(docs) / max(total_job_count * 0.3, 1))

    recent = [it for it in docs if _is_recent(it.get("upload_date"), years=1)]
    recency_intensity = min(1.0, len(recent) / len(docs))

    geo_set = set()
    for it in docs:
        loc = it.get("country") or it.get("location_code") or ""
        if loc:
            geo_set.add(str(loc)[:2].upper())
    geo_spread = min(1.0, len(geo_set) / 10.0)

    nace_set = set()
    for it in docs:
        sectors = it.get("sectors")
        if isinstance(sectors, list):
            for s in sectors:
                if s:
                    nace_set.add(str(s)[:2].upper())
        elif sectors:
            nace_set.add(str(sectors)[:2].upper())
    cross_sector = min(1.0, len(nace_set) / 5.0)

    yoy = _yoy_growth_signal([it.get("upload_date") for it in docs])

    occ_set = set()
    for it in docs:
        occ = it.get("occupation_id") or it.get("occupations")
        if occ:
            occ_set.add(str(occ))
    occ_breadth = min(1.0, len(occ_set) / 10.0)

    return {
        "posting_density":       round(posting_density,   4),
        "recency_intensity":     round(recency_intensity, 4),
        "geo_spread":            round(geo_spread,        4),
        "cross_sector_adoption": round(cross_sector,      4),
        "yoy_growth_rate":       round(yoy,               4),
        "occupation_breadth":    round(occ_breadth,       4),
    }


def run_long_term_skills_from_jobs(
    job_items:  List[dict],
    label_dict: Dict[str, str],
    top_n:      int = 50,
    sector: str = "Unknown"
) -> Dict:
    log.info("[LT/Skills/Jobs] starting long-term skill emergence pipeline from jobs")

    skill_counts: Dict[str, int] = defaultdict(int)
    for item in job_items:
        for skill in item.get("skills", []):
            skill_counts[skill] += 1

    if not skill_counts:
        log.warning("[LT/Skills/Jobs] no skills found — returning empty result")
        return {"metadata": {}, "skills": [], "sector_summary": {"total_entities_analyzed": 0}}

    top_skills = sorted(skill_counts, key=skill_counts.get, reverse=True)[:top_n]
    log.info(f"[LT/Skills/Jobs] {len(skill_counts)} unique skills, analyzing top {len(top_skills)}")

    results = []
    for idx, uri in enumerate(top_skills, 1):
        if idx % 10 == 0 or idx == len(top_skills):
            log.info(f"  ...EMERGE/Jobs progress: {idx}/{len(top_skills)}")

        signals     = compute_job_signals(uri, job_items, len(job_items))
        theta, conf = estimate_theta(signals)
        membs       = fuzzy_memberships(theta)
        dom_cat     = dominant_category(membs)
        tte         = time_to_emergence(theta, conf)
        label       = label_dict.get(uri, uri)
        recs        = _long_term_recommendations(dom_cat, label, theta, tte, signals)

        results.append({
            "uri":                   uri,
            "label":                 label,
            "emergence_quotient":    round(theta * 100),
            "theta":                 theta,
            "confidence":            conf,
            "time_to_emergence":     tte,
            "dominant_category":     dom_cat,
            "fuzzy_memberships":     membs,
            "irt_signals":           {
                k: {"value": v, "description": JOB_IRT_PARAMS[k]["desc"]}
                for k, v in signals.items()
            },
            "total_job_mentions":    skill_counts.get(uri, 0),
            "recommendations":       recs,
        })

    results.sort(key=lambda x: x["theta"], reverse=True)

    cat_dist: Dict[str, int] = defaultdict(int)
    for r in results:
        cat_dist[r["dominant_category"]] += 1

    log.info(f"[LT/Skills/Jobs] COMPLETE: {len(results)} skills — {dict(cat_dist)}")

    return {
        "metadata": {
            "analysis_type":           "long_term_skills_from_jobs",
            "framework":               "EMERGE (IRT 2PL + Trapezoidal Fuzzy Logic)",
            "forecast_horizon_years":  5,
            "n_irt_signals":           len(JOB_IRT_PARAMS),
            "fuzzy_sets":              list(FUZZY_SETS.keys()),
            "total_records_retrieved": len(job_items),
            "analysis_date":           datetime.now().isoformat(),
        },
        "skills": results,
        "sector_summary": {
            "total_entities_analyzed":  len(results),
            "total_jobs_ingested":      len(job_items),
            "category_distribution":    dict(cat_dist),
            "top_breakthrough":         [r["label"] for r in results if r["dominant_category"] == "breakthrough"][:3],
            "top_emerging":             [r["label"] for r in results if r["dominant_category"] == "emerging"][:3],
            "top_speculative":          [r["label"] for r in results if r["dominant_category"] == "speculative"][:3],
        },
    }


def run_long_term_occupations_from_jobs(
    job_items:  List[dict],
    label_dict: Dict[str, str],
    top_n:      int = 50,
    sector: str = "Unknown"
) -> Dict:
    log.info("[LT/Occupations/Jobs] starting long-term occupation emergence pipeline from jobs")
    total_jobs = len(job_items)

    # Step 1: compute skill-level thetas from jobs
    log.info("[LT/Occupations/Jobs] Step 1: computing skill thetas from jobs...")
    skills_output = run_long_term_skills_from_jobs(job_items, label_dict, top_n=500)
    skill_theta_map: Dict[str, Tuple[float, float]] = {
        s["uri"]: (s["theta"], s["confidence"])
        for s in skills_output.get("skills", [])
    }
    log.info(f"[LT/Occupations/Jobs] Step 1 done: {len(skill_theta_map)} skill thetas computed")

    if not skill_theta_map:
        log.warning("[LT/Occupations/Jobs] no skill thetas — returning empty result")
        return {"metadata": {}, "occupations": [], "sector_summary": {"total_occupations_analyzed": 0}}

    # Step 2: build occupation→skill map directly from job co-occurrence
    log.info("[LT/Occupations/Jobs] Step 2: building occupation→skill map from jobs...")
    occ_skills_map: Dict[str, List[str]] = defaultdict(list)
    for item in job_items:
        occ = item.get("occupation_id") or item.get("occupations")
        if occ:
            for skill in item.get("skills", []):
                occ_skills_map[str(occ)].append(skill)

    if not occ_skills_map:
        log.warning("[LT/Occupations/Jobs] no occupation-skill mapping found — returning empty result")
        return {
            "metadata":       {},
            "occupations":    [],
            "sector_summary": {"total_occupations_analyzed": 0, "reason": "No occupation-skill mapping found."},
        }

    # Step 3: aggregate skill thetas to occupation level
    log.info(f"[LT/Occupations/Jobs] Step 3: aggregating across {len(occ_skills_map)} occupations...")
    occ_results = []
    for idx, (occ_uri, assoc_skills) in enumerate(occ_skills_map.items(), 1):
        if idx % 25 == 0:
            log.info(f"  ...aggregation progress: {idx}/{len(occ_skills_map)}")

        theta_conf_pairs = [
            skill_theta_map[s] for s in assoc_skills if s in skill_theta_map
        ]
        if not theta_conf_pairs:
            continue

        thetas    = [tc[0] for tc in theta_conf_pairs]
        confs     = [tc[1] for tc in theta_conf_pairs]
        weights   = np.array(thetas) + 0.01
        agg_theta = float(np.average(thetas, weights=weights))
        agg_conf  = float(np.mean(confs))

        membs   = fuzzy_memberships(agg_theta)
        dom_cat = dominant_category(membs)
        tte     = time_to_emergence(agg_theta, agg_conf)
        label   = label_dict.get(occ_uri, occ_uri)

        # aggregate job signals across constituent skills
        agg_sigs: Dict[str, List[float]] = defaultdict(list)
        for s_uri in assoc_skills:
            if s_uri in skill_theta_map:
                s_signals = compute_job_signals(s_uri, job_items, total_jobs)
                for k, v in s_signals.items():
                    agg_sigs[k].append(v)
        final_sigs = {k: round(float(np.mean(v)), 4) for k, v in agg_sigs.items()}

        recs = _long_term_job_recommendations(
            dom_cat, label, agg_theta, tte, final_sigs,
            sector=sector,
            entity_type="occupation",
        )

        occ_results.append({
            "uri":                  occ_uri,
            "label":                label,
            "emergence_quotient":   round(agg_theta * 100),
            "theta":                round(agg_theta, 4),
            "confidence":           round(agg_conf, 4),
            "time_to_emergence":    tte,
            "dominant_category":    dom_cat,
            "fuzzy_memberships":    membs,
            "irt_signals":          {
                k: {"value": v, "description": JOB_IRT_PARAMS.get(k, {}).get("desc", k)}
                for k, v in final_sigs.items()
            },
            "n_associated_skills":  len(assoc_skills),
            "recommendations":      recs,
        })

    occ_results.sort(key=lambda x: x["theta"], reverse=True)
    occ_results = occ_results[:top_n]

    cat_dist: Dict[str, int] = defaultdict(int)
    for r in occ_results:
        cat_dist[r["dominant_category"]] += 1

    log.info(f"[LT/Occupations/Jobs] COMPLETE: {len(occ_results)} occupations — {dict(cat_dist)}")

    return {
        "metadata": {
            "analysis_type":           "long_term_occupations_from_jobs",
            "framework":               "EMERGE (IRT 2PL + Fuzzy Logic) — Job-based Skill Aggregation",
            "forecast_horizon_years":  5,
            "aggregation_method":      "theta-weighted mean of constituent skill scores",
            "total_records_retrieved": total_jobs,
            "analysis_date":           datetime.now().isoformat(),
        },
        "occupations": occ_results,
        "sector_summary": {
            "total_occupations_analyzed": len(occ_results),
            "total_jobs_ingested":        total_jobs,
            "category_distribution":      dict(cat_dist),
            "top_breakthrough":           [r["label"] for r in occ_results if r["dominant_category"] == "breakthrough"][:3],
            "top_emerging":               [r["label"] for r in occ_results if r["dominant_category"] == "emerging"][:3],
        },
    }


# ── 4.4  LONG-TERM OCCUPATIONS (from jobs) ───────────────────────

@app.get("/longtermanalysis/occupations")
def long_term_occupations(
    sector:       Optional[str] = Query(None, description="NACE sector code to filter job postings."),
    organization: Optional[str] = Query(None, description="Organization name to filter job postings."),
    top_n:        int           = Query(50, ge=1, le=200, description="Max occupations in output."),
):
    log.info("=" * 70)
    log.info(f"[ENDPOINT] /longtermanalysis/occupations — sector={sector}, organization={organization}")
    log.info("=" * 70)

    _ensure_folder()

    cache_key = _build_cache_key("lt_occupations", sector, organization)
    file_path = os.path.join(FOLDER, cache_key)

    cached, exists = _load_or_init_cache(file_path)
    if exists:
        return cached

    try:
        _, label_dict = load_esco_mapping()

        request_body: dict = {}
        if sector:
            request_body["sectors"] = sector
        if organization:
            request_body["organization_names"] = organization

        log.info(f"[LongTerm/Occupations] fetching jobs with filters: {request_body}")
        job_items = paginate_all(request_body, endpoint="jobs")
        log.info(f"[LongTerm/Occupations] {len(job_items)} job records retrieved")

        if not job_items:
            result = {
                "status":  "no_data",
                "message": "No job records found for the given filters.",
                "result":  [],
            }
            _save_cache(file_path, result)
            return result

        output = run_long_term_occupations_from_jobs(job_items, label_dict, top_n=top_n, sector=sector or "Unknown")
        output["metadata"].update({
            "filters_applied": {
                "sector":       sector,
                "organization": organization,
            }
        })

        _save_cache(file_path, output)
        log.info("[ENDPOINT] /longtermanalysis/occupations — DONE")
        return output

    except Exception as e:
        _error_cache(file_path, e)
        raise e