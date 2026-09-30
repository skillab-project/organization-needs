"""
demand_analysis.py
==================
SKILLAB — Demand Analysis Service
Endpoints (all take the organization name in the X-User-Organization header):
  GET /shorttermanalysis/skills       — US #23  Short-term skill demand
  GET /shorttermanalysis/occupations  — US #24  Short-term occupation demand
  GET /longtermanalysis/skills        — US #23  Long-term skill emergence (EMERGE, job-based)
  GET /longtermanalysis/occupations   — US #24  Long-term occupation emergence (EMERGE, job-based)

Endpoints are asynchronous: the first call for a (endpoint, organization, top_n)
combination starts the analysis in the background and returns 202 {"status": "started"}.
Calling again with the same values returns 202 {"status": "in_progress"} while it runs,
and the full result (200) once it has finished. A failed analysis is reported once
({"status": "failed"}) and the next call starts it again.

Rerun: add ?rerun=true to discard the stored result and compute the analysis again
(same organization and top_n). It returns 202 {"status": "started"} like a first call;
poll WITHOUT rerun afterwards. If that analysis is already running, rerun=true just
reports {"status": "in_progress"} — it never starts a duplicate job.

Every endpoint runs two analyses and returns both:
  organization_analysis — job ads of the organization from the Hiring Management API
  sector_analysis       — the organization's sectors are read from the Employee Management API,
                          then job postings for each sector are pulled from the SKILLAB Tracker
"""

import time
import re
import math
import json
import logging
import threading
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple

from fastapi import BackgroundTasks, FastAPI, Header, Query, Response
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
    root_path=os.getenv("ROOT_PATH", ""),
)

FOLDER = Path("completed_anlyses")


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — SHARED INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════

def get_token() -> str:
    res = req.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD})
    return res.text.replace('"', "")


def api_extract(request_body: dict, page: int, endpoint: str, token: str) -> dict:
    """Single-page retrieval."""
    page_size = 300
    params = {"page": page, "page_size": page_size}
    data = req.post(
        f"{API}/{endpoint}",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        data=request_body,
    )
    return data.json()


def paginate_all(request_body: dict, endpoint: str) -> List[dict]:
    """
    Retrieve every page from an endpoint and return a flat list of items.
    """
    log.info(f"[{endpoint}] fetching page 1...")
    token = get_token()
    first = api_extract(request_body, page=1, endpoint=endpoint, token=token)
    count = first.get("count", 0)
    n_pages = max(1, math.ceil(count / 300))
    log.info(f"[{endpoint}] {count} records found — {n_pages} page(s) total")

    items: List[dict] = list(first.get("items", []))
    for page in range(2, n_pages + 1):
        log.info(f"[{endpoint}] fetching page {page}/{n_pages}...")
        chunk = api_extract(request_body, page=page, endpoint=endpoint, token=token)
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


def _in_progress_stub() -> Dict[str, Any]:
    return {"status": "in_progress", "message": "Analysis is being computed", "result": None}


def _save_cache(file_path: str, data: Any) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)
    log.info(f"Saved cache: {file_path}")


def _occupations_of(item: dict) -> List[str]:
    """Occupation keys of a job: a single name/URI (hiring) or a list of URIs (tracker)."""
    occ = item.get("occupation_id") or item.get("occupations")
    if not occ:
        return []
    return [str(o) for o in (occ if isinstance(occ, list) else [occ]) if o]


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
        for occ in _occupations_of(item):
            acc[occ][q] += 1
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


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — LONG-TERM ANALYSIS  (EMERGE framework, job-based)
# ══════════════════════════════════════════════════════════════════

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
        p_params = JOB_IRT_PARAMS.get(key)
        if p_params is None:
            continue
        prob = float(np.clip(_irt_p(theta, p_params["a"], p_params["b"]), 1e-9, 1 - 1e-9))
        ll += val * math.log(prob) + (1 - val) * math.log(1 - prob)
    return ll


def _grid_theta(signal_vec: Dict[str, float]) -> float:
    grid = [i / 100.0 for i in range(1, 100)]
    return max(grid, key=lambda th: _log_likelihood(th, signal_vec))


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
    return {label: round(_trap(theta, **p), 4) for label, p in FUZZY_SETS.items()}


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


def _item_geo(item: dict) -> str:
    return str(item.get("country") or item.get("location_code") or "")


def _item_sectors(item: dict) -> List[str]:
    val = item.get("sectors")
    if isinstance(val, list):
        return [str(s) for s in val if s]
    return [str(val)] if val else []


def _available_signals(job_items: List[dict]) -> List[str]:
    """
    Hiring-management jobs carry no location or sector. Scoring those signals
    as 0 would drag every theta down, so they are left out of the IRT model
    when the dataset has no such field at all.
    """
    keys = ["posting_density", "recency_intensity", "yoy_growth_rate", "occupation_breadth"]
    if any(_item_geo(it) for it in job_items):
        keys.append("geo_spread")
    if any(_item_sectors(it) for it in job_items):
        keys.append("cross_sector_adoption")
    return keys


def compute_job_signals(
    entity_uri:      str,
    job_items:       List[dict],
    total_job_count: int,
    signal_keys:     List[str],
) -> Dict[str, float]:
    docs = [it for it in job_items if entity_uri in it.get("skills", [])]
    if not docs:
        return {k: 0.0 for k in signal_keys}

    geo_set  = {_item_geo(it)[:2].upper() for it in docs if _item_geo(it)}
    nace_set = {s[:2].upper() for it in docs for s in _item_sectors(it)}
    occ_set  = {o for it in docs for o in _occupations_of(it)}
    recent   = [it for it in docs if _is_recent(it.get("upload_date"), years=1)]

    all_signals = {
        "posting_density":       min(1.0, len(docs) / max(total_job_count * 0.3, 1)),
        "recency_intensity":     min(1.0, len(recent) / len(docs)),
        "geo_spread":            min(1.0, len(geo_set) / 10.0),
        "cross_sector_adoption": min(1.0, len(nace_set) / 5.0),
        "yoy_growth_rate":       _yoy_growth_signal([it.get("upload_date") for it in docs]),
        "occupation_breadth":    min(1.0, len(occ_set) / 10.0),
    }
    return {k: round(all_signals[k], 4) for k in signal_keys}


_LT_RECS: Dict[str, List[Dict]] = {
    "breakthrough": [
        {"detail": "Emergence is imminent with high confidence. Begin workforce transformation immediately: identify capability gaps, establish dedicated competency centres, and initiate senior hires before the market tightens."},
        {"detail": "Build sourcing partnerships with universities and training providers now. Early institutional partnerships provide preferential access to talent before competition for it peaks."},
        {"detail": "Define and publish role profiles and competency requirements before the market converges on standards, and check them against upcoming regulatory requirements in your sector."},
    ],
    "emerging": [
        {"detail": "Establish pilot training to develop internal expertise. Identify 2–3 internal champions who can lead capability building as demand approaches the mainstream."},
        {"detail": "Start building a talent pipeline through internships, traineeships and targeted partnerships while the market is still forming."},
        {"detail": "Monitor relevant EU regulatory developments (e.g. AI Act, Green Deal, ENISA guidelines) and pre-align internal capabilities with likely compliance requirements."},
    ],
    "niche": [
        {"detail": "Niche signals exist but mainstream adoption remains uncertain. Conduct an internal strategic alignment review before committing training or recruitment budgets."},
        {"detail": "Keep a light-touch relationship with specialist providers so you can source this capability on demand rather than building it in-house."},
        {"detail": "No dedicated compliance action is warranted yet. Review at the next strategy cycle."},
    ],
    "speculative": [
        {"detail": "Signals are weak across all dimensions. Maintain awareness through quarterly horizon scanning without committing organizational resources."},
        {"detail": "No partnership investment is warranted yet. Revisit if posting volume or growth picks up."},
        {"detail": "No compliance action is warranted at this stage."},
    ],
}


def _long_term_job_recommendations(
        category: str,
        entity_label: str,
        theta: float,
        tte: Dict,
        signals: Dict[str, float],
        sector: str = "Unknown",
        entity_type: str = "skill",
        data_scope: str = "sector",
) -> Dict:
    ci_width = tte["ci_upper_years"] - tte["ci_lower_years"]
    ci_note = (
        f" Note: the confidence interval spans {ci_width:.1f} years — "
        "staged commitment and quarterly reassessment are advised."
        if ci_width > 2.0 else
        f" The projection confidence interval is narrow ({ci_width:.1f} years), supporting reliable planning."
    )

    interp = {
        "posting_density":       lambda v: "high volume of job postings" if v > 0.5 else "limited posting volume so far",
        "recency_intensity":     lambda v: "recent postings dominating — strong recency signal" if v > 0.6 else "postings spread across the historical window — no strong recency surge",
        "geo_spread":            lambda v: "broad multi-country presence" if v > 0.5 else "geographically concentrated postings",
        "cross_sector_adoption": lambda v: "demanded across multiple NACE sectors" if v > 0.5 else "sector-concentrated demand",
        "yoy_growth_rate":       lambda v: "accelerating posting volume" if v > 0.5 else "stable or slowing posting volume",
        "occupation_breadth":    lambda v: "required across a broad range of occupational roles" if v > 0.6 else "concentrated within a narrow occupational cluster",
    }
    signal_lines = "\n".join(
        f"  {JOB_IRT_PARAMS[k]['desc']:<45} {v} — {interp[k](v)}"
        for k, v in signals.items() if k in interp
    )
    scope_text = (
        f"the job ads of the organization (hiring-management data, sector context: {sector})"
        if data_scope == "organization" else
        f"job postings in the {sector} sector (SKILLAB Tracker data)"
    )

    system = (
        "You are a workforce strategy advisor generating long-term capability briefings "
        "for a senior HR strategist using the SKILLAB EMERGE framework applied to job posting data. "
        "You MUST respond with valid JSON only — no prose, no numbered lists, no markdown. "
        "Your entire response must be a single JSON object matching the required structure."
    )

    user = f"""You are preparing a 5-year capability briefing for an HR strategist.
This analysis is derived from {scope_text} using the EMERGE framework.

The analysis concerns the {entity_type} "{entity_label}".

== EMERGE PROFILE (JOB-BASED) ==

Emergence Category:          {category.upper()}
Emergence Quotient (EQ):     {round(theta * 100)} / 100
Theta (latent maturity 0–1): {theta}

Time-to-Emergence Projection:
  Point estimate:             {tte.get("point_estimate_years")} years
  Confidence interval:        {tte.get("ci_lower_years")} – {tte.get("ci_upper_years")} years
  Interval assessment:       {ci_note}

Job Market Signal Profile (0–1 scale):
{signal_lines}

== EMERGENCE CATEGORY GUIDANCE ==

BREAKTHROUGH → Posting signals confirm imminent mainstream labour market diffusion.
  Convey immediacy. Decisive action within 6–12 months.
  Reference TTE and the strongest job market signal.

EMERGING → Multiple job market signals confirm trajectory but mainstream not yet reached.
  Convey active preparation. Pilot programmes, pipeline building, partnership exploration.
  Reference TTE range as the planning window.

NICHE → Some signal exists but posting volume and spread remain narrow.
  Convey cautious assessment before committing L&D or recruitment budgets.

SPECULATIVE → Weak signals across all dimensions.
  Convey horizon monitoring only. No resource commitment warranted.

== YOUR TASK ==

Generate three recommendations, one per dimension:
1. strategic_workforce_planning — 3–5 year role pipeline and capability investment
2. partnerships_and_pipeline — talent sourcing partnerships and pipeline development
3. regulatory_and_compliance — compliance readiness relevant to this {entity_type} in {sector}

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
            "partnerships_and_pipeline":    {"type": "string"},
            "regulatory_and_compliance":    {"type": "string"},
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
        "strategic_workforce_planning": static[0]["detail"],
        "partnerships_and_pipeline":    static[1]["detail"],
        "regulatory_and_compliance":    static[2]["detail"],
    }


def run_long_term_skills_from_jobs(
    job_items:        List[dict],
    label_dict:       Dict[str, str],
    top_n:            int = 50,
    sector:           str = "Unknown",
    data_scope:       str = "sector",
    with_recommendations: bool = True,
) -> Dict:
    log.info(f"[LT/Skills] starting on {len(job_items)} jobs, top_n={top_n}, recs={with_recommendations}")

    skill_counts: Dict[str, int] = defaultdict(int)
    for item in job_items:
        for skill in item.get("skills", []):
            skill_counts[skill] += 1

    if not skill_counts:
        log.warning("[LT/Skills] no skills found — returning empty result")
        return {"metadata": {}, "skills": [], "sector_summary": {"total_entities_analyzed": 0}}

    signal_keys = _available_signals(job_items)
    top_skills = sorted(skill_counts, key=skill_counts.get, reverse=True)[:top_n]
    log.info(f"[LT/Skills] {len(skill_counts)} unique skills, analyzing top {len(top_skills)} "
             f"with signals {signal_keys}")

    results = []
    for idx, uri in enumerate(top_skills, 1):
        if idx % 25 == 0 or idx == len(top_skills):
            log.info(f"  ...EMERGE progress: {idx}/{len(top_skills)}")

        signals     = compute_job_signals(uri, job_items, len(job_items), signal_keys)
        theta, conf = estimate_theta(signals)
        membs       = fuzzy_memberships(theta)
        dom_cat     = dominant_category(membs)
        tte         = time_to_emergence(theta, conf)
        label       = label_dict.get(uri, uri)
        recs = (
            _long_term_job_recommendations(dom_cat, label, theta, tte, signals,
                                           sector=sector, entity_type="skill", data_scope=data_scope)
            if with_recommendations else None
        )

        results.append({
            "uri":                uri,
            "label":              label,
            "emergence_quotient": round(theta * 100),
            "theta":              theta,
            "confidence":         conf,
            "time_to_emergence":  tte,
            "dominant_category":  dom_cat,
            "fuzzy_memberships":  membs,
            "irt_signals":        {
                k: {"value": v, "description": JOB_IRT_PARAMS[k]["desc"]}
                for k, v in signals.items()
            },
            "total_job_mentions": skill_counts.get(uri, 0),
            "recommendations":    recs,
        })

    results.sort(key=lambda x: x["theta"], reverse=True)

    cat_dist: Dict[str, int] = defaultdict(int)
    for r in results:
        cat_dist[r["dominant_category"]] += 1

    log.info(f"[LT/Skills] COMPLETE: {len(results)} skills — {dict(cat_dist)}")

    return {
        "metadata": {
            "analysis_type":           "long_term_skills_from_jobs",
            "framework":               "EMERGE (IRT 2PL + Trapezoidal Fuzzy Logic)",
            "forecast_horizon_years":  5,
            "irt_signals_used":        signal_keys,
            "fuzzy_sets":              list(FUZZY_SETS.keys()),
            "total_records_retrieved": len(job_items),
            "analysis_date":           datetime.now().isoformat(),
        },
        "skills": results,
        "sector_summary": {
            "total_entities_analyzed": len(results),
            "total_jobs_ingested":     len(job_items),
            "category_distribution":   dict(cat_dist),
            "top_breakthrough":        [r["label"] for r in results if r["dominant_category"] == "breakthrough"][:3],
            "top_emerging":            [r["label"] for r in results if r["dominant_category"] == "emerging"][:3],
            "top_speculative":         [r["label"] for r in results if r["dominant_category"] == "speculative"][:3],
        },
    }


def run_long_term_occupations_from_jobs(
    job_items:  List[dict],
    label_dict: Dict[str, str],
    top_n:      int = 50,
    sector:     str = "Unknown",
    data_scope: str = "sector",
) -> Dict:
    log.info(f"[LT/Occupations] starting on {len(job_items)} jobs, top_n={top_n}")
    total_jobs = len(job_items)

    # Step 1: skill thetas and signals, without LLM calls
    skills_output = run_long_term_skills_from_jobs(
        job_items, label_dict, top_n=500, with_recommendations=False,
    )
    skill_info: Dict[str, Dict] = {s["uri"]: s for s in skills_output.get("skills", [])}
    log.info(f"[LT/Occupations] Step 1 done: {len(skill_info)} skill thetas computed")

    if not skill_info:
        return {"metadata": {}, "occupations": [], "sector_summary": {"total_occupations_analyzed": 0}}

    # Step 2: occupation → skills from co-occurrence in the job postings
    occ_skills_map: Dict[str, set] = defaultdict(set)
    for item in job_items:
        for occ in _occupations_of(item):
            occ_skills_map[occ].update(item.get("skills", []))

    if not occ_skills_map:
        return {
            "metadata":       {},
            "occupations":    [],
            "sector_summary": {"total_occupations_analyzed": 0, "reason": "No occupation-skill mapping found."},
        }

    # Step 3: aggregate skill thetas and signals to occupation level
    log.info(f"[LT/Occupations] Step 3: aggregating across {len(occ_skills_map)} occupations...")
    scored = []
    for occ_uri, assoc_skills in occ_skills_map.items():
        known = [skill_info[s] for s in assoc_skills if s in skill_info]
        if not known:
            continue
        thetas    = [s["theta"] for s in known]
        agg_theta = float(np.average(thetas, weights=np.array(thetas) + 0.01))
        agg_conf  = float(np.mean([s["confidence"] for s in known]))

        agg_sigs: Dict[str, List[float]] = defaultdict(list)
        for s in known:
            for k, v in s["irt_signals"].items():
                agg_sigs[k].append(v["value"])

        scored.append({
            "uri":                 occ_uri,
            "theta":               round(agg_theta, 4),
            "confidence":          round(agg_conf, 4),
            "signals":             {k: round(float(np.mean(v)), 4) for k, v in agg_sigs.items()},
            "n_associated_skills": len(assoc_skills),
        })

    # Keep top_n before calling the LLM, so recommendations are only built for returned rows
    scored.sort(key=lambda x: x["theta"], reverse=True)
    scored = scored[:top_n]

    occ_results = []
    for idx, occ in enumerate(scored, 1):
        if idx % 10 == 0 or idx == len(scored):
            log.info(f"  ...occupation recommendations: {idx}/{len(scored)}")
        membs   = fuzzy_memberships(occ["theta"])
        dom_cat = dominant_category(membs)
        tte     = time_to_emergence(occ["theta"], occ["confidence"])
        label   = label_dict.get(occ["uri"], occ["uri"])
        recs = _long_term_job_recommendations(
            dom_cat, label, occ["theta"], tte, occ["signals"],
            sector=sector, entity_type="occupation", data_scope=data_scope,
        )
        occ_results.append({
            "uri":                 occ["uri"],
            "label":               label,
            "emergence_quotient":  round(occ["theta"] * 100),
            "theta":               occ["theta"],
            "confidence":          occ["confidence"],
            "time_to_emergence":   tte,
            "dominant_category":   dom_cat,
            "fuzzy_memberships":   membs,
            "irt_signals":         {
                k: {"value": v, "description": JOB_IRT_PARAMS[k]["desc"]}
                for k, v in occ["signals"].items()
            },
            "n_associated_skills": occ["n_associated_skills"],
            "recommendations":     recs,
        })

    cat_dist: Dict[str, int] = defaultdict(int)
    for r in occ_results:
        cat_dist[r["dominant_category"]] += 1

    log.info(f"[LT/Occupations] COMPLETE: {len(occ_results)} occupations — {dict(cat_dist)}")

    return {
        "metadata": {
            "analysis_type":           "long_term_occupations_from_jobs",
            "framework":               "EMERGE (IRT 2PL + Fuzzy Logic) — Job-based Skill Aggregation",
            "forecast_horizon_years":  5,
            "aggregation_method":      "theta-weighted mean of constituent skill scores",
            "irt_signals_used":        skills_output["metadata"].get("irt_signals_used"),
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


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — DATA SOURCES
# ══════════════════════════════════════════════════════════════════

HIRING_API       = os.getenv("HIRING_API",   "https://portal.skillab-project.eu/hiring-management").rstrip("/")
EMPLOYEE_API     = os.getenv("EMPLOYEE_API", "https://portal.skillab-project.eu/employee-management").rstrip("/")
PORTAL_API_TOKEN = os.getenv("PORTAL_API_TOKEN")  # optional, only if the portal gateway requires one
TRACKER_YEARS_BACK = int(os.getenv("TRACKER_YEARS_BACK", "3"))


def _portal_get(url: str, organization: Optional[str] = None, required: bool = False) -> Any:
    """GET on the SKILLAB portal with 3 attempts. Returns None on failure unless required=True."""
    headers = {"Accept": "application/json"}
    if organization:
        headers["X-User-Organization"] = organization
    if PORTAL_API_TOKEN:
        headers["Authorization"] = f"Bearer {PORTAL_API_TOKEN}"
    for attempt in range(3):
        try:
            resp = req.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning(f"[Portal] GET {url} attempt {attempt + 1} failed: {exc}")
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    if required:
        raise RuntimeError(f"Portal request failed: {url}")
    return None


# ── 4.1  Organization data: Hiring Management ────────────────────

def fetch_org_jobs(organization: str) -> Tuple[List[dict], Dict[str, str]]:
    """
    Pull every job ad of an organization from the Hiring Management API and
    reshape it into the item format the pipelines expect:
        {"id", "upload_date", "occupations", "skills": [skill_id, ...]}
    Returns (items, skill_labels) with skill_labels mapping skill_id -> title.
    """
    ads = _portal_get(f"{HIRING_API}/api/v1/jobAds", organization, required=True) or []
    log.info(f"[Hiring] {len(ads)} job ads for '{organization}'")

    items: List[dict] = []
    skill_labels: Dict[str, str] = {}
    for idx, ad in enumerate(ads, 1):
        job_id = ad.get("id")
        if job_id is None:
            continue

        # The summary has no date; the full job ad has publishDate
        detail = _portal_get(f"{HIRING_API}/api/v1/jobAds/{job_id}", organization) or {}
        skills = _portal_get(f"{HIRING_API}/api/v1/jobAds/{job_id}/interview-skills", organization) or []

        skill_ids = set()
        for s in skills:
            if s.get("id") is None:
                continue
            sid = str(s["id"])
            skill_ids.add(sid)
            skill_labels[sid] = s.get("title") or sid

        occupation = ad.get("occupationName") or (detail.get("occupation") or {}).get("title")

        items.append({
            "id":          job_id,
            "title":       ad.get("jobTitle"),
            "status":      ad.get("status"),
            "department":  ad.get("departmentName"),
            "upload_date": detail.get("publishDate"),
            "occupations": occupation,
            "skills":      sorted(skill_ids),
        })
        if idx % 25 == 0 or idx == len(ads):
            log.info(f"[Hiring] fetched details and skills for {idx}/{len(ads)} jobs")

    return items, skill_labels


# ── 4.2  Organization profile: Employee Management ───────────────

def fetch_org_profile(organization: str) -> Optional[dict]:
    """Find the organization in Employee Management by name (exact match first, then partial)."""
    orgs = _portal_get(f"{EMPLOYEE_API}/organizations", organization, required=True) or []
    target = organization.strip().lower()
    exact   = [o for o in orgs if (o.get("name") or "").strip().lower() == target]
    partial = [o for o in orgs if target in (o.get("name") or "").strip().lower()]
    match = (exact or partial or [None])[0]
    if match is None:
        log.warning(f"[Employee] organization '{organization}' not found among {len(orgs)} organizations")
        return None
    match = {**match, "sectors": _clean_sectors(match.get("sectors"))}
    log.info(f"[Employee] '{organization}' → id={match.get('id')}, sectors={match['sectors']}")
    return match


def _clean_sectors(raw: Any) -> List[str]:
    """
    Employee Management stores sectors with literal quotes, e.g. '"Computer programming activities"'.
    Strip quotes and whitespace, drop empties and duplicates, keep order.
    """
    if isinstance(raw, str):
        raw = [raw]
    cleaned: List[str] = []
    for s in raw or []:
        if s is None:
            continue
        name = str(s).strip().strip('"\'').strip()
        if name and name not in cleaned:
            cleaned.append(name)
    return cleaned


# ── 4.3  Sector data: SKILLAB Tracker ────────────────────────────

def fetch_sector_jobs(sector: str) -> List[dict]:
    start, end = _default_dates(TRACKER_YEARS_BACK)
    body = {"sectors": sector, "min_upload_date": start, "max_upload_date": end}
    log.info(f"[Tracker] fetching jobs with filters: {body}")
    return paginate_all(body, endpoint="jobs")


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — ORCHESTRATION & ENDPOINTS
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


# An analysis function takes (items, label_dict, sector_label, data_scope, organization) and returns a result dict
AnalysisFn = Any


def _organization_part(organization: str, profile: Optional[dict], analyze: AnalysisFn) -> Dict:
    part: Dict[str, Any] = {"data_source": "Hiring Management API (jobAds + interview-skills)"}
    try:
        items, skill_labels = fetch_org_jobs(organization)
        n_undated = sum(1 for it in items if not it["upload_date"])
        part["jobs_retrieved"] = len(items)
        part["jobs_without_publish_date"] = n_undated
        if n_undated:
            log.warning(f"[Org] {n_undated} job(s) without publishDate are excluded from date-based metrics")

        if not items or len(items) == n_undated:
            part.update({"status": "no_data",
                         "message": f"No dated job ads found for organization '{organization}'."})
            return part

        sector_label = ", ".join(profile.get("sectors") or []) if profile else ""
        part.update({"status": "ok",
                     **analyze(items, skill_labels, sector_label or "Unknown", "organization", organization)})
    except Exception as exc:
        log.exception(f"[Org] analysis failed for '{organization}'")
        part.update({"status": "error", "message": str(exc)})
    return part


def _sector_part(organization: str, profile: Optional[dict], analyze: AnalysisFn) -> Dict:
    part: Dict[str, Any] = {"data_source": "SKILLAB Tracker jobs (sectors from Employee Management)"}
    if profile is None:
        part.update({"status": "no_data", "sectors": [],
                     "message": f"Organization '{organization}' was not found in Employee Management."})
        return part

    sectors = [s for s in (profile.get("sectors") or []) if s]
    part["sectors"] = sectors
    if not sectors:
        part.update({"status": "no_data",
                     "message": f"Organization '{organization}' has no sectors in Employee Management."})
        return part

    _, esco_labels = load_esco_mapping()
    results_by_sector: Dict[str, Dict] = {}
    for sector in sectors:
        try:
            items = fetch_sector_jobs(sector)
            if not items:
                results_by_sector[sector] = {"status": "no_data",
                                             "message": f"No tracker jobs found for sector '{sector}'."}
                continue
            results_by_sector[sector] = {"status": "ok", "jobs_retrieved": len(items),
                                         **analyze(items, esco_labels, sector, "sector", organization)}
        except Exception as exc:
            log.exception(f"[Sector] analysis failed for sector '{sector}'")
            results_by_sector[sector] = {"status": "error", "message": str(exc)}

    statuses = {r["status"] for r in results_by_sector.values()}
    part["status"] = "ok" if "ok" in statuses else ("error" if "error" in statuses else "no_data")
    part["results_by_sector"] = results_by_sector
    return part


def _has_error(output: Dict) -> bool:
    if output["organization_analysis"].get("status") == "error":
        return True
    sec = output["sector_analysis"]
    return sec.get("status") == "error" or any(
        r.get("status") == "error" for r in sec.get("results_by_sector", {}).values()
    )


def _compute_combined(endpoint: str, organization: str, analyze: AnalysisFn) -> Dict:
    """Run both analyses (organization + sector) and return the combined output."""
    try:
        profile = fetch_org_profile(organization)
        profile_error = None
    except Exception as exc:
        log.exception("[Employee] organization lookup failed")
        profile, profile_error = None, str(exc)

    output = {
        "metadata": {
            "endpoint":      endpoint,
            "organization":  organization,
            "analysis_date": datetime.now().isoformat(),
        },
        "organization_profile":  profile,
        "organization_analysis": _organization_part(organization, profile, analyze),
        "sector_analysis":       _sector_part(organization, profile, analyze),
    }
    if profile_error:
        output["sector_analysis"].update({"status": "error",
                                          "message": f"Employee Management lookup failed: {profile_error}"})
    return output


# ── Background job bookkeeping ───────────────────────────────────
# Cache files (one per endpoint/organization/top_n) hold one of:
#   {"status": "in_progress", ...}  — the analysis is running
#   {"status": "failed", ...}       — the analysis failed; returned once, then removed so it can be retried
#   <full result>                   — the analysis finished successfully
# _RUNNING tracks jobs of this process, so an "in_progress" file left behind by a
# restart (job no longer running) is detected and the analysis is started again.
_RUNNING: set = set()
_RUNNING_LOCK = threading.Lock()


def _cache_path(endpoint: str, organization: str, top_n: int) -> str:
    return os.path.join(FOLDER, _build_cache_key(endpoint.strip("/").replace("/", "_"), organization, str(top_n)))


def _run_analysis_job(file_path: str, endpoint: str, organization: str, analyze: AnalysisFn) -> None:
    log.info("=" * 70)
    log.info(f"[JOB] {endpoint} — organization={organization} — started")
    log.info("=" * 70)
    try:
        output = _compute_combined(endpoint, organization, analyze)
        if _has_error(output):
            _save_cache(file_path, {"status": "failed",
                                    "message": "Part of the analysis failed. Call the endpoint again to retry.",
                                    "result": output})
        else:
            _save_cache(file_path, output)
        log.info(f"[JOB] {endpoint} — organization={organization} — DONE")
    except Exception as exc:
        log.exception(f"[JOB] {endpoint} — organization={organization} — FAILED")
        _save_cache(file_path, {"status": "failed",
                                "message": f"Analysis failed: {exc}. Call the endpoint again to retry.",
                                "result": None})
    finally:
        with _RUNNING_LOCK:
            _RUNNING.discard(file_path)


def _run_combined(endpoint: str, organization: str, top_n: int, analyze: AnalysisFn,
                  background_tasks: BackgroundTasks, response: Response, rerun: bool = False) -> Dict:
    """Return the finished result, or start/report the background analysis.

    rerun=True ignores (and replaces) a stored result or failure and starts the
    analysis again, unless it is already running."""
    log.info(f"[ENDPOINT] {endpoint} — organization={organization}, top_n={top_n}, rerun={rerun}")
    _ensure_folder()
    file_path = _cache_path(endpoint, organization, top_n)

    with _RUNNING_LOCK:
        if file_path in _RUNNING:
            response.status_code = 202
            return _in_progress_stub()

        if rerun and os.path.exists(file_path):
            log.info(f"Rerun requested — discarding stored result: {file_path}")
        elif os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            status = cached.get("status") if isinstance(cached, dict) else None
            if status == "failed":
                os.remove(file_path)          # report the failure once; the next call retries
                return cached
            if status != "in_progress":
                log.info(f"Cache hit: {file_path}")
                return cached
            log.warning(f"Stale in-progress file (job not running): {file_path} — restarting analysis")

        _RUNNING.add(file_path)
        _save_cache(file_path, _in_progress_stub())

    background_tasks.add_task(_run_analysis_job, file_path, endpoint, organization, analyze)
    response.status_code = 202
    return {"status": "started",
            "message": "Analysis has started. Call the same endpoint again to get its status or results.",
            "result": None}


# ── 5.1  SHORT-TERM SKILLS ────────────────────────────────────────

@app.get("/shorttermanalysis/skills")
def short_term_skills(
    background_tasks: BackgroundTasks,
    response: Response,
    organization: str = Header(..., alias="X-User-Organization", description="Organization name, e.g. 'eclipse'."),
    top_n: int = Query(50, ge=1, le=200, description="Max skills per analysis."),
    rerun: bool = Query(False, description="Discard the stored result and compute the analysis again."),
):
    def analyze(items, labels, sector, scope, org):
        return run_short_term_analysis(items, mode="skills", label_dict=labels, top_n=top_n,
                                       sector=sector, organization=org)
    return _run_combined("/shorttermanalysis/skills", organization, top_n, analyze, background_tasks, response, rerun)


# ── 5.2  SHORT-TERM OCCUPATIONS ───────────────────────────────────

@app.get("/shorttermanalysis/occupations")
def short_term_occupations(
    background_tasks: BackgroundTasks,
    response: Response,
    organization: str = Header(..., alias="X-User-Organization", description="Organization name, e.g. 'eclipse'."),
    top_n: int = Query(50, ge=1, le=200, description="Max occupations per analysis."),
    rerun: bool = Query(False, description="Discard the stored result and compute the analysis again."),
):
    def analyze(items, labels, sector, scope, org):
        return run_short_term_analysis(items, mode="occupations", label_dict=labels, top_n=top_n,
                                       sector=sector, organization=org)
    return _run_combined("/shorttermanalysis/occupations", organization, top_n, analyze, background_tasks, response, rerun)


# ── 5.3  LONG-TERM SKILLS ─────────────────────────────────────────

@app.get("/longtermanalysis/skills")
def long_term_skills(
    background_tasks: BackgroundTasks,
    response: Response,
    organization: str = Header(..., alias="X-User-Organization", description="Organization name, e.g. 'eclipse'."),
    top_n: int = Query(50, ge=1, le=200, description="Max skills per analysis."),
    rerun: bool = Query(False, description="Discard the stored result and compute the analysis again."),
):
    def analyze(items, labels, sector, scope, org):
        return run_long_term_skills_from_jobs(items, labels, top_n=top_n, sector=sector, data_scope=scope)
    return _run_combined("/longtermanalysis/skills", organization, top_n, analyze, background_tasks, response, rerun)


# ── 5.4  LONG-TERM OCCUPATIONS ────────────────────────────────────

@app.get("/longtermanalysis/occupations")
def long_term_occupations(
    background_tasks: BackgroundTasks,
    response: Response,
    organization: str = Header(..., alias="X-User-Organization", description="Organization name, e.g. 'eclipse'."),
    top_n: int = Query(50, ge=1, le=200, description="Max occupations per analysis."),
    rerun: bool = Query(False, description="Discard the stored result and compute the analysis again."),
):
    def analyze(items, labels, sector, scope, org):
        return run_long_term_occupations_from_jobs(items, labels, top_n=top_n, sector=sector, data_scope=scope)
    return _run_combined("/longtermanalysis/occupations", organization, top_n, analyze, background_tasks, response, rerun)