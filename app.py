import base64
import hashlib
import html as html_lib
import io
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import requests
except Exception:  # pragma: no cover - fallback for environments without requests
    requests = None

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except Exception:  # pragma: no cover - reportlab is optional at runtime
    REPORTLAB_AVAILABLE = False


st.set_page_config(
    page_title="Indian Election Analyzer & Predictor",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)


HISTORICAL_LS_URL = "https://raw.githubusercontent.com/Rahul28428/Indian-Elections-Data-Visualization/main/Loksabha_1962-2019%20.csv"
LS_2024_URL = "https://raw.githubusercontent.com/chandanneralgi/Indian-Loksabha-Election-2024-Result-Data-Analysis/main/data/election_results_2024.csv"
STATE_META_URL = "https://raw.githubusercontent.com/thecont1/india-votes-data/main/states.csv"
STATE_EXAMPLE_URLS = {
    "HR": "https://raw.githubusercontent.com/thecont1/india-votes-data/main/results/2024Assembly-HR.csv",
    "JH": "https://raw.githubusercontent.com/thecont1/india-votes-data/main/results/2024Assembly-JH.csv",
    "JK": "https://raw.githubusercontent.com/thecont1/india-votes-data/main/results/2024Assembly-JK.csv",
    "MH": "https://raw.githubusercontent.com/thecont1/india-votes-data/main/results/2024Assembly-MH.csv",
    "BR": "https://raw.githubusercontent.com/thecont1/india-votes-data/main/results/2025Assembly-BR.csv",
    "DL": "https://raw.githubusercontent.com/thecont1/india-votes-data/main/results/2025Assembly-DL.csv",
}

MANUAL_STATE_CATALOG = [
    {"state_code": "AN", "state_name": "Andaman and Nicobar Islands", "state_status": "UT"},
    {"state_code": "AP", "state_name": "Andhra Pradesh", "state_status": "State"},
    {"state_code": "AR", "state_name": "Arunachal Pradesh", "state_status": "State"},
    {"state_code": "AS", "state_name": "Assam", "state_status": "State"},
    {"state_code": "BR", "state_name": "Bihar", "state_status": "State"},
    {"state_code": "CH", "state_name": "Chandigarh", "state_status": "UT"},
    {"state_code": "CG", "state_name": "Chhattisgarh", "state_status": "State"},
    {"state_code": "DN", "state_name": "Dadra and Nagar Haveli and Daman and Diu", "state_status": "UT"},
    {"state_code": "DL", "state_name": "NCT of Delhi", "state_status": "UT"},
    {"state_code": "GA", "state_name": "Goa", "state_status": "State"},
    {"state_code": "GJ", "state_name": "Gujarat", "state_status": "State"},
    {"state_code": "HR", "state_name": "Haryana", "state_status": "State"},
    {"state_code": "HP", "state_name": "Himachal Pradesh", "state_status": "State"},
    {"state_code": "JK", "state_name": "Jammu & Kashmir", "state_status": "UT"},
    {"state_code": "JH", "state_name": "Jharkhand", "state_status": "State"},
    {"state_code": "KA", "state_name": "Karnataka", "state_status": "State"},
    {"state_code": "KL", "state_name": "Kerala", "state_status": "State"},
    {"state_code": "LD", "state_name": "Lakshadweep", "state_status": "UT"},
    {"state_code": "LA", "state_name": "Ladakh", "state_status": "UT"},
    {"state_code": "MP", "state_name": "Madhya Pradesh", "state_status": "State"},
    {"state_code": "MH", "state_name": "Maharashtra", "state_status": "State"},
    {"state_code": "MN", "state_name": "Manipur", "state_status": "State"},
    {"state_code": "ML", "state_name": "Meghalaya", "state_status": "State"},
    {"state_code": "MZ", "state_name": "Mizoram", "state_status": "State"},
    {"state_code": "NL", "state_name": "Nagaland", "state_status": "State"},
    {"state_code": "OD", "state_name": "Odisha", "state_status": "State"},
    {"state_code": "PB", "state_name": "Punjab", "state_status": "State"},
    {"state_code": "PY", "state_name": "Puducherry", "state_status": "UT"},
    {"state_code": "RJ", "state_name": "Rajasthan", "state_status": "State"},
    {"state_code": "SK", "state_name": "Sikkim", "state_status": "State"},
    {"state_code": "TN", "state_name": "Tamil Nadu", "state_status": "State"},
    {"state_code": "TG", "state_name": "Telangana", "state_status": "State"},
    {"state_code": "TR", "state_name": "Tripura", "state_status": "State"},
    {"state_code": "UP", "state_name": "Uttar Pradesh", "state_status": "State"},
    {"state_code": "UK", "state_name": "Uttarakhand", "state_status": "State"},
    {"state_code": "WB", "state_name": "West Bengal", "state_status": "State"},
]

NO_ASSEMBLY_UTS = {"AN", "CH", "DN", "LA", "LD"}
BUNDLED_STATE_CODES = frozenset(STATE_EXAMPLE_URLS)

PARTY_COLOR_MAP = {
    "BJP": "#FF9933",
    "INC": "#138808",
    "AAP": "#19A0D8",
    "SP": "#8B0000",
    "DMK": "#FF4500",
    "TMC": "#00A86B",
    "BSP": "#1F77B4",
    "SHS": "#6A5ACD",
    "NCP": "#6B7280",
    "CPI(M)": "#DC143C",
    "CPI": "#B22222",
    "JD(U)": "#2E8B57",
    "JD(S)": "#556B2F",
    "TDP": "#F4B400",
    "YSRCP": "#0EA5E9",
    "BJD": "#0F766E",
    "Independent": "#64748B",
}

PRIORITY_PARTIES = [
    "BJP",
    "INC",
    "AAP",
    "SP",
    "DMK",
    "TMC",
    "BSP",
    "SHS",
    "NCP",
    "CPI(M)",
    "CPI",
    "JD(U)",
    "JD(S)",
    "TDP",
    "YSRCP",
    "BJD",
]

ANALYSIS_CORE_COLUMNS = [
    "election_type",
    "source_name",
    "source_kind",
    "year",
    "state",
    "state_code",
    "state_key",
    "constituency",
    "constituency_key",
    "constituency_no",
    "candidate",
    "party",
    "party_key",
    "winner_votes",
    "total_votes",
    "electors",
    "turnout",
    "vote_share",
    "margin",
    "margin_pct",
    "runner_up_candidate",
    "runner_up_party",
    "seat",
]


# -----------------------------
# General utilities
# -----------------------------

def clean_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def smart_title(value: object) -> str:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    parts: List[str] = []
    for token in re.split(r"(\W+)", text):
        if not token:
            continue
        if token.isalpha() and token.isupper() and len(token) <= 4:
            parts.append(token.upper())
        elif token.isalpha():
            parts.append(token.capitalize())
        else:
            parts.append(token)
    return "".join(parts)


def normalize_party_name(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    low = text.lower()
    rules = [
        (r"\bbharatiya janata party\b|\bbjp\b", "BJP"),
        (r"\bindian national congress\b|\binc\b|\bcongress\b", "INC"),
        (r"\baam aadmi party\b", "AAP"),
        (r"\bsamajwadi party\b|\bsp\b", "SP"),
        (r"\bdravida munnetra kazhagam\b|\bdmk\b", "DMK"),
        (r"\ball india trinamool congress\b|\btrinamul\b|\btmc\b", "TMC"),
        (r"\bshiv sena\b", "SHS"),
        (r"\bnationalist congress party\b", "NCP"),
        (r"\bbahujan samaj party\b|\bbsp\b", "BSP"),
        (r"\bcommunist party of india \(marxist\)\b|\bcpi\(m\)\b|\bcpim\b", "CPI(M)"),
        (r"\bcommunist party of india\b|\bcpi\b", "CPI"),
        (r"\bjanata dal \(united\)\b|\bjd\(u\)\b|\bjanata dal united\b", "JD(U)"),
        (r"\bjanata dal \(secular\)\b|\bjd\(s\)\b|\bjanata dal secular\b", "JD(S)"),
        (r"\btelugu desam party\b|\btdp\b", "TDP"),
        (r"\byuvajana sramika rythu congress party\b|\bysrcp\b", "YSRCP"),
        (r"\bbiju janata dal\b|\bbjd\b", "BJD"),
        (r"\brashtriya janata dal\b|\brjd\b", "RJD"),
        (r"\bshiv sena \(uddhav balasaheb thackeray\)\b", "SHS (UBT)"),
        (r"\bindependent\b", "Independent"),
        (r"\baimim\b|\ball india majlis-e-ittehadul muslimeen\b", "AIMIM"),
        (r"\bshiromani akali dal\b", "SAD"),
        (r"\bnational people's party\b|\bnpp\b", "NPP"),
    ]
    for pattern, mapped in rules:
        if re.search(pattern, low):
            return mapped
    return text


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace({"": np.nan, "nan": np.nan}),
        errors="coerce",
    )


def parse_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )


def make_color_map(values: List[str]) -> Dict[str, str]:
    colors = []
    palette = [
        "#FF9933",
        "#138808",
        "#19A0D8",
        "#8B0000",
        "#FF4500",
        "#00A86B",
        "#1F77B4",
        "#6A5ACD",
        "#DC143C",
        "#0F766E",
        "#F4B400",
        "#6366F1",
        "#EF4444",
        "#10B981",
        "#8B5CF6",
        "#94A3B8",
    ]
    for i, value in enumerate(values):
        colors.append((value, palette[i % len(palette)]))
    return dict(colors)


def theme_styles(dark_mode: bool) -> str:
    if dark_mode:
        bg = "#0B1220"
        surface = "#121B2E"
        surface_alt = "#17233A"
        text = "#E5E7EB"
        muted = "#94A3B8"
        border = "rgba(255,255,255,0.10)"
        shadow = "0 18px 38px rgba(0,0,0,0.35)"
        hero_bg = """
            radial-gradient(circle at top left, rgba(255,153,51,0.18), transparent 26%),
            radial-gradient(circle at top right, rgba(16,185,129,0.14), transparent 24%),
            linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.94), rgba(6,78,59,0.88))
        """
        hero_title = "#F8FAFC"
        hero_copy = "#E2E8F0"
        hero_meta = "#CBD5E1"
        badge_text = "#FFF7ED"
        badge_bg = "rgba(251,146,60,0.22)"
        badge_border = "rgba(255,255,255,0.12)"
    else:
        bg = "#FFF9F1"
        surface = "#FFFFFF"
        surface_alt = "#F7FAFC"
        text = "#0F172A"
        muted = "#475569"
        border = "rgba(15,23,42,0.10)"
        shadow = "0 18px 38px rgba(15,23,42,0.10)"
        hero_bg = "linear-gradient(135deg, rgba(255,153,51,0.10), rgba(255,255,255,0.95), rgba(19,136,8,0.10))"
        hero_title = "#0F172A"
        hero_copy = "#334155"
        hero_meta = "#475569"
        badge_text = "#7C2D12"
        badge_bg = "rgba(255,153,51,0.20)"
        badge_border = "rgba(15,23,42,0.10)"

    return f"""
    <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(255,153,51,0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(19,136,8,0.16), transparent 28%),
                linear-gradient(180deg, {bg} 0%, {bg} 100%);
            color: {text};
        }}
        .hero {{
            border: 1px solid {border};
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            background: {hero_bg};
            box-shadow: {shadow};
            margin-bottom: 1rem;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 2.05rem;
            line-height: 1.1;
            color: {hero_title};
        }}
        .hero p {{
            margin: 0.35rem 0 0 0;
            color: {hero_copy};
            font-size: 0.98rem;
            line-height: 1.55;
        }}
        .hero strong {{
            color: {hero_title};
        }}
        .hero p:last-of-type {{
            color: {hero_meta};
        }}
        .badge {{
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            margin-bottom: 0.6rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: {badge_text};
            background: {badge_bg};
            border: 1px solid {badge_border};
        }}
        .section-title {{
            font-size: 1.15rem;
            font-weight: 800;
            margin: 0.4rem 0 0.8rem;
            color: {text};
        }}
        .mini-card {{
            border: 1px solid {border};
            border-radius: 18px;
            background: {surface};
            box-shadow: {shadow};
            padding: 0.8rem 0.95rem;
            margin-bottom: 0.7rem;
        }}
        .mini-card h3 {{
            margin: 0;
            font-size: 0.9rem;
            color: {muted};
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .mini-card p {{
            margin: 0.35rem 0 0;
            font-size: 1.25rem;
            font-weight: 800;
            color: {text};
        }}
        .report-box {{
            border: 1px solid {border};
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: {surface};
            box-shadow: {shadow};
            margin-bottom: 1rem;
        }}
        .report-box h2, .report-box h3 {{
            margin-top: 0;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        .report-table th, .report-table td {{
            border: 1px solid {border};
            padding: 0.45rem 0.55rem;
            text-align: left;
        }}
        .report-table th {{
            background: {surface_alt};
        }}
        footer {{
            margin-top: 2rem;
            padding: 1rem 0 0.5rem;
            border-top: 1px solid {border};
            color: {muted};
            text-align: center;
            font-size: 0.9rem;
        }}
    </style>
    """


def apply_theme(template_dark: bool) -> str:
    return "plotly_dark" if template_dark else "plotly_white"


def lookup_state_catalog() -> pd.DataFrame:
    try:
        catalog = load_csv_from_url(STATE_META_URL, cache_bust=0)
        catalog = catalog.rename(
            columns={
                "state_code": "state_code",
                "state_name": "state_name",
                "state_status": "state_status",
                "assembly_seats": "assembly_seats",
                "parliamentary_seats": "parliamentary_seats",
            }
        )
        catalog["state_name"] = catalog["state_name"].astype(str).str.strip()
        catalog["state_code"] = catalog["state_code"].astype(str).str.strip().str.upper()
        catalog["state_status"] = catalog["state_status"].astype(str).str.strip()
        if "assembly_seats" in catalog.columns:
            catalog["assembly_seats"] = coerce_numeric(catalog["assembly_seats"])
        if "parliamentary_seats" in catalog.columns:
            catalog["parliamentary_seats"] = coerce_numeric(catalog["parliamentary_seats"])
        if "Lakshadweep" not in catalog["state_name"].tolist():
            catalog = pd.concat(
                [
                    catalog,
                    pd.DataFrame(
                        [
                            {
                                "state_code": "LD",
                                "state_name": "Lakshadweep",
                                "state_status": "UT",
                                "assembly_seats": 0,
                                "parliamentary_seats": 1,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        return catalog.drop_duplicates(subset=["state_code"]).sort_values(["state_status", "state_name"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(MANUAL_STATE_CATALOG)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_csv_from_url(url: str, cache_bust: int = 0) -> pd.DataFrame:
    if not url:
        raise ValueError("Empty URL")
    text = None
    last_error = None
    if requests is not None:
        try:
            response = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding or "utf-8"
            text = response.text
        except Exception as exc:
            last_error = exc
    if text is None:
        try:
            from urllib.request import Request, urlopen

            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=45) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            if last_error is not None:
                raise RuntimeError(f"Unable to fetch CSV from {url}: {last_error}") from exc
            raise RuntimeError(f"Unable to fetch CSV from {url}: {exc}") from exc
    return parse_csv_text(text)


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(data: bytes, file_name: str, cache_bust: int = 0) -> pd.DataFrame:
    return parse_csv_bytes(data)


def parse_csv_text(text: str) -> pd.DataFrame:
    for kwargs in ({}, {"engine": "python", "sep": None}):
        try:
            return pd.read_csv(io.StringIO(text), **kwargs)
        except Exception:
            continue
    raise ValueError("The text could not be parsed as CSV.")


def parse_csv_bytes(data: bytes) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            text = data.decode(encoding)
            return parse_csv_text(text)
        except Exception:
            continue
    raise ValueError("Uploaded file could not be parsed as CSV.")


def ensure_analysis_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ANALYSIS_CORE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df

    merged = {}
    ordered_cols = []
    for col in df.columns:
        if col in merged:
            continue
        same_name = df.loc[:, df.columns == col]
        merged[col] = same_name.bfill(axis=1).iloc[:, 0]
        ordered_cols.append(col)
    return pd.DataFrame(merged, index=df.index)[ordered_cols]


def apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    exact = {}
    cleaned = {}
    for col in df.columns:
        label = str(col).strip()
        exact.setdefault(label.lower(), col)
        cleaned.setdefault(clean_key(label), col)

    alias_map = {
        "year": ["year", "electionyear", "election_year", "pollyear"],
        "election_type": ["electiontype", "election_type", "type"],
        "state": ["state", "state_name", "electionstate", "election_state", "pcstate"],
        "state_code": ["statecode", "state_code"],
        "constituency": ["constituency", "pcname", "pc_name", "constituencyname", "leadingconstituency"],
        "constituency_no": ["constituencyno", "constituency_no", "constno", "no", "constituencynumber", "constno"],
        "candidate": ["candidate", "candidate_name", "leadingcandidate", "winningcandidate"],
        "party": ["party", "leadingparty", "winningparty", "party_name"],
        "votes": ["votes", "winnervotes", "vote_count", "totalvotes"],
        "electors": ["electors", "electorate", "registered_voters"],
        "turnout": ["turnout", "turnoutpct", "turnout_percent", "turnout%"],
        "margin": ["margin", "winningmargin", "vote_margin"],
        "margin_pct": ["marginpct", "marginpercent", "margin%", "margin_percent"],
        "status": ["status"],
        "serial_no": ["serialno", "serial_no", "sno", "srno"],
        "runner_up_candidate": ["trailingcandidate", "runnerupcandidate"],
        "runner_up_party": ["trailingparty", "runnerupparty"],
        "evm_votes": ["evmvotes", "evm_votes"],
        "postal_votes": ["postalvotes", "postal_votes"],
    }
    rename = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            exact_match = exact.get(str(alias).strip().lower())
            if exact_match is not None:
                rename[exact_match] = canonical
                break
            cleaned_match = cleaned.get(clean_key(alias))
            if cleaned_match is not None:
                rename[cleaned_match] = canonical
                break
    renamed = df.rename(columns=rename).copy()
    return collapse_duplicate_columns(renamed)


def infer_schema(df: pd.DataFrame) -> str:
    cols = {clean_key(c) for c in df.columns}
    if {"leadingcandidate", "leadingparty"}.issubset(cols):
        return "loksabha_2024"
    if {"electionyear", "electionstate", "candidate", "party", "evmvotes", "postalvotes"}.issubset(cols):
        return "assembly_candidates"
    if {"pcname", "candidate", "party", "votes", "year"}.issubset(cols) or {"pcname", "candidate_name", "party", "votes", "year"}.issubset(cols):
        return "loksabha_historical"
    if {"candidate", "party", "votes"}.issubset(cols) and ("constituency" in cols or "pcname" in cols):
        return "winner_level"
    return "generic"


def enrich_state_and_keys(df: pd.DataFrame, state_lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "state" in out.columns:
        out["state"] = out["state"].astype(str).str.strip()
        out.loc[out["state"].isin(["", "nan", "None"]), "state"] = np.nan
    if "state_code" in out.columns:
        out["state_code"] = out["state_code"].astype(str).str.strip().str.upper()
        out.loc[out["state_code"].isin(["", "NAN", "NONE"]), "state_code"] = np.nan

    if "state" not in out.columns:
        out["state"] = np.nan
    if "state_code" not in out.columns:
        out["state_code"] = np.nan

    code_to_name = dict(zip(state_lookup["state_code"].astype(str), state_lookup["state_name"].astype(str)))
    name_to_code = dict(zip(state_lookup["state_name"].astype(str), state_lookup["state_code"].astype(str)))

    # If the source stored a 2-letter state code in the state column, expand it to the full state name.
    if out["state"].notna().any():
        state_upper = out["state"].astype(str).str.strip().str.upper()
        out.loc[state_upper.isin(code_to_name.keys()), "state"] = state_upper.map(code_to_name)

    out["state"] = out["state"].fillna(out["state_code"].map(code_to_name))
    out["state_code"] = out["state_code"].fillna(out["state"].map(name_to_code))
    out["state"] = out["state"].map(smart_title)

    if "constituency" in out.columns:
        out["constituency"] = out["constituency"].map(smart_title)
    else:
        out["constituency"] = np.nan

    if "candidate" in out.columns:
        out["candidate"] = out["candidate"].map(smart_title)
    else:
        out["candidate"] = np.nan

    if "runner_up_candidate" in out.columns:
        out["runner_up_candidate"] = out["runner_up_candidate"].map(smart_title)
    else:
        out["runner_up_candidate"] = np.nan

    if "party" in out.columns:
        out["party"] = out["party"].map(normalize_party_name)
    else:
        out["party"] = np.nan

    if "runner_up_party" in out.columns:
        out["runner_up_party"] = out["runner_up_party"].map(normalize_party_name)
    else:
        out["runner_up_party"] = np.nan

    out["state_key"] = out["state"].astype(str).str.lower().str.strip()
    out["constituency_key"] = out["constituency"].astype(str).str.lower().str.strip()
    out["party_key"] = out["party"].astype(str).str.lower().str.strip()
    return out


def normalize_loksabha_historical(df: pd.DataFrame, state_lookup: pd.DataFrame, source_name: str) -> pd.DataFrame:
    data = apply_column_aliases(df)
    expected = ["year", "state", "constituency", "constituency_no", "candidate", "party", "electors", "votes", "turnout", "margin", "margin_pct"]
    for col in expected:
        if col not in data.columns:
            data[col] = np.nan
    data["year"] = coerce_numeric(data["year"]).astype("Int64")
    data["constituency_no"] = coerce_numeric(data["constituency_no"]).astype("Int64")
    data["electors"] = coerce_numeric(data["electors"])
    data["winner_votes"] = coerce_numeric(data["votes"])
    data["turnout"] = parse_percent(data["turnout"])
    data["margin"] = coerce_numeric(data["margin"])
    data["margin_pct"] = parse_percent(data["margin_pct"])
    data["total_votes"] = np.where(
        data["electors"].notna() & data["turnout"].notna(),
        data["electors"] * data["turnout"] / 100.0,
        np.nan,
    )
    data["vote_share"] = np.where(
        data["winner_votes"].notna() & data["total_votes"].notna() & (data["total_votes"] > 0),
        data["winner_votes"] / data["total_votes"] * 100.0,
        np.nan,
    )
    data["seat"] = 1
    data["election_type"] = "Lok Sabha"
    data["source_name"] = source_name
    data["source_kind"] = "loksabha_historical"
    data = enrich_state_and_keys(data, state_lookup)
    return data[
        [
            "election_type",
            "source_name",
            "source_kind",
            "year",
            "state",
            "state_code",
            "state_key",
            "constituency",
            "constituency_key",
            "constituency_no",
            "candidate",
            "party",
            "party_key",
            "winner_votes",
            "total_votes",
            "electors",
            "turnout",
            "vote_share",
            "margin",
            "margin_pct",
            "runner_up_candidate",
            "runner_up_party",
            "seat",
        ]
    ]


def normalize_loksabha_2024(df: pd.DataFrame, state_lookup: pd.DataFrame, constituency_state_map: Optional[Dict[str, str]], source_name: str) -> pd.DataFrame:
    data = apply_column_aliases(df)
    for col in ["year", "constituency", "constituency_no", "candidate", "party", "runner_up_candidate", "runner_up_party", "margin", "status"]:
        if col not in data.columns:
            data[col] = np.nan
    data["year"] = pd.to_numeric(data["year"], errors="coerce").fillna(2024).astype("Int64")
    data["constituency_no"] = coerce_numeric(data["constituency_no"]).astype("Int64")
    data["margin"] = coerce_numeric(data["margin"])
    data["state"] = np.nan
    if constituency_state_map:
        data["state"] = data["constituency"].astype(str).str.lower().str.strip().map(constituency_state_map)
    data["state_code"] = data["state"].map(dict(zip(state_lookup["state_name"], state_lookup["state_code"])))
    data["winner_votes"] = np.nan
    data["total_votes"] = np.nan
    data["electors"] = np.nan
    data["turnout"] = np.nan
    data["vote_share"] = np.nan
    data["margin_pct"] = np.nan
    data["seat"] = 1
    data["election_type"] = "Lok Sabha"
    data["source_name"] = source_name
    data["source_kind"] = "loksabha_2024"
    data = enrich_state_and_keys(data, state_lookup)
    return data[
        [
            "election_type",
            "source_name",
            "source_kind",
            "year",
            "state",
            "state_code",
            "state_key",
            "constituency",
            "constituency_key",
            "constituency_no",
            "candidate",
            "party",
            "party_key",
            "winner_votes",
            "total_votes",
            "electors",
            "turnout",
            "vote_share",
            "margin",
            "margin_pct",
            "runner_up_candidate",
            "runner_up_party",
            "seat",
        ]
    ]


def normalize_assembly_candidates(df: pd.DataFrame, state_lookup: pd.DataFrame, source_name: str) -> pd.DataFrame:
    data = apply_column_aliases(df)
    for col in ["year", "election_year", "election_state", "constituency", "constituency_no", "candidate", "party", "evm_votes", "postal_votes", "serial_no"]:
        if col not in data.columns:
            data[col] = np.nan
    if "year" not in data.columns or data["year"].isna().all():
        data["year"] = data["election_year"]
    if "state_code" not in data.columns or data["state_code"].isna().all():
        data["state_code"] = data["election_state"]
    data["year"] = coerce_numeric(data["year"]).astype("Int64")
    data["constituency_no"] = coerce_numeric(data["constituency_no"]).astype("Int64")
    data["evm_votes"] = coerce_numeric(data["evm_votes"])
    data["postal_votes"] = coerce_numeric(data["postal_votes"])
    data["candidate_votes"] = data["evm_votes"].fillna(0) + data["postal_votes"].fillna(0)
    data["party"] = data["party"].map(normalize_party_name)
    data["candidate"] = data["candidate"].map(smart_title)
    if "state" not in data.columns:
        data["state"] = np.nan
    if "state_code" not in data.columns:
        data["state_code"] = np.nan
    state_codes = set(state_lookup["state_code"].astype(str).str.upper().tolist())
    name_to_code = dict(zip(state_lookup["state_name"].astype(str), state_lookup["state_code"].astype(str)))
    data["state_code"] = data["state_code"].astype(str).str.strip().str.upper()
    data.loc[data["state_code"].isin(["", "NAN", "NONE"]), "state_code"] = np.nan
    if data["state_code"].isna().all():
        raw_state = data["state"].astype(str).str.strip()
        state_upper = raw_state.str.upper()
        data["state_code"] = np.where(state_upper.isin(state_codes), state_upper, raw_state.map(name_to_code))
    if data["state"].isna().all() and "election_state" in data.columns:
        mapped_state = data["election_state"].astype(str).str.strip().str.upper().map(dict(zip(state_lookup["state_code"], state_lookup["state_name"])))
        data["state"] = mapped_state
        data["state_code"] = data["state_code"].fillna(data["election_state"].astype(str).str.strip().str.upper())
    data["state"] = data["state"].fillna(data["state_code"].map(dict(zip(state_lookup["state_code"], state_lookup["state_name"]))))
    data["constituency"] = data["constituency"].map(smart_title)
    winners = []
    group_cols = ["year", "state_code", "constituency", "constituency_no"]
    for keys, group in data.groupby(group_cols, dropna=False):
        group = group.sort_values("candidate_votes", ascending=False).reset_index(drop=True)
        winner = group.iloc[0].copy()
        runner_up = group.iloc[1].copy() if len(group) > 1 else None
        winner_votes = float(winner["candidate_votes"])
        runner_up_votes = float(runner_up["candidate_votes"]) if runner_up is not None else np.nan
        total_votes = float(group["candidate_votes"].sum())
        winners.append(
            {
                "election_type": "Assembly",
                "source_name": source_name,
                "source_kind": "assembly_candidates",
                "year": int(winner["year"]) if pd.notna(winner["year"]) else np.nan,
                "state": winner["state"],
                "state_code": winner["state_code"],
                "state_key": str(winner["state"]).lower().strip() if pd.notna(winner["state"]) else np.nan,
                "constituency": winner["constituency"],
                "constituency_key": str(winner["constituency"]).lower().strip() if pd.notna(winner["constituency"]) else np.nan,
                "constituency_no": int(winner["constituency_no"]) if pd.notna(winner["constituency_no"]) else np.nan,
                "candidate": winner["candidate"],
                "party": winner["party"],
                "party_key": str(winner["party"]).lower().strip() if pd.notna(winner["party"]) else np.nan,
                "winner_votes": winner_votes,
                "total_votes": total_votes,
                "electors": np.nan,
                "turnout": np.nan,
                "vote_share": winner_votes / total_votes * 100.0 if total_votes > 0 else np.nan,
                "margin": winner_votes - runner_up_votes if pd.notna(runner_up_votes) else np.nan,
                "margin_pct": np.nan,
                "runner_up_candidate": runner_up["candidate"] if runner_up is not None else np.nan,
                "runner_up_party": runner_up["party"] if runner_up is not None else np.nan,
                "seat": 1,
            }
        )
    result = pd.DataFrame(winners)
    result = enrich_state_and_keys(result, state_lookup)
    return result[
        [
            "election_type",
            "source_name",
            "source_kind",
            "year",
            "state",
            "state_code",
            "state_key",
            "constituency",
            "constituency_key",
            "constituency_no",
            "candidate",
            "party",
            "party_key",
            "winner_votes",
            "total_votes",
            "electors",
            "turnout",
            "vote_share",
            "margin",
            "margin_pct",
            "runner_up_candidate",
            "runner_up_party",
            "seat",
        ]
    ]


def normalize_winner_level(df: pd.DataFrame, state_lookup: pd.DataFrame, source_name: str) -> pd.DataFrame:
    data = apply_column_aliases(df)
    for col in ["year", "state", "constituency", "constituency_no", "candidate", "party", "votes", "margin", "margin_pct"]:
        if col not in data.columns:
            data[col] = np.nan
    data["year"] = coerce_numeric(data["year"]).astype("Int64")
    data["constituency_no"] = coerce_numeric(data["constituency_no"]).astype("Int64")
    data["winner_votes"] = coerce_numeric(data["votes"])
    data["margin"] = coerce_numeric(data["margin"])
    data["margin_pct"] = parse_percent(data["margin_pct"])
    if "state" not in data.columns:
        data["state"] = np.nan
    if "state_code" not in data.columns:
        data["state_code"] = np.nan
    data["state_code"] = data["state_code"].astype(str).str.strip().str.upper()
    data.loc[data["state_code"].isin(["", "NAN", "NONE"]), "state_code"] = np.nan
    data["state"] = data["state"].fillna(data["state_code"].map(dict(zip(state_lookup["state_code"], state_lookup["state_name"]))))
    data["candidate"] = data["candidate"].map(smart_title)
    data["party"] = data["party"].map(normalize_party_name)
    data["constituency"] = data["constituency"].map(smart_title)
    data["turnout"] = np.nan
    data["electors"] = np.nan
    data["vote_share"] = np.nan
    data["total_votes"] = np.nan
    data["runner_up_candidate"] = np.nan
    data["runner_up_party"] = np.nan
    data["seat"] = 1
    data["election_type"] = "Election"
    data["source_name"] = source_name
    data["source_kind"] = "winner_level"
    data = enrich_state_and_keys(data, state_lookup)
    return data[
        [
            "election_type",
            "source_name",
            "source_kind",
            "year",
            "state",
            "state_code",
            "state_key",
            "constituency",
            "constituency_key",
            "constituency_no",
            "candidate",
            "party",
            "party_key",
            "winner_votes",
            "total_votes",
            "electors",
            "turnout",
            "vote_share",
            "margin",
            "margin_pct",
            "runner_up_candidate",
            "runner_up_party",
            "seat",
        ]
    ]


def normalize_generic(df: pd.DataFrame, state_lookup: pd.DataFrame, source_name: str) -> pd.DataFrame:
    schema = infer_schema(df)
    if schema == "loksabha_historical":
        return normalize_loksabha_historical(df, state_lookup, source_name)
    if schema == "loksabha_2024":
        return normalize_loksabha_2024(df, state_lookup, None, source_name)
    if schema == "assembly_candidates":
        return normalize_assembly_candidates(df, state_lookup, source_name)
    if schema == "winner_level":
        return normalize_winner_level(df, state_lookup, source_name)

    data = apply_column_aliases(df)
    if {"candidate", "party", "votes"}.issubset(set(data.columns)) and ("constituency" in data.columns or "pc_name" in data.columns):
        return normalize_winner_level(data, state_lookup, source_name)
    if {"candidate", "party", "evm_votes", "postal_votes"}.issubset(set(data.columns)):
        return normalize_assembly_candidates(data, state_lookup, source_name)
    raise ValueError("Could not infer the CSV schema. Please upload a constituency-level election result CSV.")


def build_constituency_state_lookup(df: pd.DataFrame) -> Dict[str, str]:
    lookup = {}
    temp = df.dropna(subset=["constituency", "state"]).copy()
    if temp.empty:
        return lookup
    for key, group in temp.groupby("constituency_key"):
        states = group["state"].dropna().astype(str)
        if not states.empty:
            lookup[str(key)] = states.mode().iloc[0]
    return lookup


def patch_missing_states(df: pd.DataFrame, lookup: Dict[str, str], state_lookup: pd.DataFrame) -> pd.DataFrame:
    out = ensure_analysis_schema(df)
    mask = out["state"].isna() | out["state"].astype(str).str.lower().isin(["nan", "none", ""])
    out.loc[mask, "state"] = out.loc[mask, "constituency_key"].map(lookup)
    code_map = dict(zip(state_lookup["state_name"], state_lookup["state_code"]))
    out["state_code"] = out["state_code"].fillna(out["state"].map(code_map))
    out["state_key"] = out["state"].astype(str).str.lower().str.strip()
    return out


def sort_years(years: List[int]) -> List[int]:
    unique = []
    for yr in years:
        if pd.notna(yr):
            try:
                yr_int = int(yr)
                if yr_int not in unique:
                    unique.append(yr_int)
            except Exception:
                continue
    return sorted(unique)


def load_national_analysis_bundle(cache_bust: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    state_lookup = lookup_state_catalog()
    hist_raw = load_csv_from_url(HISTORICAL_LS_URL, cache_bust=cache_bust)
    hist = normalize_loksabha_historical(hist_raw, state_lookup, "Lok Sabha 1962-2019")
    ls2024_raw = load_csv_from_url(LS_2024_URL, cache_bust=cache_bust)
    constituency_state_lookup = build_constituency_state_lookup(hist)
    ls2024 = normalize_loksabha_2024(ls2024_raw, state_lookup, constituency_state_lookup, "Lok Sabha 2024")
    combined = pd.concat([hist, ls2024], ignore_index=True)
    combined = patch_missing_states(combined, constituency_state_lookup, state_lookup)
    years = sort_years(combined["year"].dropna().astype(int).tolist())
    return combined, hist, "Lok Sabha (National) - Historical 1962-2019 + 2024"


def load_state_analysis_bundle(
    state_code: str,
    custom_url: str,
    uploaded_bytes: Optional[bytes],
    uploaded_name: Optional[str],
    cache_bust: int,
) -> Tuple[pd.DataFrame, str]:
    state_lookup = lookup_state_catalog()
    source_label = ""
    selected_df = None
    if custom_url:
        selected_df = load_csv_from_url(custom_url.strip(), cache_bust=cache_bust)
        source_label = f"Custom URL: {custom_url.strip()}"
    else:
        bundled_url = STATE_EXAMPLE_URLS.get(state_code)
        if bundled_url:
            selected_df = load_csv_from_url(bundled_url, cache_bust=cache_bust)
            source_label = f"Bundled example: {state_code}"
    if selected_df is None and uploaded_bytes is not None:
        selected_df = load_csv_from_bytes(uploaded_bytes, uploaded_name or "uploaded.csv", cache_bust=cache_bust)
        source_label = f"Uploaded CSV: {uploaded_name or 'uploaded.csv'}"
    if selected_df is None:
        state_name = state_lookup.loc[state_lookup["state_code"] == state_code, "state_name"]
        state_label = state_name.iloc[0] if not state_name.empty else state_code
        if state_code in NO_ASSEMBLY_UTS:
            raise ValueError(f"{state_label} does not have a legislative assembly. Upload a CSV or provide a raw GitHub CSV URL.")
        raise ValueError(f"No bundled online dataset is configured for {state_label}. Upload a CSV or provide a raw GitHub CSV URL.")
    normalized = normalize_generic(selected_df, state_lookup, source_label)
    if normalized["state"].isna().all():
        state_name = state_lookup.loc[state_lookup["state_code"] == state_code, "state_name"]
        if not state_name.empty:
            normalized["state"] = state_name.iloc[0]
            normalized["state_code"] = state_code
            normalized["state_key"] = normalized["state"].astype(str).str.lower().str.strip()
    return normalized, source_label


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_national_prediction_corpus(cache_bust: int) -> pd.DataFrame:
    state_lookup = lookup_state_catalog()
    hist = normalize_loksabha_historical(load_csv_from_url(HISTORICAL_LS_URL, cache_bust=cache_bust), state_lookup, "Lok Sabha 1962-2019")
    constituency_state_lookup = build_constituency_state_lookup(hist)
    ls2024 = normalize_loksabha_2024(load_csv_from_url(LS_2024_URL, cache_bust=cache_bust), state_lookup, constituency_state_lookup, "Lok Sabha 2024")
    combined = pd.concat([hist, ls2024], ignore_index=True)
    combined = patch_missing_states(combined, constituency_state_lookup, state_lookup)
    return combined


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_state_prediction_corpus(cache_bust: int) -> pd.DataFrame:
    state_lookup = lookup_state_catalog()
    frames = []
    for code, url in STATE_EXAMPLE_URLS.items():
        try:
            raw = load_csv_from_url(url, cache_bust=cache_bust)
            frames.append(normalize_generic(raw, state_lookup, f"Bundled example {code}"))
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# -----------------------------
# Analytics
# -----------------------------

def filtered_years(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    if not years:
        return df.copy()
    return df[df["year"].isin(years)].copy()


def compute_party_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby("party", dropna=False)
    out = grp.agg(
        Seats=("seat", "sum"),
        Winner_Votes=("winner_votes", "sum"),
        Avg_Margin=("margin", "mean"),
        Avg_Turnout=("turnout", "mean"),
        Avg_Vote_Share=("vote_share", "mean"),
        First_Year=("year", "min"),
        Last_Year=("year", "max"),
        State_Count=("state", "nunique"),
    ).reset_index()
    total_seats = out["Seats"].sum()
    out["Seat_Share"] = np.where(total_seats > 0, out["Seats"] / total_seats * 100.0, np.nan)
    out = out.sort_values(["Seats", "Winner_Votes"], ascending=False).reset_index(drop=True)
    return out


def compute_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "state" not in df.columns or df["state"].isna().all():
        return pd.DataFrame()
    rows = []
    for state, group in df.groupby("state", dropna=False):
        if pd.isna(state):
            continue
        party_summary = group.groupby("party")["seat"].sum().sort_values(ascending=False)
        top_party = party_summary.index[0] if not party_summary.empty else None
        rows.append(
            {
                "State": state,
                "Seats": int(group["seat"].sum()),
                "Top Party": top_party,
                "Top Party Seats": int(party_summary.iloc[0]) if not party_summary.empty else np.nan,
                "Avg Margin": float(group["margin"].mean()) if group["margin"].notna().any() else np.nan,
                "Avg Turnout": float(group["turnout"].mean()) if group["turnout"].notna().any() else np.nan,
                "Constituencies": int(group["constituency"].nunique()),
            }
        )
    out = pd.DataFrame(rows).sort_values("Seats", ascending=False).reset_index(drop=True)
    return out


def compute_top_margins(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    cols = ["year", "state", "constituency", "party", "margin", "vote_share"]
    out = df[cols].dropna(subset=["margin"]).sort_values("margin", ascending=False).head(top_n).copy()
    return out


def trend_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    yearly = df.groupby("year").agg(
        Seats=("seat", "sum"),
        Avg_Turnout=("turnout", "mean"),
        Avg_Margin=("margin", "mean"),
        Avg_Vote_Share=("vote_share", "mean"),
    ).reset_index()
    return yearly.sort_values("year")


def generate_insights(df: pd.DataFrame, party_summary: pd.DataFrame, state_summary: pd.DataFrame) -> str:
    if df.empty:
        return "No data is available for the current selection."
    latest_year = int(df["year"].max())
    earliest_year = int(df["year"].min())
    top_party = party_summary.iloc[0]["party"] if not party_summary.empty else "N/A"
    top_party_seats = int(party_summary.iloc[0]["Seats"]) if not party_summary.empty else 0
    total_seats = int(df["seat"].sum())
    avg_margin = df["margin"].mean()
    avg_turnout = df["turnout"].mean()
    top_margin_row = df.sort_values("margin", ascending=False).head(1)
    top_margin_text = "N/A"
    if not top_margin_row.empty and pd.notna(top_margin_row.iloc[0]["margin"]):
        r = top_margin_row.iloc[0]
        top_margin_text = f"{r['constituency']} ({r['state']}) - {r['party']} won by {int(r['margin']):,}"

    bullet_lines = [
        f"- Total seats analyzed: **{total_seats:,}** across **{df['year'].nunique()}** election year(s).",
        f"- Leading party in the filtered data: **{top_party}** with **{top_party_seats:,}** seats.",
        f"- Average winning margin: **{avg_margin:,.1f}** votes." if pd.notna(avg_margin) else "- Average winning margin: **N/A**.",
        f"- Average turnout: **{avg_turnout:.1f}%**." if pd.notna(avg_turnout) else "- Average turnout: **N/A**.",
        f"- Biggest win margin: **{top_margin_text}**.",
    ]

    if df["year"].nunique() > 1:
        first = df[df["year"] == earliest_year].groupby("party")["seat"].sum()
        last = df[df["year"] == latest_year].groupby("party")["seat"].sum()
        swings = (last.subtract(first, fill_value=0)).sort_values(ascending=False)
        if not swings.empty:
            winner = swings.index[0]
            loser = swings.index[-1]
            bullet_lines.append(f"- Biggest seat gain: **{winner}** ({int(swings.iloc[0]):+d}).")
            bullet_lines.append(f"- Biggest seat loss: **{loser}** ({int(swings.iloc[-1]):+d}).")
        first_turnout = df[df["year"] == earliest_year]["turnout"].mean()
        last_turnout = df[df["year"] == latest_year]["turnout"].mean()
        if pd.notna(first_turnout) and pd.notna(last_turnout):
            bullet_lines.append(f"- Turnout changed from **{first_turnout:.1f}%** to **{last_turnout:.1f}%** between {earliest_year} and {latest_year}.")

    if not state_summary.empty:
        top_state = state_summary.iloc[0]
        bullet_lines.append(f"- Highest seat concentration in a state/UT: **{top_state['State']}** with **{int(top_state['Seats']):,}** seats.")

    return "\n".join(bullet_lines)


# -----------------------------
# Charts
# -----------------------------

def plot_party_seats(party_summary: pd.DataFrame, dark_mode: bool) -> go.Figure:
    if party_summary.empty:
        return go.Figure()
    top = party_summary.head(12).copy()
    fig = px.bar(
        top,
        x="party",
        y="Seats",
        color="party",
        color_discrete_map=PARTY_COLOR_MAP,
        text="Seats",
        title="Party-wise Seats Won",
    )
    fig.update_layout(template=apply_theme(dark_mode), xaxis_title="", yaxis_title="Seats", legend_title="", margin=dict(l=10, r=10, t=60, b=10))
    fig.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>Seats: %{y}<extra></extra>")
    return fig


def plot_party_treemap(party_summary: pd.DataFrame, dark_mode: bool) -> go.Figure:
    if party_summary.empty:
        return go.Figure()
    top = party_summary.head(15).copy()
    fig = px.treemap(top, path=[px.Constant("All Parties"), "party"], values="Seats", color="party", color_discrete_map=PARTY_COLOR_MAP)
    fig.update_layout(template=apply_theme(dark_mode), margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_share_pie(party_summary: pd.DataFrame, metric_label: str, dark_mode: bool) -> go.Figure:
    if party_summary.empty:
        return go.Figure()
    data = party_summary.copy()
    if metric_label == "Vote share" and data["Avg_Vote_Share"].notna().any():
        data["share_value"] = data["Avg_Vote_Share"].fillna(0)
    else:
        data["share_value"] = data["Seat_Share"].fillna(0)
        metric_label = "Seat share proxy"
    data = data[data["share_value"] > 0].sort_values("share_value", ascending=False).head(10)
    if data.empty:
        return go.Figure()
    fig = px.pie(
        data,
        names="party",
        values="share_value",
        color="party",
        color_discrete_map=PARTY_COLOR_MAP,
        title=f"{metric_label} by Party",
        hole=0.38,
    )
    fig.update_layout(template=apply_theme(dark_mode), margin=dict(l=10, r=10, t=60, b=10))
    fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="<b>%{label}</b><br>Share: %{value:.2f}<extra></extra>")
    return fig


def plot_trend_lines(df: pd.DataFrame, party_summary: pd.DataFrame, dark_mode: bool) -> go.Figure:
    if df.empty:
        return go.Figure()
    trend = trend_data(df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    top_parties = party_summary.head(5)["party"].tolist() if not party_summary.empty else []
    for party in top_parties:
        yearly = df[df["party"] == party].groupby("year")["seat"].sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=yearly["year"],
                y=yearly["seat"],
                mode="lines+markers",
                name=f"{party} seats",
            ),
            secondary_y=False,
        )
    if not trend.empty and trend["Avg_Turnout"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=trend["year"],
                y=trend["Avg_Turnout"],
                mode="lines+markers",
                name="Avg turnout %",
                line=dict(color="#0EA5E9", dash="dot", width=3),
            ),
            secondary_y=True,
        )
    fig.update_layout(
        template=apply_theme(dark_mode),
        title="Historical Trend: Seats and Turnout",
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title="",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Seats", secondary_y=False)
    fig.update_yaxes(title_text="Turnout %", secondary_y=True)
    return fig


def plot_top_margins(df: pd.DataFrame, dark_mode: bool) -> go.Figure:
    top = compute_top_margins(df, top_n=10)
    if top.empty:
        return go.Figure()
    label = top.apply(lambda r: f"{r['constituency']} ({r['year']})", axis=1)
    fig = px.bar(
        top.assign(Label=label),
        x="margin",
        y="Label",
        orientation="h",
        color="party",
        color_discrete_map=PARTY_COLOR_MAP,
        title="Top 10 Constituencies by Margin",
        hover_data={"state": True, "vote_share": ":.2f"},
    )
    fig.update_layout(template=apply_theme(dark_mode), margin=dict(l=10, r=10, t=60, b=10), yaxis_title="", xaxis_title="Winning margin")
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Margin: %{x:,}<extra></extra>")
    return fig


# -----------------------------
# Report export
# -----------------------------

def dataframe_html_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "<p>No data available.</p>"
    return df.head(max_rows).to_html(index=False, classes="report-table", border=0, escape=True)


def html_report(summary_text: str, party_summary: pd.DataFrame, state_summary: pd.DataFrame, top_margins: pd.DataFrame, filtered_df: pd.DataFrame, title: str) -> str:
    css = """
    <style>
        body { font-family: Arial, sans-serif; padding: 24px; color: #0f172a; }
        h1, h2, h3 { color: #0f172a; }
        .box { border: 1px solid #cbd5e1; border-radius: 14px; padding: 18px; margin-bottom: 18px; }
        .muted { color: #475569; }
        table { border-collapse: collapse; width: 100%; margin-top: 8px; }
        th, td { border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }
        th { background: #f1f5f9; }
    </style>
    """
    overview = f"""
    <div class="box">
        <h1>{html_lib.escape(title)}</h1>
        <p class="muted">Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}</p>
        <pre style="white-space: pre-wrap; font-family: inherit;">{html_lib.escape(summary_text)}</pre>
    </div>
    """
    party_block = f'<div class="box"><h2>Party Summary</h2>{dataframe_html_table(party_summary)}</div>'
    state_block = f'<div class="box"><h2>State Summary</h2>{dataframe_html_table(state_summary)}</div>' if not state_summary.empty else ""
    margin_block = f'<div class="box"><h2>Top Constituencies by Margin</h2>{dataframe_html_table(top_margins)}</div>'
    data_block = f'<div class="box"><h2>Data Preview</h2>{dataframe_html_table(filtered_df)}</div>'
    return f"<html><head><meta charset='utf-8'>{css}</head><body>{overview}{party_block}{state_block}{margin_block}{data_block}</body></html>"


def html_download_link(html_content: str, filename: str = "indian_election_report.html") -> str:
    encoded = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    return f'<a href="data:text/html;base64,{encoded}" download="{filename}">Download HTML report</a>'


def pdf_report_bytes(summary_text: str, party_summary: pd.DataFrame, state_summary: pd.DataFrame, top_margins: pd.DataFrame, title: str) -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9, leading=12))
    safe_summary = html_lib.escape(summary_text).replace("\n", "<br/>")
    story = [Paragraph(title, styles["CenterTitle"]), Spacer(1, 0.12 * inch), Paragraph(safe_summary, styles["BodySmall"]), Spacer(1, 0.14 * inch)]

    def table_data(df: pd.DataFrame, max_rows: int = 10) -> List[List[str]]:
        if df.empty:
            return [["No data available"]]
        limited = df.head(max_rows).copy()
        header = [str(c) for c in limited.columns]
        rows = [header]
        for _, row in limited.iterrows():
            rows.append([str(v) if pd.notna(v) else "" for v in row.tolist()])
        return rows

    def styled_table(df: pd.DataFrame, heading: str) -> Table:
        data = table_data(df)
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("LEADING", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(table)
        story.append(Spacer(1, 0.14 * inch))
        return table

    styled_table(party_summary, "Party Summary")
    if not state_summary.empty:
        styled_table(state_summary, "State Summary")
    styled_table(top_margins, "Top Constituencies by Margin")
    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# -----------------------------
# Prediction helpers
# -----------------------------

def build_party_year_corpus(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby(["year", "party"], dropna=False)
    party_year = grp.agg(
        seats=("seat", "sum"),
        avg_vote_share=("vote_share", "mean"),
        avg_margin=("margin", "mean"),
        avg_turnout=("turnout", "mean"),
    ).reset_index()
    party_year = party_year.sort_values(["party", "year"]).reset_index(drop=True)
    party_year["lag_share"] = party_year.groupby("party")["avg_vote_share"].shift(1)
    party_year["lag_seats"] = party_year.groupby("party")["seats"].shift(1)
    party_year["lag_margin"] = party_year.groupby("party")["avg_margin"].shift(1)
    party_year["lag_turnout"] = party_year.groupby("party")["avg_turnout"].shift(1)
    party_year["prev_year"] = party_year.groupby("party")["year"].shift(1)
    party_year["year_gap"] = party_year["year"] - party_year["prev_year"]
    return party_year


def fit_vote_share_models(corpus: pd.DataFrame):
    party_year = build_party_year_corpus(corpus)
    if party_year.empty:
        return {"available": False, "reason": "No party-year data found."}
    target_rows = party_year.dropna(subset=["lag_share", "year", "party"]).copy()
    if len(target_rows) < 8:
        return {"available": False, "reason": "Not enough historical party-year observations to train a stable vote-share model."}

    features = ["year", "party", "lag_share", "lag_seats", "lag_margin", "lag_turnout", "year_gap"]
    target = "avg_vote_share"
    data = target_rows.dropna(subset=[target]).copy()
    if data.empty:
        return {"available": False, "reason": "No valid vote-share target available."}
    X = data[features].copy()
    y = data[target].astype(float)

    numeric_features = ["year", "lag_share", "lag_seats", "lag_margin", "lag_turnout", "year_gap"]
    categorical_features = ["party"]
    lin_pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    rf_pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    if len(data) >= 12:
        test_size = 0.25
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    lin = Pipeline([("pre", lin_pre), ("model", LinearRegression())])
    rf = Pipeline([("pre", rf_pre), ("model", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=1))])
    lin.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lin_pred = lin.predict(X_test)
    rf_pred = rf.predict(X_test)
    try:
        lin_r2 = r2_score(y_test, lin_pred)
    except Exception:
        lin_r2 = float("nan")
    try:
        rf_r2 = r2_score(y_test, rf_pred)
    except Exception:
        rf_r2 = float("nan")
    try:
        lin_mae = mean_absolute_error(y_test, lin_pred)
    except Exception:
        lin_mae = float("nan")
    try:
        rf_mae = mean_absolute_error(y_test, rf_pred)
    except Exception:
        rf_mae = float("nan")

    use_rf = np.nan_to_num(rf_r2, nan=-999) >= np.nan_to_num(lin_r2, nan=-999)
    chosen = rf if use_rf else lin
    chosen_name = "RandomForestRegressor" if use_rf else "LinearRegression"

    feature_names = numeric_features + list(chosen.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(categorical_features))
    if use_rf:
        importances = chosen.named_steps["model"].feature_importances_
    else:
        coeffs = chosen.named_steps["model"].coef_
        importances = np.abs(coeffs)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    latest = data.sort_values("year").groupby("party", dropna=False).tail(1).copy()
    latest["year"] = latest["year"].astype(int)
    latest["future_year"] = latest["year"] + 1

    return {
        "available": True,
        "model": chosen,
        "model_name": chosen_name,
        "lin_r2": lin_r2,
        "rf_r2": rf_r2,
        "lin_mae": lin_mae,
        "rf_mae": rf_mae,
        "feature_importance": importance_df,
        "training_frame": data,
        "latest_party_frame": latest,
    }


def predict_vote_share(model_bundle: dict, future_year: int, poll_map: Dict[str, float]) -> pd.DataFrame:
    data = model_bundle["latest_party_frame"].copy()
    if data.empty:
        return pd.DataFrame()
    X_future = pd.DataFrame(
        {
            "year": future_year,
            "party": data["party"].values,
            "lag_share": data["avg_vote_share"].values,
            "lag_seats": data["seats"].values,
            "lag_margin": data["avg_margin"].values,
            "lag_turnout": data["avg_turnout"].values,
            "year_gap": future_year - data["year"].values,
        }
    )
    predicted = np.clip(model_bundle["model"].predict(X_future), 0, 100)
    result = pd.DataFrame({"party": data["party"].values, "predicted_vote_share": predicted})
    if poll_map:
        latest_baseline = model_bundle["training_frame"].groupby("party")["avg_vote_share"].last().to_dict()
        adjusted = []
        for _, row in result.iterrows():
            party = row["party"]
            baseline = max(float(latest_baseline.get(party, row["predicted_vote_share"])), 1e-6)
            poll = poll_map.get(party)
            if poll is not None:
                blended = 0.6 * row["predicted_vote_share"] + 0.4 * poll
                adjusted.append(blended)
            else:
                adjusted.append(row["predicted_vote_share"])
        result["predicted_vote_share"] = adjusted
    result = result.sort_values("predicted_vote_share", ascending=False).reset_index(drop=True)
    return result


def build_constituency_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    data = df.copy().sort_values(["state_key", "constituency_key", "year"])
    grp = data.groupby(["state_key", "constituency_key"], dropna=False)
    data["prev_party"] = grp["party"].shift(1)
    data["prev_margin"] = grp["margin"].shift(1)
    data["prev_vote_share"] = grp["vote_share"].shift(1)
    data["prev_turnout"] = grp["turnout"].shift(1)
    data["prev_year"] = grp["year"].shift(1)
    data["year_gap"] = data["year"] - data["prev_year"]
    data["incumbent"] = (data["party"] == data["prev_party"]).astype(float)
    return data


def fit_winner_classifier(corpus: pd.DataFrame):
    history = build_constituency_history(corpus)
    data = history.dropna(subset=["prev_party"]).copy()
    if data.empty or data["party"].nunique() < 2:
        return {"available": False, "reason": "Not enough constituency history to train a winning-party classifier."}

    features = ["year", "state", "constituency_no", "prev_party", "prev_margin", "prev_vote_share", "prev_turnout", "year_gap", "incumbent"]
    data["constituency_no"] = pd.to_numeric(data["constituency_no"], errors="coerce")
    data["year_gap"] = pd.to_numeric(data["year_gap"], errors="coerce")
    X = data[features].copy()
    y = data["party"].astype(str)
    numeric_features = ["year", "constituency_no", "prev_margin", "prev_vote_share", "prev_turnout", "year_gap", "incumbent"]
    categorical_features = ["state", "prev_party"]
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    if len(data) >= 30 and y.nunique() > 1:
        stratify = y if y.value_counts().min() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
        except Exception:
            X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    if y_train.nunique() < 2:
        model = Pipeline([("pre", pre), ("model", DummyClassifier(strategy="most_frequent"))])
    else:
        model = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    try:
        acc = accuracy_score(y_test, pred)
    except Exception:
        acc = float("nan")

    feature_names = numeric_features + list(model.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(categorical_features))
    if hasattr(model.named_steps["model"], "feature_importances_"):
        importance = model.named_steps["model"].feature_importances_
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    else:
        importance_df = pd.DataFrame({"feature": feature_names, "importance": np.nan})

    latest = history.sort_values("year").groupby(["state_key", "constituency_key"], dropna=False).tail(1).copy()
    return {
        "available": True,
        "model": model,
        "accuracy": acc,
        "feature_importance": importance_df,
        "latest_history": latest,
        "training_frame": data,
    }


def predict_winners(model_bundle: dict, future_year: int, poll_map: Dict[str, float]) -> pd.DataFrame:
    latest = model_bundle["latest_history"].copy()
    if latest.empty:
        return pd.DataFrame()
    future = latest.copy()
    future["prev_year_value"] = future["year"]
    future["year"] = future_year
    future["year_gap"] = future["year"] - future["prev_year_value"]
    future["incumbent"] = 1.0
    X_future = future[
        ["year", "state", "constituency_no", "prev_party", "prev_margin", "prev_vote_share", "prev_turnout", "year_gap", "incumbent"]
    ].copy()
    model = model_bundle["model"]
    predicted_party = model.predict(X_future)
    if hasattr(model.named_steps["model"], "predict_proba"):
        proba = model.predict_proba(X_future)
        classes = list(model.named_steps["model"].classes_)
        proba_df = pd.DataFrame(proba, columns=classes)
    else:
        classes = list(np.unique(predicted_party))
        proba_df = pd.DataFrame({c: 1.0 / max(len(classes), 1) for c in classes}, index=range(len(predicted_party)))

    if poll_map:
        baseline_counts = model_bundle["training_frame"].groupby("party")["seat"].sum().to_dict()
        baseline_total = sum(baseline_counts.values()) or 1.0
        for party in proba_df.columns:
            poll_share = poll_map.get(party)
            if poll_share is not None:
                baseline_share = (float(baseline_counts.get(party, 0.0)) / baseline_total) * 100.0
                weight = (poll_share + 1e-6) / (baseline_share + 1e-6)
                proba_df[party] = proba_df[party] * (0.75 + 0.25 * weight)
        proba_df = proba_df.div(proba_df.sum(axis=1), axis=0).fillna(0)

    winners = proba_df.idxmax(axis=1)
    confidence = proba_df.max(axis=1)
    result = future[["year", "state", "state_code", "constituency", "constituency_no", "party"]].copy()
    result = result.rename(columns={"party": "last_election_party"})
    result["predicted_winner_party"] = winners.values
    result["confidence"] = confidence.values
    result["previous_margin"] = future["prev_margin"].values
    result["previous_vote_share"] = future["prev_vote_share"].values
    result["previous_turnout"] = future["prev_turnout"].values
    result["adjusted_probabilities"] = proba_df.max(axis=1).values
    result = result.sort_values(["confidence", "constituency"], ascending=[False, True]).reset_index(drop=True)
    return result


def manual_poll_map_from_inputs(parties: List[str], values: Dict[str, float], upload_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    poll_map = {}
    if upload_df is not None and not upload_df.empty:
        tmp = upload_df.copy()
        tmp.columns = [clean_key(c) for c in tmp.columns]
        party_col = None
        share_col = None
        for c in tmp.columns:
            if c in {"party", "parties"}:
                party_col = c
            if c in {"share", "voteshare", "vote_share", "percent", "percentage"}:
                share_col = c
        if party_col is None:
            party_col = tmp.columns[0]
        if share_col is None and len(tmp.columns) > 1:
            share_col = tmp.columns[1]
        if share_col is not None:
            tmp["party_norm"] = tmp[party_col].map(normalize_party_name)
            tmp["share_val"] = coerce_numeric(tmp[share_col])
            tmp = tmp.dropna(subset=["party_norm", "share_val"])
            for _, row in tmp.iterrows():
                poll_map[str(row["party_norm"])] = float(row["share_val"])
    for party, val in values.items():
        if val is not None and not np.isnan(val) and float(val) > 0:
            poll_map[party] = float(val)
    total = sum(v for v in poll_map.values() if v > 0)
    if total > 0:
        poll_map = {k: (v / total) * 100.0 for k, v in poll_map.items()}
    return poll_map


def summarize_prediction_comparison(last_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    if last_df.empty or pred_df.empty:
        return pd.DataFrame()
    latest_year = last_df["year"].max()
    latest = last_df[last_df["year"] == latest_year].copy()
    last = latest.groupby("party")["seat"].sum().reset_index().rename(columns={"seat": "last_seats"})
    pred = pred_df.groupby("predicted_winner_party").size().reset_index(name="predicted_seats").rename(columns={"predicted_winner_party": "party"})
    merged = last.merge(pred, on="party", how="outer").fillna(0)
    merged["seat_change"] = merged["predicted_seats"] - merged["last_seats"]
    merged = merged.sort_values("predicted_seats", ascending=False).reset_index(drop=True)
    return merged


# -----------------------------
# Rendering helpers
# -----------------------------

def render_hero(dark_mode: bool, source_label: str, analysis_df: pd.DataFrame) -> None:
    if analysis_df.empty:
        seats_text = "0"
        year_text = "No data loaded"
    else:
        seats_text = f"{int(analysis_df['seat'].sum()):,}"
        if analysis_df["year"].notna().any():
            yr_min = int(analysis_df["year"].min())
            yr_max = int(analysis_df["year"].max())
            year_text = f"{yr_min} to {yr_max}"
        else:
            year_text = "Unknown years"
    st.markdown(
        f"""
        <div class="hero">
            <div class="badge">Indian Election Data Analysis and Prediction</div>
            <h1>Indian Election Analyzer &amp; Predictor</h1>
            <p>Live GitHub CSV ingestion, interactive visual analytics, and machine-learning forecasts for Lok Sabha and State Assembly elections.</p>
            <p><strong>Source:</strong> {html_lib.escape(source_label)} | <strong>Seats analyzed:</strong> {seats_text} | <strong>Years:</strong> {html_lib.escape(year_text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(df: pd.DataFrame, party_summary: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    total_seats = int(df["seat"].sum()) if not df.empty else 0
    avg_turnout = df["turnout"].mean() if not df.empty and df["turnout"].notna().any() else np.nan
    avg_margin = df["margin"].mean() if not df.empty and df["margin"].notna().any() else np.nan
    top_party = party_summary.iloc[0]["party"] if not party_summary.empty else "N/A"
    top_party_seats = int(party_summary.iloc[0]["Seats"]) if not party_summary.empty else 0
    metrics = [
        ("Total Seats", f"{total_seats:,}", "Selected years"),
        ("Top Party", top_party, f"{top_party_seats:,} seats"),
        ("Avg Turnout", f"{avg_turnout:.1f}%" if pd.notna(avg_turnout) else "N/A", "Across filtered data"),
        ("Avg Margin", f"{avg_margin:,.1f}" if pd.notna(avg_margin) else "N/A", "Winning margin"),
    ]
    for col, (label, value, sub) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <h3>{label}</h3>
                    <p>{value}</p>
                    <div style="color:#64748b;font-size:0.82rem;margin-top:0.15rem;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_overview_tab(df: pd.DataFrame, party_summary: pd.DataFrame, state_summary: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    render_metric_cards(df, party_summary)
    if not party_summary.empty:
        st.dataframe(
            party_summary[["party", "Seats", "Seat_Share", "Avg_Vote_Share", "Avg_Margin", "Avg_Turnout", "State_Count"]].head(10),
            use_container_width=True,
            hide_index=True,
        )
    preview_cols = ["year", "state", "constituency", "party", "winner_votes", "margin", "vote_share"]
    st.markdown("**Data Preview**")
    st.dataframe(df[preview_cols].head(20), use_container_width=True, hide_index=True)
    if not state_summary.empty:
        st.markdown("**State / UT Summary**")
        st.dataframe(state_summary.head(20), use_container_width=True, hide_index=True)


def render_visualizations_tab(df: pd.DataFrame, party_summary: pd.DataFrame, dark_mode: bool, state_mode: bool) -> None:
    st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_party_seats(party_summary, dark_mode), use_container_width=True)
    with right:
        st.plotly_chart(plot_party_treemap(party_summary, dark_mode), use_container_width=True)

    metric_label = "Vote share" if df["vote_share"].notna().any() else "Seat share"
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_share_pie(party_summary, metric_label, dark_mode), use_container_width=True)
    with right:
        st.plotly_chart(plot_trend_lines(df, party_summary, dark_mode), use_container_width=True)

    st.plotly_chart(plot_top_margins(df, dark_mode), use_container_width=True)

    if state_mode and not df["state"].isna().all():
        st.markdown("**State-wise Summary Table**")
        st.dataframe(compute_state_summary(df), use_container_width=True, hide_index=True)


def render_insights_tab(df: pd.DataFrame, party_summary: pd.DataFrame, state_summary: pd.DataFrame) -> str:
    st.markdown('<div class="section-title">Key Insights &amp; Summarization</div>', unsafe_allow_html=True)
    text = generate_insights(df, party_summary, state_summary)
    st.markdown(text)
    st.markdown("**Quick Reference Tables**")
    c1, c2 = st.columns(2)
    with c1:
        if not party_summary.empty:
            st.dataframe(party_summary[["party", "Seats", "Seat_Share", "Avg_Vote_Share"]].head(8), use_container_width=True, hide_index=True)
    with c2:
        top_margins = compute_top_margins(df, top_n=8)
        if not top_margins.empty:
            st.dataframe(top_margins, use_container_width=True, hide_index=True)
    return text


def render_prediction_tab(
    analysis_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    mode: str,
    future_year_default: int,
    dark_mode: bool,
    cache_bust: int,
    selected_state_code: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict], Dict[str, float]]:
    st.markdown('<div class="section-title">Prediction Module</div>', unsafe_allow_html=True)

    if analysis_df.empty:
        st.warning("Prediction is unavailable because no historical data is currently loaded.")
        return None, None, None, {}

    if mode == "Lok Sabha (National)":
        corpus = load_national_prediction_corpus(cache_bust)
    else:
        backfill = load_state_prediction_corpus(cache_bust)
        corpus = pd.concat([backfill, analysis_df], ignore_index=True) if not backfill.empty else analysis_df.copy()
        corpus = corpus.drop_duplicates(subset=["year", "state_key", "constituency_key", "party"], keep="last")

    if corpus.empty:
        st.warning("No training corpus available for the prediction module.")
        return None, None, None, {}

    if "vote_share" not in corpus.columns or corpus["vote_share"].notna().sum() < 8:
        st.info("Vote-share data is sparse, so the predictor will rely more on seat patterns and recent winner transitions.")

    latest_year = int(filtered_df["year"].max()) if filtered_df["year"].notna().any() else datetime.now().year
    year_min = max(latest_year + 1, datetime.now().year + 1, future_year_default)
    year_max = year_min + 15
    future_year = st.number_input("Future election year", min_value=year_min, max_value=year_max, value=max(year_min, future_year_default), step=1)

    st.caption("Default prediction uses historical election results only. Opinion-poll inputs are optional and turned off by default.")
    use_poll_override = st.toggle(
        "Use optional opinion-poll inputs",
        value=False,
        help="Leave this off to let the model predict from historical trends without any manual opinion input.",
    )

    top_parties = corpus.groupby("party")["seat"].sum().sort_values(ascending=False).head(8).index.tolist()
    if not top_parties:
        top_parties = party_summary_top_parties(filtered_df)

    poll_map: Dict[str, float] = {}
    if use_poll_override:
        with st.expander("Optional opinion poll / hypothetical vote % inputs", expanded=True):
            poll_file = st.file_uploader("Upload opinion-poll CSV", type=["csv"], key="poll_upload")
            manual_cols = st.columns(2)
            manual_values = {}
            for idx, party in enumerate(top_parties):
                with manual_cols[idx % 2]:
                    manual_values[party] = st.number_input(
                        f"{party} vote %",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.1,
                        key=f"manual_poll_{party}",
                    )
            upload_df = None
            if poll_file is not None:
                try:
                    upload_df = load_csv_from_bytes(poll_file.getvalue(), poll_file.name, cache_bust)
                    st.success("Opinion-poll CSV loaded.")
                    st.dataframe(upload_df.head(10), use_container_width=True)
                except Exception as exc:
                    st.warning(f"Poll CSV could not be parsed: {exc}")
            poll_map = manual_poll_map_from_inputs(top_parties, manual_values, upload_df)
            if poll_map:
                st.caption("Poll input was normalized to a 100% share base before being blended into predictions.")
                st.dataframe(
                    pd.DataFrame({"party": list(poll_map.keys()), "poll_share": list(poll_map.values())}).sort_values("poll_share", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No positive poll values were provided, so the forecast will still run as a model-only prediction.")
    else:
        st.info("Model-only forecast is active. The prediction will be generated from historical vote share, turnout, margins, and seat trends.")

    predict_now = st.button("Run Model Prediction", type="primary", use_container_width=True)

    if not predict_now and "latest_prediction" in st.session_state and st.session_state.get("prediction_signature") == str((mode, future_year, selected_state_code, sorted(poll_map.items()))):
        cached = st.session_state["latest_prediction"]
    else:
        cached = None

    if predict_now:
        with st.spinner("Training models and generating predictions..."):
            # Vote-share model
            vote_bundle = fit_vote_share_models(corpus)
            if vote_bundle.get("available"):
                vote_forecast = predict_vote_share(vote_bundle, int(future_year), poll_map)
            else:
                vote_forecast = pd.DataFrame()

            # Winning-party classifier
            clf_bundle = fit_winner_classifier(corpus)
            if clf_bundle.get("available"):
                seat_prediction = predict_winners(clf_bundle, int(future_year), poll_map)
            else:
                seat_prediction = pd.DataFrame()

            comparison = summarize_prediction_comparison(filtered_df, seat_prediction) if not seat_prediction.empty else pd.DataFrame()

            st.session_state["latest_prediction"] = {
                "vote_forecast": vote_forecast,
                "seat_prediction": seat_prediction,
                "comparison": comparison,
                "poll_map": poll_map,
                "vote_bundle": vote_bundle,
                "clf_bundle": clf_bundle,
            }
            st.session_state["prediction_signature"] = str((mode, int(future_year), selected_state_code, sorted(poll_map.items())))
    elif cached is not None:
        vote_forecast = cached["vote_forecast"]
        seat_prediction = cached["seat_prediction"]
        comparison = cached["comparison"]
        vote_bundle = cached["vote_bundle"]
        clf_bundle = cached["clf_bundle"]
        poll_map = cached["poll_map"]
    else:
        st.info("Press 'Run Prediction' to train the models and generate forecasts.")
        return None, None, None, poll_map

    # Output blocks
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Future Vote Share Forecast")
        if vote_bundle.get("available") and not vote_forecast.empty:
            st.dataframe(vote_forecast, use_container_width=True, hide_index=True)
            fig_vote = px.bar(
                vote_forecast.head(12),
                x="party",
                y="predicted_vote_share",
                color="party",
                color_discrete_map=PARTY_COLOR_MAP,
                title="Predicted Vote Share by Party",
            )
            fig_vote.update_layout(template=apply_theme(dark_mode), xaxis_title="", yaxis_title="Vote share %", legend_title="")
            st.plotly_chart(fig_vote, use_container_width=True)
        else:
            st.info("Vote-share forecast could not be trained from the available data.")

    with col2:
        st.subheader("Predicted Seat Distribution")
        if clf_bundle.get("available") and not seat_prediction.empty:
            seat_counts = seat_prediction.groupby("predicted_winner_party").size().reset_index(name="Seats").sort_values("Seats", ascending=False)
            st.dataframe(seat_prediction.head(25), use_container_width=True, hide_index=True)
            fig_seats = px.bar(
                seat_counts,
                x="predicted_winner_party",
                y="Seats",
                color="predicted_winner_party",
                color_discrete_map=PARTY_COLOR_MAP,
                title="Predicted Seat Distribution",
            )
            fig_seats.update_layout(template=apply_theme(dark_mode), xaxis_title="", yaxis_title="Seats", legend_title="")
            st.plotly_chart(fig_seats, use_container_width=True)
        else:
            st.info("Winning-party classification could not be trained from the available data.")

    st.subheader("Comparison With Last Election")
    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=comparison["party"], y=comparison["last_seats"], name="Last election"))
        fig_comp.add_trace(go.Bar(x=comparison["party"], y=comparison["predicted_seats"], name="Predicted"))
        fig_comp.update_layout(template=apply_theme(dark_mode), barmode="group", title="Last Election vs Predicted Seats", xaxis_title="", yaxis_title="Seats")
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Comparison data is unavailable because the prediction output could not be generated.")

    # Model metrics and feature importance
    st.subheader("Model Performance")
    perf_cols = st.columns(3)
    vote_r2 = vote_bundle.get("rf_r2") if vote_bundle.get("available") else np.nan
    clf_acc = clf_bundle.get("accuracy") if clf_bundle.get("available") else np.nan
    confidence = None
    if clf_bundle.get("available") and not seat_prediction.empty:
        confidence = float(seat_prediction["confidence"].mean())
    with perf_cols[0]:
        st.metric("Vote-share R2", f"{vote_r2:.3f}" if pd.notna(vote_r2) else "N/A")
    with perf_cols[1]:
        st.metric("Classifier Accuracy", f"{clf_acc:.3f}" if pd.notna(clf_acc) else "N/A")
    with perf_cols[2]:
        st.metric("Average Confidence", f"{confidence:.1%}" if confidence is not None else "N/A")

    if vote_bundle.get("available"):
        st.markdown("**Vote-share model feature importance**")
        feat = vote_bundle["feature_importance"].head(12)
        st.dataframe(feat, use_container_width=True, hide_index=True)
        if not feat.empty:
            fig_imp = px.bar(feat[::-1], x="importance", y="feature", orientation="h", title="Vote-share Feature Importance")
            fig_imp.update_layout(template=apply_theme(dark_mode), xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig_imp, use_container_width=True)
    if clf_bundle.get("available"):
        st.markdown("**Winning-party classifier feature importance**")
        feat = clf_bundle["feature_importance"].head(12)
        st.dataframe(feat, use_container_width=True, hide_index=True)
        if not feat.empty:
            fig_imp2 = px.bar(feat[::-1], x="importance", y="feature", orientation="h", title="Classifier Feature Importance")
            fig_imp2.update_layout(template=apply_theme(dark_mode), xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig_imp2, use_container_width=True)

    return vote_forecast, seat_prediction, clf_bundle if clf_bundle.get("available") else None, poll_map


def party_summary_top_parties(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    return df.groupby("party")["seat"].sum().sort_values(ascending=False).head(8).index.tolist()


def render_export_tab(summary_text: str, party_summary: pd.DataFrame, state_summary: pd.DataFrame, df: pd.DataFrame, dark_mode: bool) -> None:
    st.markdown('<div class="section-title">Export &amp; Extras</div>', unsafe_allow_html=True)
    top_margins = compute_top_margins(df, top_n=10)
    title = "Indian Election Analyzer & Predictor - Report"
    html_content = html_report(summary_text, party_summary, state_summary, top_margins, df, title)
    pdf_bytes = pdf_report_bytes(summary_text, party_summary, state_summary, top_margins, title)
    st.markdown("**Download report**")
    st.download_button("Download HTML report", data=html_content.encode("utf-8"), file_name="indian_election_report.html", mime="text/html", use_container_width=True)
    if pdf_bytes is not None:
        st.download_button("Download PDF report", data=pdf_bytes, file_name="indian_election_report.pdf", mime="application/pdf", use_container_width=True)
    st.markdown(html_download_link(html_content), unsafe_allow_html=True)

    st.markdown("**Download predictions CSV**")
    if "latest_prediction" in st.session_state and st.session_state["latest_prediction"].get("seat_prediction") is not None:
        pred = st.session_state["latest_prediction"]
        if isinstance(pred.get("seat_prediction"), pd.DataFrame) and not pred["seat_prediction"].empty:
            st.download_button(
                "Download seat predictions CSV",
                data=pred["seat_prediction"].to_csv(index=False).encode("utf-8"),
                file_name="seat_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if isinstance(pred.get("vote_forecast"), pd.DataFrame) and not pred["vote_forecast"].empty:
            st.download_button(
                "Download vote forecast CSV",
                data=pred["vote_forecast"].to_csv(index=False).encode("utf-8"),
                file_name="vote_forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("Run a prediction first to enable CSV downloads.")

    st.markdown("**Theme and display**")
    st.caption("Use the sidebar theme toggle to switch between light and dark layouts.")
    if dark_mode:
        st.success("Dark theme is active.")
    else:
        st.info("Light theme is active.")
    st.markdown("**Project Footer**")
    st.write("Made for Election Data Analysis Project - Fully Python-based")


# -----------------------------
# Sidebar and app driver
# -----------------------------

def main() -> None:
    state_lookup = lookup_state_catalog()
    if "refresh_nonce" not in st.session_state:
        st.session_state.refresh_nonce = 0
    if "prediction_signature" not in st.session_state:
        st.session_state.prediction_signature = None

    dark_mode = st.sidebar.toggle("Dark theme", value=False)
    st.markdown(theme_styles(dark_mode), unsafe_allow_html=True)

    st.sidebar.markdown("### Controls")
    mode = st.sidebar.radio("Election scope", ["Lok Sabha (National)", "State Assembly"], index=0)
    selected_state_name = None
    selected_state_code = None
    custom_url = ""
    uploaded_file = st.sidebar.file_uploader("Upload CSV fallback", type=["csv"])

    if mode == "State Assembly":
        options = state_lookup.copy()
        options["label"] = options.apply(
            lambda r: (
                f"{r['state_name']} ({r['state_code']})"
                if r["state_code"] in BUNDLED_STATE_CODES
                else f"{r['state_name']} ({r['state_code']}) - needs URL/upload"
            ),
            axis=1,
        )
        default_state_index = 0
        bundled_matches = options.index[options["state_code"].isin(BUNDLED_STATE_CODES)].tolist()
        if bundled_matches:
            default_state_index = int(bundled_matches[0])
        selected_state_name = st.sidebar.selectbox("State / UT", options["label"].tolist(), index=default_state_index)
        selected_state_code = options.loc[options["label"] == selected_state_name, "state_code"].iloc[0]
        state_info = state_lookup.loc[state_lookup["state_code"] == selected_state_code].iloc[0]
        availability_note = "Bundled example available" if selected_state_code in BUNDLED_STATE_CODES else "Custom URL or CSV needed"
        st.sidebar.caption(f"Selected: {state_info['state_name']} [{state_info['state_status']}] | {availability_note}")
        if selected_state_code in NO_ASSEMBLY_UTS:
            st.sidebar.warning("This UT does not have an assembly election dataset. Use a custom raw GitHub CSV URL or upload a CSV.")
        elif selected_state_code not in BUNDLED_STATE_CODES:
            st.sidebar.info("No bundled online dataset is configured for this state yet. Add a raw GitHub CSV URL or upload a CSV.")
        custom_url = st.sidebar.text_input("Custom raw GitHub CSV URL", value="", placeholder="https://raw.githubusercontent.com/.../your_file.csv")

    refresh_clicked = st.sidebar.button("Load Latest Online Data", use_container_width=True)
    if refresh_clicked:
        st.session_state.refresh_nonce += 1

    source_label = ""
    analysis_df = pd.DataFrame()
    base_df = pd.DataFrame()

    loading_box = st.empty()
    progress = st.progress(0)
    try:
        loading_box.info("Loading election data...")
        progress.progress(10)
        if mode == "Lok Sabha (National)":
            analysis_df, base_df, source_label = load_national_analysis_bundle(st.session_state.refresh_nonce)
        else:
            uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
            if not custom_url.strip() and uploaded_bytes is None and selected_state_code not in BUNDLED_STATE_CODES:
                state_label = state_lookup.loc[state_lookup["state_code"] == selected_state_code, "state_name"].iloc[0]
                loading_box.info(
                    f"No bundled online dataset is configured for {state_label}. Add a raw GitHub CSV URL or upload a CSV to analyze this assembly."
                )
                progress.empty()
                st.stop()
            analysis_df, source_label = load_state_analysis_bundle(
                selected_state_code,
                custom_url,
                uploaded_bytes,
                uploaded_file.name if uploaded_file is not None else None,
                st.session_state.refresh_nonce,
            )
            base_df = analysis_df.copy()
        progress.progress(80)
        if analysis_df.empty:
            raise ValueError("The loaded dataset is empty.")
        loading_box.success("Data loaded successfully.")
        progress.progress(100)
    except Exception as exc:
        loading_box.warning(f"Primary load failed: {exc}")
        if uploaded_file is not None:
            try:
                loading_box.info("Attempting to load the uploaded CSV fallback...")
                uploaded_bytes = uploaded_file.getvalue()
                raw = load_csv_from_bytes(uploaded_bytes, uploaded_file.name, st.session_state.refresh_nonce)
                analysis_df = normalize_generic(raw, state_lookup, f"Uploaded CSV: {uploaded_file.name}")
                base_df = analysis_df.copy()
                source_label = f"Uploaded CSV: {uploaded_file.name}"
                loading_box.success("Loaded uploaded CSV fallback.")
                progress.progress(100)
            except Exception as fallback_exc:
                st.error(f"Unable to load any dataset: {fallback_exc}")
                st.stop()
        else:
            st.error("Unable to load any dataset. Please check the source URL or upload a CSV.")
            st.stop()

    analysis_df = ensure_analysis_schema(analysis_df)
    base_df = ensure_analysis_schema(base_df)

    # Build constituency state lookup after the data is loaded so 2024 national rows can inherit state names.
    if mode == "Lok Sabha (National)" and "state" in analysis_df.columns and analysis_df["state"].isna().any():
        constituency_lookup = build_constituency_state_lookup(analysis_df)
        analysis_df = patch_missing_states(analysis_df, constituency_lookup, state_lookup)
        base_df = patch_missing_states(base_df, constituency_lookup, state_lookup)

    available_years = sort_years(analysis_df["year"].dropna().astype(int).tolist())
    if not available_years:
        available_years = [datetime.now().year]

    default_years = available_years[-3:] if len(available_years) >= 3 else available_years
    year_key = f"year_selector_{hashlib.md5((mode + source_label).encode('utf-8')).hexdigest()[:8]}"
    selected_years = st.sidebar.multiselect("Year / election selector", options=available_years, default=default_years, key=year_key)
    if not selected_years:
        selected_years = available_years

    filtered_df = filtered_years(analysis_df, selected_years)
    if filtered_df.empty:
        st.warning("The selected years do not match any loaded records. Showing the full loaded dataset instead.")
        filtered_df = analysis_df.copy()

    party_summary = compute_party_summary(filtered_df)
    state_summary = compute_state_summary(filtered_df)
    top_party_list = party_summary["party"].head(8).tolist() if not party_summary.empty else []

    render_hero(dark_mode, source_label or mode, filtered_df)

    tabs = st.tabs(["Overview", "Visualizations", "Insights", "Prediction", "Export"])
    with tabs[0]:
        render_overview_tab(filtered_df, party_summary, state_summary)
    with tabs[1]:
        render_visualizations_tab(filtered_df, party_summary, dark_mode, mode == "State Assembly")
    with tabs[2]:
        summary_text = render_insights_tab(filtered_df, party_summary, state_summary)
    with tabs[3]:
        future_default = max(int(filtered_df["year"].max()) + 5 if filtered_df["year"].notna().any() else datetime.now().year + 1, datetime.now().year + 1)
        vote_forecast, seat_prediction, clf_bundle, poll_map = render_prediction_tab(
            analysis_df=analysis_df,
            filtered_df=filtered_df,
            mode=mode,
            future_year_default=future_default,
            dark_mode=dark_mode,
            cache_bust=st.session_state.refresh_nonce,
            selected_state_code=selected_state_code or "",
        )
        if "latest_prediction" in st.session_state and st.session_state["latest_prediction"].get("vote_bundle"):
            st.session_state["latest_vote_bundle"] = st.session_state["latest_prediction"]["vote_bundle"]
            st.session_state["latest_clf_bundle"] = st.session_state["latest_prediction"]["clf_bundle"]
    with tabs[4]:
        summary_text = generate_insights(filtered_df, party_summary, state_summary)
        render_export_tab(summary_text, party_summary, state_summary, filtered_df, dark_mode)

    st.markdown("<footer>Made for Election Data Analysis Project - Fully Python-based</footer>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
