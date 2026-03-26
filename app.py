import json
import os
import re
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ─── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="ExpenseIQ",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0d0f14; color: #e8eaf0; }

[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

[data-testid="stMetric"] {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 20px 24px !important;
}
[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: #f0f2f8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.9rem !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

[data-testid="stSelectbox"] > div > div,
[data-testid="stDateInput"] > div > div > input,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
textarea {
    background: #1a1d26 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(108,99,255,0.35) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1e2130 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

hr { border-color: #1e2130 !important; }

[data-testid="stAlert"] { border-radius: 10px !important; }

[data-baseweb="tab-list"] {
    background: #13151c !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #6b7280 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #6c63ff !important;
    color: white !important;
}

.card {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
}

.predict-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6c63ff22, #4f46e511);
    border: 1px solid #6c63ff44;
    color: #a89fff;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.big-amount {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0f2f8;
    line-height: 1;
}

.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f0f2f8;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.1;
}
.header-sub {
    color: #6b7280;
    font-size: 0.9rem;
    margin-top: 4px;
}

.lock-box {
    background: #13151c;
    border: 1px solid #2a2d3e;
    border-radius: 16px;
    padding: 40px 32px;
    text-align: center;
    margin-top: 24px;
}
.lock-icon { font-size: 3rem; margin-bottom: 12px; }
.lock-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #f0f2f8;
    margin-bottom: 8px;
}
.lock-sub { color: #6b7280; font-size: 0.9rem; line-height: 1.5; }

.progress-track {
    background: #1e2130;
    border-radius: 999px;
    height: 8px;
    margin: 20px 0 8px 0;
    overflow: hidden;
}
.progress-fill {
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, #6c63ff, #a89fff);
    transition: width 0.4s ease;
}
.progress-label {
    color: #6b7280;
    font-size: 0.78rem;
    text-align: right;
}

.diff-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
}
.over  { background: #3f1515; color: #f87171; }
.under { background: #0f3025; color: #34d399; }
.exact { background: #1a1d26; color: #9ca3af; }

.budget-row {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.budget-bar-track {
    background: #1e2130;
    border-radius: 999px;
    height: 6px;
    margin: 8px 0 4px 0;
    overflow: hidden;
}
.budget-bar-fill-ok   { height: 6px; border-radius: 999px; background: linear-gradient(90deg, #10b981, #34d399); }
.budget-bar-fill-warn { height: 6px; border-radius: 999px; background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.budget-bar-fill-over { height: 6px; border-radius: 999px; background: linear-gradient(90deg, #ef4444, #f87171); }
</style>
""", unsafe_allow_html=True)


# ─── Constants ──────────────────────────────────────────────
SAVE_FILE = "my_expenses.csv"
CONTEXT_FILE = "expense_context.json"
COLS = ["Date", "Description", "Category", "Payment_Mode", "Amount"]
DAYS_REQUIRED = 10

CATEGORIES = [
    "Food", "Transport", "Entertainment", "Health", "Rent", "Shopping", "Utilities",
    "Education", "Travel",
]
PAYMENT_MODES = ["UPI", "Card", "Cash", "Bank Transfer", "Other"]

CATEGORY_COLORS = {
    "Food": "#f59e0b",
    "Transport": "#3b82f6",
    "Entertainment": "#ec4899",
    "Health": "#10b981",
    "Rent": "#8b5cf6",
    "Shopping": "#f97316",
    "Utilities": "#06b6d4",
    "Education": "#84cc16",
    "Travel": "#f43f5e",
}

COLUMN_ALIASES = {
    "date": [
        "txn date", "transaction date", "date", "posting date", "value date", "transaction date/time"
    ],
    "description": [
        "description", "narration", "remarks", "transaction remarks", "details",
        "particulars", "transaction details", "transaction description"
    ],
    "debit": [
        "dr amount", "debit", "debit amount", "withdrawal", "withdrawal amount",
        "paid out", "money out", "debit amt"
    ],
    "credit": [
        "cr amount", "credit", "credit amount", "deposit", "deposit amount",
        "paid in", "money in", "credit amt"
    ],
    "amount": [
        "amount", "transaction amount", "txn amount"
    ],
    "type": [
        "dr/cr", "type", "transaction type", "txn type", "cr/dr"
    ],
    "balance": [
        "balance", "closing balance", "running balance", "available balance", "acct balance"
    ],
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#13151c",
    font=dict(family="DM Sans", color="#9ca3af"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="#13151c", bordercolor="#1e2130", borderwidth=1),
)


# ─── Persistence ────────────────────────────────────────────
def ensure_expense_schema(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for c in COLS:
        if c not in df.columns:
            if c == "Date":
                df[c] = pd.NaT
            elif c == "Amount":
                df[c] = 0.0
            else:
                df[c] = ""
    df = df[COLS]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["Description"] = df["Description"].fillna("—").astype(str)
    df["Category"] = df["Category"].fillna("Shopping").astype(str)
    df["Payment_Mode"] = df["Payment_Mode"].fillna("Other").astype(str)
    df = df.dropna(subset=["Date"])
    df = df[df["Amount"] > 0].copy()
    return df.reset_index(drop=True)


def load_saved_context() -> dict:
    default_context = {
        "inferred_monthly_income": None,
        "current_balance": None,
        "budgets": {},
    }
    if not os.path.exists(CONTEXT_FILE):
        return default_context

    try:
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            "inferred_monthly_income": raw.get("inferred_monthly_income"),
            "current_balance": raw.get("current_balance"),
            "budgets": raw.get("budgets", {}),
        }
    except Exception:
        return default_context


def save_context() -> None:
    payload = {
        "inferred_monthly_income": (
            float(st.session_state.inferred_monthly_income)
            if st.session_state.inferred_monthly_income is not None
            else None
        ),
        "current_balance": (
            float(st.session_state.current_balance)
            if st.session_state.current_balance is not None
            else None
        ),
        "budgets": st.session_state.get("budgets", {}),
    }
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


if "expenses" not in st.session_state:
    if os.path.exists(SAVE_FILE):
        try:
            st.session_state.expenses = ensure_expense_schema(pd.read_csv(SAVE_FILE))
        except Exception:
            st.session_state.expenses = pd.DataFrame(columns=COLS)
    else:
        st.session_state.expenses = pd.DataFrame(columns=COLS)

saved_context = load_saved_context()

if "inferred_monthly_income" not in st.session_state:
    st.session_state.inferred_monthly_income = saved_context.get("inferred_monthly_income")

if "current_balance" not in st.session_state:
    st.session_state.current_balance = saved_context.get("current_balance")

if "budgets" not in st.session_state:
    st.session_state.budgets = saved_context.get("budgets", {})


def save_expenses() -> None:
    st.session_state.expenses = ensure_expense_schema(st.session_state.expenses)
    st.session_state.expenses.to_csv(SAVE_FILE, index=False)
    save_context()


# ─── Bank parsing helpers ───────────────────────────────────
def clean_col_name(name) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[\n\r\t]+", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def pick_best_column(columns, alias_list):
    for alias in alias_list:
        if alias in columns:
            return alias
    return None


def normalize_amount_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = (
        s.str.replace(",", "", regex=False)
         .str.replace("₹", "", regex=False)
         .str.replace("INR", "", regex=False)
         .str.replace("CR.", "", regex=False)
         .str.replace("DR.", "", regex=False)
         .str.replace("Cr.", "", regex=False)
         .str.replace("Dr.", "", regex=False)
         .str.replace("Cr", "", regex=False)
         .str.replace("Dr", "", regex=False)
         .str.strip()
    )
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")


def parse_statement_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    d2 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return d1 if d1.notna().sum() >= d2.notna().sum() else d2


@st.cache_data(show_spinner=False)
def read_uploaded_statement(file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    file_obj = BytesIO(raw_bytes)
    lower_name = file_name.lower()

    if lower_name.endswith(".csv"):
        raw = pd.read_csv(file_obj, header=None, dtype=str, encoding_errors="ignore")
    else:
        file_obj.seek(0)
        raw = pd.read_excel(file_obj, header=None, dtype=str, engine="openpyxl")

    best_row = None
    best_score = -1

    for i, row in raw.iterrows():
        vals = [clean_col_name(x) for x in row.tolist()]
        score = 0
        for v in vals:
            if v in COLUMN_ALIASES["date"]:
                score += 3
            if v in COLUMN_ALIASES["description"]:
                score += 3
            if v in COLUMN_ALIASES["debit"]:
                score += 2
            if v in COLUMN_ALIASES["credit"]:
                score += 2
            if v in COLUMN_ALIASES["amount"]:
                score += 1
            if v in COLUMN_ALIASES["type"]:
                score += 1
        if score > best_score:
            best_score = score
            best_row = i

    if best_row is None or best_score < 4:
        raise ValueError("Could not detect the transaction table header automatically.")

    file_obj.seek(0)
    if lower_name.endswith(".csv"):
        df = pd.read_csv(file_obj, skiprows=best_row, dtype=str, encoding_errors="ignore")
    else:
        df = pd.read_excel(file_obj, skiprows=best_row, dtype=str, engine="openpyxl")

    df.columns = [clean_col_name(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df


def normalize_statement(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    cols = set(df.columns)

    date_col = pick_best_column(cols, COLUMN_ALIASES["date"])
    desc_col = pick_best_column(cols, COLUMN_ALIASES["description"])
    debit_col = pick_best_column(cols, COLUMN_ALIASES["debit"])
    credit_col = pick_best_column(cols, COLUMN_ALIASES["credit"])
    amount_col = pick_best_column(cols, COLUMN_ALIASES["amount"])
    type_col = pick_best_column(cols, COLUMN_ALIASES["type"])

    if date_col is None or desc_col is None:
        raise ValueError("Could not find required date and description columns.")

    dates = parse_statement_dates(df[date_col])
    descriptions = df[desc_col].fillna("").astype(str).str.strip()

    debit = normalize_amount_series(df[debit_col]) if debit_col else pd.Series(np.nan, index=df.index)
    credit = normalize_amount_series(df[credit_col]) if credit_col else pd.Series(np.nan, index=df.index)
    amount = normalize_amount_series(df[amount_col]) if amount_col else pd.Series(np.nan, index=df.index)
    txn_type = df[type_col].fillna("").astype(str).str.lower().str.strip() if type_col else pd.Series("", index=df.index)

    expense_amount = pd.Series(np.nan, index=df.index, dtype=float)

    if debit_col:
        expense_amount = debit.copy()

    if amount_col and type_col:
        dr_mask = txn_type.str.contains(r"\bdr\b|debit|withdraw|out", regex=True, na=False)
        expense_amount = pd.Series(np.where(dr_mask, amount, expense_amount), index=df.index, dtype=float)

    if amount_col and expense_amount.isna().all():
        negative_mask = amount < 0
        expense_amount = pd.Series(np.where(negative_mask, amount.abs(), np.nan), index=df.index, dtype=float)

    if debit_col and amount_col:
        expense_amount = debit.fillna(expense_amount)

    # Extra protection: never treat explicit credits as expenses.
    if credit_col:
        expense_amount = expense_amount.where(credit.isna() | (credit <= 0), np.nan)

    out = pd.DataFrame({
        "Date": dates,
        "Description": descriptions,
        "Amount": pd.to_numeric(expense_amount, errors="coerce"),
    })

    out = out.dropna(subset=["Date", "Amount"])
    out = out[out["Amount"] > 0]
    out = out[out["Description"].str.strip() != ""]
    out = out[~out["Description"].str.lower().isin(["nan", "none"])]
    return out.reset_index(drop=True)


def extract_statement_financial_context(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    cols = set(df.columns)

    date_col = pick_best_column(cols, COLUMN_ALIASES["date"])
    desc_col = pick_best_column(cols, COLUMN_ALIASES["description"])
    credit_col = pick_best_column(cols, COLUMN_ALIASES["credit"])
    amount_col = pick_best_column(cols, COLUMN_ALIASES["amount"])
    type_col = pick_best_column(cols, COLUMN_ALIASES["type"])
    balance_col = pick_best_column(cols, COLUMN_ALIASES["balance"])

    if date_col is None:
        return {
            "inferred_monthly_income": None,
            "current_balance": None,
        }

    dates = parse_statement_dates(df[date_col])
    descriptions = df[desc_col].fillna("").astype(str).str.strip() if desc_col else pd.Series("", index=df.index)
    credit = normalize_amount_series(df[credit_col]) if credit_col else pd.Series(np.nan, index=df.index)
    amount = normalize_amount_series(df[amount_col]) if amount_col else pd.Series(np.nan, index=df.index)
    balance = normalize_amount_series(df[balance_col]) if balance_col else pd.Series(np.nan, index=df.index)
    txn_type = df[type_col].fillna("").astype(str).str.lower().str.strip() if type_col else pd.Series("", index=df.index)

    if amount_col and type_col and credit_col is None:
        cr_mask = txn_type.str.contains(r"\bcr\b|credit|deposit|in", regex=True, na=False)
        credit = pd.Series(np.where(cr_mask, amount.abs(), np.nan), index=df.index, dtype=float)

    credit_df = pd.DataFrame({
        "Date": dates,
        "Description": descriptions,
        "Credit": pd.to_numeric(credit, errors="coerce"),
    })
    credit_df = credit_df.dropna(subset=["Date", "Credit"])
    credit_df = credit_df[credit_df["Credit"] > 0].copy()

    if not credit_df.empty:
        noisy_credit_keywords = [
            "refund", "reversal", "cashback", "interest", "upi", "imps", "neft", "rtgs",
            "transfer", "self", "cash deposit", "chargeback"
        ]
        salary_like_keywords = [
            "salary", "sal", "payroll", "wage", "stipend", "allowance", "bonus", "incentive",
            "company", "pvt ltd", "ltd", "solutions", "technologies"
        ]

        desc_lower = credit_df["Description"].str.lower()
        salary_like_mask = desc_lower.apply(lambda x: any(k in x for k in salary_like_keywords))
        noisy_mask = desc_lower.apply(lambda x: any(k in x for k in noisy_credit_keywords))

        salary_credits = credit_df[salary_like_mask].copy()
        clean_credits = credit_df[~noisy_mask].copy()
        fallback_credits = clean_credits if not clean_credits.empty else credit_df.copy()

        inferred_monthly_income = None

        if not salary_credits.empty:
            salary_monthly = salary_credits.groupby(salary_credits["Date"].dt.to_period("M"))["Credit"].sum()
            salary_monthly = salary_monthly[salary_monthly > 0]
            if len(salary_monthly) >= 2:
                inferred_monthly_income = float(salary_monthly.tail(6).median())

        if inferred_monthly_income is None and not fallback_credits.empty:
            monthly_credit_totals = fallback_credits.groupby(fallback_credits["Date"].dt.to_period("M"))["Credit"].sum()
            monthly_credit_totals = monthly_credit_totals[monthly_credit_totals > 0]
            if len(monthly_credit_totals) >= 2:
                recent = monthly_credit_totals.tail(6).astype(float)
                clipped = recent.clip(upper=recent.quantile(0.85))
                inferred_monthly_income = float(clipped.median() * 0.80)

        if inferred_monthly_income is not None and inferred_monthly_income <= 0:
            inferred_monthly_income = None
    else:
        inferred_monthly_income = None

    balance_df = pd.DataFrame({
        "Date": dates,
        "Balance": pd.to_numeric(balance, errors="coerce"),
    }).dropna(subset=["Date", "Balance"]).sort_values("Date")
    current_balance = float(balance_df["Balance"].iloc[-1]) if not balance_df.empty else None

    return {
        "inferred_monthly_income": inferred_monthly_income,
        "current_balance": current_balance,
    }


def apply_income_constraint(predicted_total: float, inferred_monthly_income=None, current_balance=None, recent_monthly_spend=None) -> float:
    adjusted = float(max(predicted_total, 0.0))

    income = float(inferred_monthly_income) if inferred_monthly_income is not None and inferred_monthly_income > 0 else None
    recent = float(recent_monthly_spend) if recent_monthly_spend is not None and recent_monthly_spend > 0 else None
    balance = float(current_balance) if current_balance is not None and current_balance > 0 else None

    if income is None and recent is None and balance is None:
        return adjusted

    # Keep a realistic baseline so constraints do not crush the model completely.
    baseline = adjusted
    if recent is not None:
        baseline = max(baseline, recent * 0.85)
    if income is not None:
        baseline = max(baseline, income * 0.75)

    # Income should act like a soft budget signal, not a hard clamp.
    if income is not None:
        reference = recent if recent is not None else income
        soft_limit = max(income * 1.10, reference * 1.05)
        hard_limit = max(income * 1.30, reference * 1.15)

        if adjusted <= soft_limit:
            pass
        elif adjusted <= hard_limit:
            adjusted = (0.60 * adjusted) + (0.40 * soft_limit)
        else:
            adjusted = soft_limit + 0.25 * (adjusted - soft_limit)
            adjusted = min(adjusted, hard_limit)

    # Liquidity check: only rein in clearly impossible forecasts.
    if balance is not None:
        refill = income if income is not None else (recent if recent is not None else 0.0)
        if refill > 0:
            liquidity_soft = balance + 0.65 * refill
            liquidity_hard = balance + 0.95 * refill

            if adjusted > liquidity_soft:
                adjusted = min((0.65 * adjusted) + (0.35 * liquidity_soft), liquidity_hard)

    adjusted = max(adjusted, baseline * 0.85)
    return float(max(adjusted, 0.0))


CATEGORY_KEYWORDS = {
    "Food": ["swiggy", "zomato", "zepto", "blinkit", "restaurant", "cafe", "dominos", "pizza", "burger", "food"],
    "Transport": ["uber", "ola", "rapido", "metro", "fuel", "petrol", "diesel", "irctc", "bus", "auto", "transport", "fastag"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "shopping", "store", "mart", "retail", "meesho"],
    "Utilities": ["airtel", "jio", "vi", "electricity", "water", "bill", "broadband", "wifi", "recharge", "utility", "bses"],
    "Entertainment": ["netflix", "spotify", "prime", "hotstar", "bookmyshow", "movie", "cinema", "game"],
    "Health": ["pharmacy", "hospital", "clinic", "apollo", "medicine", "lab", "health"],
    "Rent": ["rent", "house rent", "flat rent", "pg rent", "hostel rent"],
    "Education": ["udemy", "coursera", "school", "college", "university", "tuition", "fees", "course", "book", "library", "exam"],
    "Travel": ["hotel", "airbnb", "oyo", "makemytrip", "goibibo", "flight", "airline", "indigo", "spicejet", "holiday", "trip", "tour"],
}


def detect_payment_mode(text: str) -> str:
    t = str(text).lower()
    if any(k in t for k in ["upi", "gpay", "google pay", "phonepe", "paytm", "bhim", "imps", "upi/"]):
        return "UPI"
    if any(k in t for k in ["atm", "cash", "cash withdrawal"]):
        return "Cash"
    if any(k in t for k in ["card", "pos", "visa", "mastercard", "debit card", "credit card"]):
        return "Card"
    if any(k in t for k in ["neft", "rtgs", "bank transfer", "transfer"]):
        return "Bank Transfer"
    return "Other"


def detect_category(desc: str) -> str:
    d = str(desc).lower()
    for cat, keys in CATEGORY_KEYWORDS.items():
        if any(k in d for k in keys):
            return cat
    return "Shopping"



def statement_to_expenses(df_norm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_norm.iterrows():
        desc = str(r["Description"]).strip() or "—"
        rows.append({
            "Date": pd.Timestamp(r["Date"]),
            "Description": desc,
            "Category": detect_category(desc),
            "Payment_Mode": detect_payment_mode(desc),
            "Amount": round(float(r["Amount"]), 2),
        })
    return pd.DataFrame(rows, columns=COLS)


# ─── Forecast helpers ───────────────────────────────────────
def build_daily_series(expense_df: pd.DataFrame) -> pd.Series:
    daily_ts = (
        expense_df.groupby(expense_df["Date"].dt.normalize())["Amount"]
        .sum()
        .sort_index()
    )
    full_range = pd.date_range(daily_ts.index.min(), daily_ts.index.max(), freq="D")
    daily_ts = daily_ts.reindex(full_range, fill_value=0.0)
    daily_ts.index.name = "Date"
    return daily_ts.astype(float)



def prepare_training_series(daily_ts: pd.Series) -> pd.Series:
    ts = daily_ts.copy().astype(float)

    if len(ts) > 180:
        ts = ts.tail(180).copy()

    positive = ts[ts > 0]
    if len(positive) >= 12:
        q95 = float(positive.quantile(0.95))
        median_pos = float(positive.median())
        clip_upper = max(q95, median_pos * 2.75)
        ts = ts.clip(lower=0.0, upper=clip_upper)

    return ts



def make_ts_features(series: pd.Series) -> pd.DataFrame:
    df_ts = pd.DataFrame({"y": series.astype(float)})
    idx = df_ts.index

    df_ts["lag_1"] = df_ts["y"].shift(1)
    df_ts["lag_2"] = df_ts["y"].shift(2)
    df_ts["lag_3"] = df_ts["y"].shift(3)
    df_ts["lag_7"] = df_ts["y"].shift(7)
    df_ts["lag_14"] = df_ts["y"].shift(14)
    df_ts["lag_21"] = df_ts["y"].shift(21)

    df_ts["roll_mean_3"] = df_ts["y"].shift(1).rolling(3).mean()
    df_ts["roll_mean_7"] = df_ts["y"].shift(1).rolling(7).mean()
    df_ts["roll_mean_14"] = df_ts["y"].shift(1).rolling(14).mean()
    df_ts["roll_mean_21"] = df_ts["y"].shift(1).rolling(21).mean()
    df_ts["roll_std_7"] = df_ts["y"].shift(1).rolling(7).std()
    df_ts["roll_max_7"] = df_ts["y"].shift(1).rolling(7).max()
    df_ts["roll_nonzero_7"] = df_ts["y"].shift(1).rolling(7).apply(lambda x: float(np.count_nonzero(x)), raw=True)

    df_ts["dow"] = idx.dayofweek
    df_ts["dom"] = idx.day
    df_ts["month"] = idx.month
    df_ts["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df_ts["week_of_year"] = idx.isocalendar().week.astype(int)
    df_ts["quarter"] = idx.quarter

    df_ts["dow_sin"] = np.sin(2 * np.pi * df_ts["dow"] / 7)
    df_ts["dow_cos"] = np.cos(2 * np.pi * df_ts["dow"] / 7)
    df_ts["month_sin"] = np.sin(2 * np.pi * df_ts["month"] / 12)
    df_ts["month_cos"] = np.cos(2 * np.pi * df_ts["month"] / 12)

    return df_ts.dropna().copy()



def train_best_forecast_model(daily_ts: pd.Series):
    feat_df = make_ts_features(daily_ts)
    if len(feat_df) < 30:
        return None, None, None, None

    feature_cols = [c for c in feat_df.columns if c != "y"]
    X = feat_df[feature_cols]
    y = feat_df["y"]

    split = max(int(len(feat_df) * 0.8), len(feat_df) - 14)
    split = min(max(split, 20), len(feat_df) - 1)

    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    candidates = {
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=3,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    best_name = None
    best_model = None
    best_mae = float("inf")

    for name, mdl in candidates.items():
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_val)
        pred = np.clip(pred, 0, None)
        mae = mean_absolute_error(y_val, pred)
        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_model = mdl

    best_model.fit(X, y)
    return best_model, best_name, feature_cols, best_mae



def forecast_future_days(daily_ts: pd.Series, model, feature_cols, days_ahead: int) -> pd.Series:
    history = daily_ts.copy().astype(float)
    preds = []

    for _ in range(days_ahead):
        next_date = history.index.max() + pd.Timedelta(days=1)
        temp_series = pd.concat([history, pd.Series([0.0], index=[next_date])])
        feat_df = make_ts_features(temp_series)
        x_next = feat_df.loc[[next_date], feature_cols]

        raw_pred = max(float(model.predict(x_next)[0]), 0.0)

        recent_7_avg = float(history.tail(min(7, len(history))).mean())
        recent_30_avg = float(history.tail(min(30, len(history))).mean())
        recent_90_avg = float(history.tail(min(90, len(history))).mean())

        hist_tail = history.tail(min(180, len(history))).astype(float)
        positive_tail = hist_tail[hist_tail > 0]
        p95 = float(np.percentile(positive_tail.values, 95)) if len(positive_tail) else 0.0

        blended = (0.50 * raw_pred) + (0.30 * recent_7_avg) + (0.20 * recent_30_avg)

        trend_soft = max(recent_30_avg * 1.18, recent_90_avg * 1.10, 250.0)
        trend_hard = max(p95 * 1.10, recent_30_avg * 1.55, recent_90_avg * 1.30, 350.0)

        if blended <= trend_soft:
            pred = blended
        elif blended <= trend_hard:
            pred = trend_soft + 0.35 * (blended - trend_soft)
        else:
            pred = trend_soft + 0.35 * (trend_hard - trend_soft)

        pred = float(np.clip(pred, 0.0, trend_hard))

        history.loc[next_date] = pred
        preds.append((next_date, pred))

    return pd.Series([p for _, p in preds], index=[d for d, _ in preds], name="Predicted")



def build_category_ratios(df_exp: pd.DataFrame):
    cutoff = df_exp["Date"].max() - pd.Timedelta(days=60)
    recent_df = df_exp[df_exp["Date"] >= cutoff].copy()
    ratio_source = recent_df if not recent_df.empty else df_exp.copy()

    cat_ratio = ratio_source.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    cat_ratio = cat_ratio[cat_ratio > 0]
    if cat_ratio.empty:
        return None
    return cat_ratio / cat_ratio.sum()


@st.cache_data(show_spinner=False)
def compute_forecast_cached(
    expense_df: pd.DataFrame,
    inferred_monthly_income,
    current_balance,
    recent_monthly_spend,
):
    df_local = ensure_expense_schema(expense_df)
    if df_local.empty:
        return {"status": "empty"}

    daily_ts = build_daily_series(df_local)
    training_ts = prepare_training_series(daily_ts)
    if len(training_ts) < 30:
        return {"status": "insufficient_history"}

    model_ts, best_model_name, feature_cols, best_mae = train_best_forecast_model(training_ts)
    if model_ts is None:
        return {"status": "insufficient_features"}

    cat_ratio = build_category_ratios(df_local)
    if cat_ratio is None:
        return {"status": "insufficient_categories"}

    months_ahead = ["Next Month", "Month +2", "Month +3"]
    last_date = training_ts.index.max()
    third_future = (last_date + pd.DateOffset(months=3)).normalize()
    days_needed = (third_future - last_date).days + 31
    future_pred = forecast_future_days(training_ts, model_ts, feature_cols, days_needed)

    forecast_totals = {}
    forecast_rows = []
    current_ref = last_date.normalize()

    for i, label in enumerate(months_ahead, start=1):
        target = current_ref + pd.DateOffset(months=i)
        y_m = future_pred[(future_pred.index.year == target.year) & (future_pred.index.month == target.month)]
        monthly_total = float(y_m.sum())
        monthly_total = apply_income_constraint(
            monthly_total,
            inferred_monthly_income=inferred_monthly_income,
            current_balance=current_balance,
            recent_monthly_spend=recent_monthly_spend,
        )
        forecast_totals[label] = monthly_total

        for cat, ratio in cat_ratio.items():
            amt = monthly_total * float(ratio)
            if amt > 0:
                forecast_rows.append({
                    "Period": label,
                    "Category": cat,
                    "Amount": round(float(amt), 2),
                })

    return {
        "status": "ok",
        "months_ahead": months_ahead,
        "forecast_df": pd.DataFrame(forecast_rows),
        "totals": pd.Series(forecast_totals).reindex(months_ahead),
        "best_model_name": best_model_name,
        "best_mae": best_mae,
    }


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="header-title">💸 ExpenseIQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">ML-powered expense tracker</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Add Expense")

    exp_date = st.date_input("Date", value=date.today(), key="dt")
    description = st.text_input("Description", placeholder="e.g. Groceries at BigBazaar", key="desc")
    category = st.selectbox("Category", CATEGORIES, key="cat")
    payment_mode = st.selectbox("Payment Mode", PAYMENT_MODES, key="pay")

    amount = st.number_input(
        "Amount (₹)",
        min_value=0.0,
        value=0.0,
        step=10.0,
        format="%.2f",
        key="amt",
        help="Enter the actual amount you spent.",
    )

    add_clicked = st.button("➕  Add Expense")
    if add_clicked:
        if amount <= 0:
            st.error("Please enter an amount greater than ₹0.")
        else:
            new_row = pd.DataFrame([{
                "Date": pd.Timestamp(exp_date),
                "Description": description.strip() or "—",
                "Category": category,
                "Payment_Mode": payment_mode,
                "Amount": round(float(amount), 2),
            }])
            st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index=True)
            save_expenses()
            st.success(f"✅ ₹{amount:,.0f} added for {category}!")

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Bank Statement (CSV/XLSX)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue()
            bank_df_raw = read_uploaded_statement(uploaded_file.name, raw_bytes)
            bank_df_norm = normalize_statement(bank_df_raw)
            imported_preview = statement_to_expenses(bank_df_norm)

            financial_context = extract_statement_financial_context(bank_df_raw)
            st.session_state.inferred_monthly_income = financial_context.get("inferred_monthly_income")
            st.session_state.current_balance = financial_context.get("current_balance")
            save_context()

            st.info(
                f"Detected {len(imported_preview)} debit transactions. "
                f"Credits are excluded from expense totals."
            )

            if st.button("📥  Import Statement Expenses"):
                if imported_preview.empty:
                    st.warning("No debit expense rows found in this file.")
                else:
                    st.session_state.expenses = pd.concat(
                        [st.session_state.expenses, imported_preview],
                        ignore_index=True,
                    )
                    st.session_state.expenses = ensure_expense_schema(st.session_state.expenses)
                    st.session_state.expenses = st.session_state.expenses.drop_duplicates(
                        subset=["Date", "Description", "Amount"], keep="first"
                    ).reset_index(drop=True)
                    save_expenses()
                    st.success(f"✅ Imported {len(imported_preview)} expenses from statement.")
        except Exception as e:
            st.error(f"Could not read this statement: {e}")


# ─── Main dataset ───────────────────────────────────────────
df = ensure_expense_schema(st.session_state.expenses)
st.session_state.expenses = df

now = pd.Timestamp(datetime.now()).normalize()
total = float(df["Amount"].sum()) if not df.empty else 0.0
count = int(len(df))
unique_days = int(df["Date"].dt.normalize().nunique()) if not df.empty else 0
forecast_unlocked = unique_days >= DAYS_REQUIRED

this_month_mask = (df["Date"].dt.year == now.year) & (df["Date"].dt.month == now.month)
last_month_ref = now - pd.DateOffset(months=1)
last_month_mask = (df["Date"].dt.year == last_month_ref.year) & (df["Date"].dt.month == last_month_ref.month)

this_month = df[this_month_mask]
last_month = df[last_month_mask]
month_total = float(this_month["Amount"].sum()) if not this_month.empty else 0.0
prev_total = float(last_month["Amount"].sum()) if not last_month.empty else 0.0
delta_pct = ((month_total - prev_total) / prev_total * 100) if prev_total > 0 else 0.0

cat_totals_sorted = (
    df.groupby("Category")["Amount"]
    .sum()
    .reset_index()
    .rename(columns={"Amount": "Total"})
)
cat_totals_sorted = cat_totals_sorted[cat_totals_sorted["Total"] > 0].sort_values("Total", ascending=False).reset_index(drop=True)

recent_monthly_spend = None
if not df.empty:
    monthly_spend_series = (
        df.groupby(df["Date"].dt.to_period("M"))["Amount"]
        .sum()
        .sort_index()
    )
    if not monthly_spend_series.empty:
        recent_monthly_spend = float(monthly_spend_series.tail(min(3, len(monthly_spend_series))).median())


# ─── Header / metrics ───────────────────────────────────────
st.markdown('<p class="header-title">Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="header-sub">Your ML-powered spending overview</p>', unsafe_allow_html=True)
st.markdown("")

c1, c2 = st.columns(2)
with c1:
    st.metric("Total Spent", f"₹{total:,.0f}", f"{count} entries")
with c2:
    st.metric("This Month", f"₹{month_total:,.0f}", f"{delta_pct:+.1f}% vs last month" if prev_total > 0 else "First month")

st.markdown("")

if not cat_totals_sorted.empty:
    cols_per_row = 4
    rows = [cat_totals_sorted.iloc[i:i + cols_per_row] for i in range(0, len(cat_totals_sorted), cols_per_row)]
    for row_df in rows:
        cols = st.columns(len(row_df))
        for col, (_, r) in zip(cols, row_df.iterrows()):
            color = CATEGORY_COLORS.get(r["Category"], "#6b7280")
            share = (r["Total"] / total * 100) if total > 0 else 0
            with col:
                st.markdown(f"""
                <div style="background:#13151c;border:1px solid #1e2130;border-top:3px solid {color};
                            border-radius:12px;padding:18px 20px;">
                    <div style="color:#6b7280;font-size:0.72rem;text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:6px;">{r['Category']}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;
                                color:#f0f2f8;line-height:1;">₹{r['Total']:,.0f}</div>
                    <div style="color:#6b7280;font-size:0.75rem;margin-top:4px;">{share:.1f}% of total</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("")

forecast_label = "🔮  Forecast" if forecast_unlocked else f"🔒  Forecast ({unique_days}/{DAYS_REQUIRED} days)"
tab1, tab2, tab3, tab4 = st.tabs(["📈  Charts", "📋  Expenses", forecast_label, "🎯  Budgets"])


# ── Tab 1: Charts ────────────────────────────────────────────
with tab1:
    if df.empty:
        st.info("Add or import some expenses to see your charts.")
    else:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            daily = df.copy()
            daily["Day"] = daily["Date"].dt.strftime("%Y-%m-%d")
            daily = daily.groupby("Day")["Amount"].sum().reset_index().sort_values("Day")

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=daily["Day"], y=daily["Amount"],
                mode="lines+markers",
                line=dict(color="#6c63ff", width=3),
                marker=dict(size=7, color="#6c63ff", line=dict(color="#0d0f14", width=2)),
                fill="tozeroy",
                fillcolor="rgba(108,99,255,0.08)",
                name="Daily Total",
            ))
            fig_line.update_layout(
                **PLOTLY_LAYOUT, title="Daily Spending Trend",
                xaxis=dict(gridcolor="#1e2130", showgrid=True),
                yaxis=dict(gridcolor="#1e2130", showgrid=True, tickprefix="₹"),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        with col_right:
            cat_totals = df.groupby("Category")["Amount"].sum().reset_index()
            cat_totals = cat_totals[cat_totals["Amount"] > 0]
            colors = [CATEGORY_COLORS.get(c, "#6b7280") for c in cat_totals["Category"]]

            fig_donut = go.Figure(go.Pie(
                labels=cat_totals["Category"],
                values=cat_totals["Amount"],
                hole=0.6,
                marker=dict(colors=colors, line=dict(color="#0d0f14", width=2)),
                textinfo="percent",
                textfont=dict(size=12),
            ))
            fig_donut.update_layout(
                **PLOTLY_LAYOUT, title="By Category", showlegend=True,
                annotations=[dict(
                    text=f"₹{total:,.0f}", x=0.5, y=0.5,
                    font=dict(size=15, color="#f0f2f8", family="Syne"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            fig_bar = px.bar(
                cat_totals.sort_values("Amount", ascending=True),
                x="Amount", y="Category",
                orientation="h",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                title="Total by Category",
            )
            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                xaxis=dict(gridcolor="#1e2130", tickprefix="₹"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b2:
            pay_totals = df.groupby("Payment_Mode")["Amount"].sum().reset_index()
            fig_pay = px.bar(
                pay_totals,
                x="Payment_Mode", y="Amount",
                color="Payment_Mode",
                color_discrete_sequence=["#6c63ff", "#10b981", "#f59e0b", "#06b6d4", "#ec4899"],
                title="By Payment Mode",
            )
            fig_pay.update_layout(
                **PLOTLY_LAYOUT,
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(gridcolor="#1e2130", tickprefix="₹"),
                showlegend=False,
            )
            st.plotly_chart(fig_pay, use_container_width=True)

        # ── Monthly comparison chart ──────────────────────────────
        monthly_agg = (
            df.groupby([df["Date"].dt.to_period("M"), "Category"])["Amount"]
            .sum()
            .reset_index()
        )
        if not monthly_agg.empty:
            monthly_agg["Month"] = monthly_agg["Date"].astype(str)
            month_order = sorted(monthly_agg["Month"].unique())
            fig_monthly = px.bar(
                monthly_agg,
                x="Month", y="Amount",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                title="Monthly Spending by Category",
                barmode="stack",
                category_orders={"Month": month_order},
            )
            fig_monthly.update_layout(
                **PLOTLY_LAYOUT,
                xaxis=dict(gridcolor="rgba(0,0,0,0)", title=""),
                yaxis=dict(gridcolor="#1e2130", tickprefix="₹"),
            )
            st.plotly_chart(fig_monthly, use_container_width=True)


# ── Tab 2: Expense Table ─────────────────────────────────────
with tab2:
    st.markdown("### All Expenses")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_cats = st.multiselect("Filter by Category", CATEGORIES, default=[], key="fc")
    with fc2:
        filter_pay = st.multiselect("Filter by Payment Mode", PAYMENT_MODES, default=[], key="fp")
    with fc3:
        sort_by = st.selectbox("Sort by", ["Date ↓", "Date ↑", "Amount ↓", "Amount ↑"], key="sb")

    display_df = df.copy()
    if filter_cats:
        display_df = display_df[display_df["Category"].isin(filter_cats)]
    if filter_pay:
        display_df = display_df[display_df["Payment_Mode"].isin(filter_pay)]

    sort_col, asc = {
        "Date ↓": ("Date", False),
        "Date ↑": ("Date", True),
        "Amount ↓": ("Amount", False),
        "Amount ↑": ("Amount", True),
    }[sort_by]
    display_df = display_df.sort_values(sort_col, ascending=asc).reset_index(drop=True)

    show = display_df.copy()
    if not show.empty:
        show["Date"] = show["Date"].dt.strftime("%d %b %Y")
        show["Amount"] = show["Amount"].apply(lambda x: f"₹{x:,.2f}")
        show = show[["Date", "Description", "Category", "Payment_Mode", "Amount"]]
        show.columns = ["Date", "Description", "Category", "Payment Mode", "Amount"]

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width=110),
            "Description": st.column_config.TextColumn("Description", width=220),
            "Category": st.column_config.TextColumn("Category", width=120),
            "Payment Mode": st.column_config.TextColumn("Payment Mode", width=120),
            "Amount": st.column_config.TextColumn("Amount", width=110),
        },
    )

    footer_col1, footer_col2, footer_col3 = st.columns([3, 1, 1])
    with footer_col1:
        st.markdown(
            f"<div style='color:#6b7280;font-size:0.82rem;margin-top:8px;'>"
            f"Showing {len(show)} of {len(df)} entries · "
            f"Total: ₹{display_df['Amount'].sum():,.2f}</div>",
            unsafe_allow_html=True,
        )
    with footer_col2:
        csv_data = display_df.copy()
        csv_data["Date"] = csv_data["Date"].dt.strftime("%Y-%m-%d")
        st.download_button(
            "⬇️ Export CSV",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name="expenses_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with footer_col3:
        if not df.empty:
            del_idx = st.number_input(
                "Row # to delete", min_value=1, max_value=len(show),
                value=1, step=1, key="del_idx",
                help="Enter the 1-based row number from the table above to delete that entry."
            )

    if not df.empty:
        if st.button("🗑️ Delete Selected Row", use_container_width=False):
            # Map display row back to original df index
            original_idx = display_df.index[int(del_idx) - 1]
            st.session_state.expenses = st.session_state.expenses.drop(index=original_idx).reset_index(drop=True)
            save_expenses()
            st.success(f"Row {del_idx} deleted.")
            st.rerun()


# ── Tab 3: Forecast ──────────────────────────────────────────
with tab3:
    if not forecast_unlocked:
        pct = unique_days / DAYS_REQUIRED * 100 if DAYS_REQUIRED else 0
        fill_w = min(int(pct), 100)
        st.markdown(f"""
        <div class="lock-box">
            <div class="lock-icon">🔒</div>
            <div class="lock-title">Forecast Locked</div>
            <div class="lock-sub">
                Log expenses across <strong style="color:#a89fff">{DAYS_REQUIRED} different days</strong>
                to unlock predictive forecasting.<br>
                The model needs enough real spending data to produce reliable projections.
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{fill_w}%"></div>
            </div>
            <div class="progress-label">{unique_days} / {DAYS_REQUIRED} days logged</div>
        </div>
        """, unsafe_allow_html=True)
    elif df.empty:
        st.info("Add or import some expenses first.")
    else:
        st.markdown("### Monthly Forecast")
        st.markdown("Projected spending for the next 3 months using ML on daily debit totals. Categories are used only for breakdown.")

        forecast_result = compute_forecast_cached(
            df,
            st.session_state.inferred_monthly_income,
            st.session_state.current_balance,
            recent_monthly_spend,
        )

        if forecast_result["status"] == "insufficient_history":
            st.warning("Need at least 30 days of history for robust ML forecasting.")
        elif forecast_result["status"] == "insufficient_features":
            st.warning("Not enough usable time-series data to train the forecast model.")
        elif forecast_result["status"] == "insufficient_categories":
            st.warning("Not enough category data to build a forecast breakdown.")
        elif forecast_result["status"] == "empty":
            st.info("Add or import some expenses first.")
        else:
            months_ahead = forecast_result["months_ahead"]
            forecast_df = forecast_result["forecast_df"]
            totals = forecast_result["totals"]
            best_model_name = forecast_result["best_model_name"]
            best_mae = forecast_result["best_mae"]
            current_actual = float(this_month["Amount"].sum()) if not this_month.empty else 0.0

            f1, f2, f3 = st.columns(3)
            for col, period in zip([f1, f2, f3], months_ahead):
                total_val = float(totals[period])
                delta_vs_now = ((total_val - current_actual) / current_actual * 100) if current_actual > 0 else 0
                with col:
                    st.metric(
                        period,
                        f"₹{total_val:,.0f}",
                        f"{delta_vs_now:+.1f}% vs this month" if current_actual > 0 else None,
                    )

            fig_forecast = px.bar(
                forecast_df,
                x="Period",
                y="Amount",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                title="Forecasted Spending by Category",
                category_orders={"Period": months_ahead},
                barmode="stack",
            )
            fig_forecast.update_layout(
                **PLOTLY_LAYOUT,
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(gridcolor="#1e2130", tickprefix="₹"),
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.markdown("### Forecast Breakdown by Category")
            pivot = forecast_df.pivot_table(
                index="Category",
                columns="Period",
                values="Amount",
                aggfunc="sum",
            ).reindex(columns=months_ahead).fillna(0)
            pivot["Avg Forecast"] = pivot.mean(axis=1)
            pivot = pivot.round(2)

            fmt_pivot = pivot.copy()
            for col in fmt_pivot.columns:
                fmt_pivot[col] = fmt_pivot[col].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(fmt_pivot, use_container_width=True)

            st.markdown(
                f"<div style='color:#6b7280;font-size:0.8rem;margin-top:8px;'>"
                f"⚡ ML model used: <strong>{best_model_name}</strong>. "
                f"Validation MAE: <strong>₹{best_mae:,.2f}</strong>. "
                f"Training uses recent history with outlier clipping, trend damping, persisted financial context, and cached forecast computation. "
                f"Forecast target = daily debit total. "
                f"Category and payment mode are used only for explanation and breakdown, not for predicting total spending."
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Tab 4: Budgets ───────────────────────────────────────────
with tab4:
    st.markdown("### Monthly Budgets")
    st.markdown("Set a spending limit per category. Progress is tracked against this month's actual spending.")

    # ── Budget setter ─────────────────────────────────────────
    with st.expander("⚙️  Set / Edit Budgets", expanded=False):
        bcols = st.columns(3)
        budget_inputs = {}
        for i, cat in enumerate(CATEGORIES):
            existing = st.session_state.budgets.get(cat, 0.0)
            with bcols[i % 3]:
                color = CATEGORY_COLORS.get(cat, "#6b7280")
                budget_inputs[cat] = st.number_input(
                    f"{cat}",
                    min_value=0.0, value=float(existing), step=500.0,
                    format="%.0f", key=f"budget_{cat}",
                    help=f"Monthly budget for {cat} (₹0 = no limit)",
                )
        if st.button("💾  Save Budgets", use_container_width=False):
            st.session_state.budgets = {k: v for k, v in budget_inputs.items()}
            save_context()
            st.success("Budgets saved!")
            st.rerun()

    budgets = st.session_state.budgets

    if not any(v > 0 for v in budgets.values()):
        st.info("No budgets set yet. Use the panel above to add per-category limits.")
    else:
        # This month's spending per category
        this_month_cat = (
            this_month.groupby("Category")["Amount"].sum()
            if not this_month.empty
            else pd.Series(dtype=float)
        )

        over_budget_cats = []
        warn_cats = []

        for cat in CATEGORIES:
            limit = float(budgets.get(cat, 0.0))
            if limit <= 0:
                continue
            spent = float(this_month_cat.get(cat, 0.0))
            pct = spent / limit * 100
            fill_pct = min(pct, 100)
            remaining = limit - spent

            if pct >= 100:
                bar_class = "budget-bar-fill-over"
                status_html = f'<span class="diff-badge over">OVER by ₹{abs(remaining):,.0f}</span>'
                over_budget_cats.append(cat)
            elif pct >= 80:
                bar_class = "budget-bar-fill-warn"
                status_html = f'<span class="diff-badge" style="background:#3a2a00;color:#fbbf24;">₹{remaining:,.0f} left</span>'
                warn_cats.append(cat)
            else:
                bar_class = "budget-bar-fill-ok"
                status_html = f'<span class="diff-badge under">₹{remaining:,.0f} left</span>'

            color = CATEGORY_COLORS.get(cat, "#6b7280")
            st.markdown(f"""
            <div class="budget-row">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="color:{color};font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;">{cat}</span>
                        &nbsp;&nbsp;{status_html}
                    </div>
                    <div style="text-align:right;">
                        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#f0f2f8;">₹{spent:,.0f}</span>
                        <span style="color:#6b7280;font-size:0.85rem;"> / ₹{limit:,.0f}</span>
                    </div>
                </div>
                <div class="budget-bar-track">
                    <div class="{bar_class}" style="width:{fill_pct:.1f}%"></div>
                </div>
                <div style="color:#6b7280;font-size:0.75rem;text-align:right;">{pct:.1f}% used</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Alerts ────────────────────────────────────────────
        if over_budget_cats:
            st.error(f"🚨 Over budget this month: **{', '.join(over_budget_cats)}**")
        if warn_cats:
            st.warning(f"⚠️ Approaching limit (>80%): **{', '.join(warn_cats)}**")

        # ── Budget summary chart ──────────────────────────────
        chart_cats = [c for c in CATEGORIES if float(budgets.get(c, 0)) > 0]
        if chart_cats:
            budget_chart_df = pd.DataFrame({
                "Category": chart_cats,
                "Spent": [float(this_month_cat.get(c, 0.0)) for c in chart_cats],
                "Budget": [float(budgets[c]) for c in chart_cats],
            })
            fig_budget = go.Figure()
            fig_budget.add_trace(go.Bar(
                name="Budget",
                x=budget_chart_df["Category"],
                y=budget_chart_df["Budget"],
                marker_color="rgba(108,99,255,0.25)",
                marker_line=dict(color="#6c63ff", width=1.5),
            ))
            fig_budget.add_trace(go.Bar(
                name="Spent",
                x=budget_chart_df["Category"],
                y=budget_chart_df["Spent"],
                marker_color=[
                    "#ef4444" if budget_chart_df.loc[i, "Spent"] >= budget_chart_df.loc[i, "Budget"]
                    else "#f59e0b" if budget_chart_df.loc[i, "Spent"] >= 0.8 * budget_chart_df.loc[i, "Budget"]
                    else "#10b981"
                    for i in budget_chart_df.index
                ],
            ))
            fig_budget.update_layout(
                **PLOTLY_LAYOUT,
                barmode="overlay",
                title="Budget vs Actual (This Month)",
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                yaxis=dict(gridcolor="#1e2130", tickprefix="₹"),
            )
            st.plotly_chart(fig_budget, use_container_width=True) 