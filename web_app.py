import os
from datetime import datetime

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

try:
    from crewai import Agent, Task, Crew, Process, LLM
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 15% 0%, #172554 0%, #080d19 34%, #060a12 100%); color:#e5e7eb; }
.block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 4rem; }
[data-testid="stHeader"] { background: transparent; }
.hero { padding: 26px 0 18px; }
.eyebrow { color:#8b5cf6; font-size:12px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
.hero h1 { font-size:48px; line-height:1.05; margin:8px 0; color:#f8fafc; letter-spacing:-1.5px; }
.hero p { color:#94a3b8; font-size:17px; max-width:760px; }
.panel { background:rgba(15,23,42,.78); border:1px solid rgba(148,163,184,.16); border-radius:20px; padding:22px; margin:14px 0; box-shadow:0 18px 50px rgba(0,0,0,.18); }
.panel-title { color:#f8fafc; font-size:21px; font-weight:800; margin-bottom:4px; }
.panel-sub { color:#64748b; font-size:13px; margin-bottom:18px; }
.metric { background:linear-gradient(145deg, rgba(30,41,59,.92), rgba(15,23,42,.9)); border:1px solid rgba(148,163,184,.13); border-radius:16px; padding:18px; min-height:115px; }
.metric-label { color:#94a3b8; font-size:12px; text-transform:uppercase; letter-spacing:.7px; }
.metric-value { color:#f8fafc; font-size:27px; font-weight:800; margin-top:8px; }
.metric-note { color:#64748b; font-size:12px; margin-top:4px; }
.verdict { border-radius:22px; padding:30px; text-align:center; border:1px solid rgba(255,255,255,.12); box-shadow:0 20px 60px rgba(0,0,0,.22); }
.verdict-buy { background:linear-gradient(145deg,#052e1b,#0f172a); }
.verdict-hold { background:linear-gradient(145deg,#3b2f06,#0f172a); }
.verdict-sell { background:linear-gradient(145deg,#3b0a0a,#0f172a); }
.verdict-label { color:#cbd5e1; font-size:12px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; }
.verdict-value { font-size:54px; font-weight:900; margin:4px 0; }
.buy-text { color:#4ade80; } .hold-text { color:#facc15; } .sell-text { color:#f87171; }
.badge { display:inline-block; padding:6px 10px; border-radius:999px; font-size:11px; font-weight:800; }
.badge-green { background:#052e1b; color:#4ade80; border:1px solid #166534; }
.badge-yellow { background:#3b2f06; color:#facc15; border:1px solid #854d0e; }
.badge-red { background:#3b0a0a; color:#f87171; border:1px solid #991b1b; }
.badge-blue { background:#172554; color:#93c5fd; border:1px solid #1d4ed8; }
.flow { display:flex; align-items:center; justify-content:center; gap:8px; flex-wrap:wrap; margin:16px 0 8px; }
.flow-box { background:#111827; border:1px solid #334155; border-radius:12px; padding:12px 15px; text-align:center; color:#e2e8f0; font-size:12px; font-weight:700; }
.flow-arrow { color:#8b5cf6; font-size:18px; }
.small { color:#94a3b8; font-size:13px; line-height:1.55; }
div.stButton > button { border-radius:12px; height:46px; font-weight:800; }
[data-testid="stDownloadButton"] button { border-radius:12px; font-weight:700; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ML CONSTANTS
# ============================================================
FEATURES = ["MA5_Ratio", "Momentum5", "Volatility5", "RSI14", "Return1"]
CLASS_ORDER = ["SELL", "HOLD", "BUY"]
HORIZONS = {"5 Trading Days": 5, "1 Month": 21, "1 Quarter": 63}


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    rsi[(loss == 0) & (gain > 0)] = 100
    return rsi


def download_history(ticker, period="5y"):
    data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty or "Close" not in data.columns:
        raise ValueError("No usable market data was returned.")
    return data.dropna(subset=["Close"]).copy()


def build_features(ticker, horizon):
    history = download_history(ticker, "5y")
    close = history["Close"].astype(float)
    history["MA5"] = close.rolling(5).mean()
    history["MA5_Ratio"] = close / history["MA5"] - 1
    history["Momentum5"] = close.pct_change(5)
    history["Return1"] = close.pct_change()
    history["Volatility5"] = history["Return1"].rolling(5).std()
    history["RSI14"] = calculate_rsi(close, 14)
    history["Future_Return"] = close.shift(-horizon) / close - 1

    # Binary target for Logistic Regression.
    history["UpDown"] = np.where(
        history["Future_Return"].notna(),
        (history["Future_Return"] > 0).astype(int),
        np.nan,
    )

    # Three-class target for Decision Tree / Random Forest.
    # +/-1% is used as a neutral zone to avoid treating tiny moves as BUY/SELL.
    history["Signal"] = np.select(
        [history["Future_Return"] > 0.01, history["Future_Return"] < -0.01],
        ["BUY", "SELL"],
        default="HOLD",
    )
    history.loc[history["Future_Return"].isna(), "Signal"] = np.nan

    model = history[FEATURES + ["Future_Return", "UpDown", "Signal"]].replace([np.inf, -np.inf], np.nan).dropna()
    latest = history[FEATURES].replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]]
    return history, model, latest


def chronological_split(X, y, horizon=5):
    """Chronological split with a purge gap to stop training labels crossing into test time.

    Each label uses a future return over `horizon` sessions. The final `horizon`
    training rows are removed so their label windows cannot overlap the test period.
    """
    boundary = int(len(X) * 0.8)
    train_end = boundary - int(horizon)
    if train_end < 50 or len(X) - boundary < 20:
        raise ValueError("Not enough observations for a chronological split with a forward-label purge gap.")
    return (
        X.iloc[:train_end], X.iloc[boundary:],
        y.iloc[:train_end], y.iloc[boundary:],
        boundary,
    )


def classification_metrics(y_true, y_pred, average="binary"):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def baseline_accuracy(y_test, multiclass=False):
    majority = y_test.value_counts().idxmax()
    return accuracy_score(y_test, [majority] * len(y_test))


def train_logistic(ticker, horizon):
    _, model, latest = build_features(ticker, horizon)
    X = model[FEATURES]
    y = model["UpDown"].astype(int)
    X_train, X_test, y_train, y_test, split = chronological_split(X, y, horizon)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = classification_metrics(y_test, pred)
    baseline = baseline_accuracy(y_test)
    latest_pred = int(pipe.predict(latest[FEATURES])[0])
    probs = pipe.predict_proba(latest[FEATURES])[0]
    probability = float(probs[latest_pred])
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    return {
        "model": "Logistic Regression",
        "pipeline": pipe,
        "prediction": "UP" if latest_pred else "DOWN",
        "probability": probability,
        "metrics": metrics,
        "baseline": baseline,
        "cm": cm,
        "class_report": classification_report(y_test, pred, labels=[0, 1], target_names=["DOWN", "UP"], output_dict=True, zero_division=0),
        "classes": ["DOWN", "UP"],
        "train_size": len(X_train), "test_size": len(X_test), "split": split,
    }


def tune_tree(X_train, y_train, horizon):
    cv = TimeSeriesSplit(n_splits=3, gap=horizon)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"max_depth": [3, 5, 7, None], "min_samples_leaf": [2, 4, 8]},
        scoring="f1_macro", cv=cv, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def tune_forest(X_train, y_train, horizon):
    cv = TimeSeriesSplit(n_splits=3, gap=horizon)
    search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
        {
            "n_estimators": [150, 250],
            "max_depth": [5, 8, None],
            "min_samples_leaf": [2, 4],
        },
        scoring="f1_macro", cv=cv, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def train_multiclass_model(ticker, horizon, kind):
    _, model, latest = build_features(ticker, horizon)
    X = model[FEATURES]
    y = model["Signal"]
    X_train, X_test, y_train, y_test, split = chronological_split(X, y, horizon)

    if kind == "Decision Tree":
        search = tune_tree(X_train, y_train, horizon)
    else:
        search = tune_forest(X_train, y_train, horizon)

    best = search.best_estimator_
    pred = best.predict(X_test)
    metrics = classification_metrics(y_test, pred, average="macro")
    baseline = baseline_accuracy(y_test, multiclass=True)
    latest_pred = str(best.predict(latest[FEATURES])[0])
    probs = best.predict_proba(latest[FEATURES])[0]
    probability = float(max(probs))
    labels = list(best.classes_)
    cm = confusion_matrix(y_test, pred, labels=CLASS_ORDER)

    result = {
        "model": kind,
        "estimator": best,
        "prediction": latest_pred,
        "probability": probability,
        "metrics": metrics,
        "baseline": baseline,
        "cm": cm,
        "class_report": classification_report(y_test, pred, labels=CLASS_ORDER, target_names=CLASS_ORDER, output_dict=True, zero_division=0),
        "classes": labels,
        "best_params": search.best_params_,
        "train_size": len(X_train), "test_size": len(X_test), "split": split,
    }
    if kind == "Random Forest":
        result["feature_importance"] = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": best.feature_importances_,
        }).sort_values("Importance", ascending=False)
    return result


@st.cache_data(show_spinner=False)
def run_all_models(ticker, horizon):
    return {
        "logistic": train_logistic(ticker, horizon),
        "tree": train_multiclass_model(ticker, horizon, "Decision Tree"),
        "forest": train_multiclass_model(ticker, horizon, "Random Forest"),
    }


def get_current_data(ticker):
    history = download_history(ticker, "3mo")
    close = history["Close"].astype(float)
    history["MA5"] = close.rolling(5).mean()
    history["Return1"] = close.pct_change()
    history["Momentum5"] = close.pct_change(5)
    history["Volatility5"] = history["Return1"].rolling(5).std()
    history["RSI14"] = calculate_rsi(close, 14)
    price = float(close.iloc[-1])
    ma5 = float(history["MA5"].iloc[-1])
    info = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        pass
    return {
        "history": history,
        "price": price,
        "ma5": ma5,
        "ma5_ratio": price / ma5 - 1,
        "momentum": float(history["Momentum5"].iloc[-1]),
        "volatility": float(history["Volatility5"].iloc[-1] * 100),
        "rsi": float(history["RSI14"].iloc[-1]),
        "company": info.get("longName") or info.get("shortName") or ticker,
        "exchange": info.get("exchange", ""),
        "currency": info.get("currency", "USD"),
        "date": history.index[-1].strftime("%b %d, %Y"),
    }


def final_recommendation(results):
    # Combine only models with the same three-class BUY/HOLD/SELL target.
    tree_vote = results["tree"]["prediction"]
    forest_vote = results["forest"]["prediction"]
    return (tree_vote, True) if tree_vote == forest_vote else ("HOLD", False)


def tree_path_summary(estimator, latest_row):
    """Summarize the actual decision path for the current observation."""
    try:
        node_indicator = estimator.decision_path(latest_row)
        leaf_id = int(estimator.apply(latest_row)[0])
        tree = estimator.tree_
        feature_names = list(latest_row.columns)
        node_ids = node_indicator.indices[
            node_indicator.indptr[0]:node_indicator.indptr[1]
        ]
        conditions = []
        for node_id in node_ids:
            if node_id == leaf_id:
                continue
            feature_index = tree.feature[node_id]
            if feature_index < 0:
                continue
            feature = feature_names[feature_index]
            threshold = float(tree.threshold[node_id])
            value = float(latest_row.iloc[0, feature_index])
            operator = "<=" if value <= threshold else ">"
            conditions.append(f"{feature} {operator} {threshold:.4f}")
        if not conditions:
            return "The fitted tree's current decision path contains no readable split conditions."
        return "The fitted Decision Tree reached its current prediction through: " + " → ".join(conditions[:4]) + "."
    except Exception:
        return "The current tree path could not be summarized."


def baseline_summary(results):
    rows = []
    for key in ["logistic", "tree", "forest"]:
        r = results[key]
        accuracy = float(r["metrics"]["accuracy"])
        baseline = float(r["baseline"])
        delta = accuracy - baseline
        rows.append({
            "Model": r["model"],
            "Test accuracy": accuracy,
            "Majority baseline": baseline,
            "Difference": delta,
            "Result": "Above baseline" if delta > 0 else "Matches baseline" if delta == 0 else "Below baseline",
        })
    return pd.DataFrame(rows)


def model_row(result):
    m = result["metrics"]
    return {
        "Model": result["model"],
        "Prediction": result["prediction"],
        "Target": "UP / DOWN" if result["model"] == "Logistic Regression" else "BUY / HOLD / SELL",
        "Predicted class probability*": result["probability"],
        "Accuracy": m["accuracy"],
        "Balanced accuracy": m["balanced_accuracy"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1": m["f1"],
        "Baseline": result["baseline"],
    }


def generate_ai_analysis(current, results, horizon_label, recommendation, models_agree):
    """Short explanation layer; model outputs remain the source of truth."""
    tree = results["tree"]["prediction"]
    forest = results["forest"]["prediction"]
    logistic = results["logistic"]["prediction"]
    signal_note = (f"Decision Tree and Random Forest agree on {tree}." if models_agree else
                   f"Decision Tree predicts {tree}, while Random Forest predicts {forest}; the three-class signal is inconclusive and displayed as HOLD.")
    if not CREWAI_AVAILABLE:
        return (f"**What the models predict:** Logistic Regression predicts **{logistic}** over {horizon_label.lower()}. {signal_note}\n\n"
                f"**Why:** The models use MA5 ratio, five-session momentum, five-session volatility, RSI and daily return. These describe recent price behaviour; they do not establish causation.\n\n"
                f"**What could be wrong:** Market news, earnings, gaps and regime changes are not captured reliably by these indicators.\n\n"
                f"**How reliable they have been:** Compare balanced accuracy and F1 with each model's majority-class baseline in Model evaluation. Historical test metrics do not guarantee future performance.")
    try:
        key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not key:
            return "Add `GROQ_API_KEY` to Streamlit Secrets to enable the AI explanation. The quantitative summary remains available above."
        os.environ["GROQ_API_KEY"] = key
        llm = LLM(model="groq/openai/gpt-oss-20b", temperature=0.2)
        agent = Agent(role="Financial data explainer", goal="Explain model outputs accurately and concisely without overstating predictive power.",
                      backstory="You explain an educational ML experiment, not provide personalized investment advice.",
                      llm=llm, verbose=False, allow_delegation=False)
        task = Task(description=f"""Write a concise report with exactly four headings: What the models predict; Why; What could make this wrong; How reliable the models have been.
Use only these facts: {current['company']} ({current['exchange']}); price {current['price']:.2f} {current['currency']}; horizon {horizon_label}; data through {current['date']}.
Market: price vs MA5 {current['ma5_ratio']:+.2%}; momentum {current['momentum']:+.2%}; 5-session volatility {current['volatility']:.2f}%; RSI {current['rsi']:.1f}.
Logistic predicts {logistic} on a binary target, estimated class probability {results['logistic']['probability']:.1%}; Decision Tree predicts {tree} with {results['tree']['probability']:.1%}; Random Forest predicts {forest} with {results['forest']['probability']:.1%}. Signal: {recommendation}; tree/forest agree: {models_agree}.
Evaluation format is accuracy / balanced accuracy / macro F1 / majority baseline. LR {results['logistic']['metrics']['accuracy']:.1%}/{results['logistic']['metrics']['balanced_accuracy']:.1%}/{results['logistic']['metrics']['f1']:.1%}/{results['logistic']['baseline']:.1%}; DT {results['tree']['metrics']['accuracy']:.1%}/{results['tree']['metrics']['balanced_accuracy']:.1%}/{results['tree']['metrics']['f1']:.1%}/{results['tree']['baseline']:.1%}; RF {results['forest']['metrics']['accuracy']:.1%}/{results['forest']['metrics']['balanced_accuracy']:.1%}/{results['forest']['metrics']['f1']:.1%}/{results['forest']['baseline']:.1%}.
Do not rank models without naming the metric. If accuracy is below baseline, say so. Raw class probabilities are not calibrated future-price probabilities. Do not invent facts or give investment advice.""",
                    expected_output="A concise four-part explanation consistent with supplied metrics.", agent=agent)
        return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff().raw
    except Exception as exc:
        return f"AI explanation unavailable ({exc}). The model outputs and measured evaluation remain available above."

def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def show_confusion_matrix(matrix, labels, title):
    # Compact, readable table rather than a giant default Matplotlib plot.
    frame = pd.DataFrame(matrix, index=[f"Actual {x}" for x in labels],
                         columns=[f"Predicted {x}" for x in labels])
    st.markdown(f"**{title}**")
    st.dataframe(frame, use_container_width=True)



# ============================================================
# GLOBAL MARKET PULSE
# ============================================================
GLOBAL_MARKETS = [
    {"region": "United States", "exchange": "NYSE / Nasdaq", "ticker": "^GSPC", "lat": 40.7128, "lon": -74.0060},
    {"region": "United Kingdom", "exchange": "London Stock Exchange", "ticker": "^FTSE", "lat": 51.5072, "lon": -0.1276},
    {"region": "Germany", "exchange": "Frankfurt", "ticker": "^GDAXI", "lat": 50.1109, "lon": 8.6821},
    {"region": "India", "exchange": "NSE", "ticker": "^NSEI", "lat": 19.0760, "lon": 72.8777},
    {"region": "Hong Kong", "exchange": "Hong Kong Exchange", "ticker": "^HSI", "lat": 22.3193, "lon": 114.1694},
    {"region": "Japan", "exchange": "Tokyo Stock Exchange", "ticker": "^N225", "lat": 35.6762, "lon": 139.6503},
]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_global_market_pulse():
    """Fetch recent index performance and relative trading activity for tracked markets."""
    rows = []
    for market in GLOBAL_MARKETS:
        try:
            hist = yf.Ticker(market["ticker"]).history(period="3mo", auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist.columns or len(hist) < 6:
                continue
            close = hist["Close"].dropna()
            if len(close) < 6:
                continue
            latest_return = float(close.iloc[-1] / close.iloc[-2] - 1)
            week_return = float(close.iloc[-1] / close.iloc[-6] - 1)
            volume_value = None
            volume_ratio = None
            if "Volume" in hist.columns:
                volume = hist["Volume"].dropna()
                if len(volume) >= 6 and float(volume.tail(20).mean()) > 0:
                    volume_value = int(volume.iloc[-1])
                    volume_ratio = float(volume.iloc[-1] / volume.tail(20).mean())
            rows.append({
                **market,
                "last_close": float(close.iloc[-1]),
                "daily_return": latest_return,
                "weekly_return": week_return,
                "volume": volume_value,
                "volume_ratio": volume_ratio,
                "as_of": str(close.index[-1].date()),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# ============================================================
# UI
# ============================================================
st.markdown('''<div class="hero"><div class="eyebrow">Machine Learning · Explainable AI · Market Intelligence</div><h1>AI Financial Analyst</h1><p>Train and compare machine-learning models on historical market data, understand what drives their predictions, and generate an AI-assisted financial interpretation.</p></div>''', unsafe_allow_html=True)

c1, c2, c3 = st.columns([3.5, 1.5, 1])
with c1:
    company_input = st.text_input("Company or ticker", value=st.session_state.get("selected_ticker_input", ""), placeholder="e.g. NVIDIA, Apple, MSFT, RELIANCE.NS", label_visibility="collapsed")
with c2:
    horizon_options = list(HORIZONS)
    saved_horizon = st.session_state.get("selected_horizon", horizon_options[0])
    horizon_label = st.selectbox("Prediction horizon", horizon_options, index=horizon_options.index(saved_horizon) if saved_horizon in horizon_options else 0, label_visibility="collapsed")
with c3:
    analyze = st.button("Analyze →", type="primary", use_container_width=True)

if analyze:
    if not company_input.strip():
        st.warning("Enter a company name or ticker first.")
        st.stop()

    with st.spinner("Finding company and evaluating the models..."):
        try:
            ticker_input = company_input.strip().upper()
            ticker = ticker_input
            try:
                test = yf.Ticker(ticker).history(period="5d")
                if test.empty:
                    raise ValueError
            except Exception:
                search = yf.Search(company_input.strip(), max_results=10)
                quotes = getattr(search, "quotes", []) or []
                if not quotes or not quotes[0].get("symbol"):
                    raise ValueError("Company not found. Try its exchange ticker, e.g. RELIANCE.NS.")
                ticker = quotes[0]["symbol"]
            horizon = HORIZONS[horizon_label]
            current = get_current_data(ticker)
            results = run_all_models(ticker, horizon)
            st.session_state["analysis_payload"] = {"ticker": ticker, "current": current, "results": results, "horizon": horizon, "horizon_label": horizon_label, "company_input": company_input.strip()}
            st.session_state["selected_ticker_input"] = company_input.strip()
            st.session_state["selected_horizon"] = horizon_label
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()
elif st.session_state.get("analysis_payload"):
    saved = st.session_state["analysis_payload"]
    ticker = saved["ticker"]
    current = saved["current"]
    results = saved["results"]
    horizon = saved["horizon"]
    horizon_label = saved["horizon_label"]
    company_input = saved["company_input"]

if st.session_state.get("analysis_payload"):
    saved = st.session_state["analysis_payload"]
    ticker = saved["ticker"]
    current = saved["current"]
    results = saved["results"]
    horizon = saved["horizon"]
    horizon_label = saved["horizon_label"]
    recommendation, models_agree = final_recommendation(results)
    verdict_class = {"BUY":"verdict-buy buy-text", "HOLD":"verdict-hold hold-text", "SELL":"verdict-sell sell-text"}[recommendation]
    votes = [results["tree"]["prediction"], results["forest"]["prediction"]]
    accuracy_below_baseline = any(results[k]["metrics"]["accuracy"] < results[k]["baseline"] for k in ["logistic", "tree", "forest"])

    st.markdown(f'''<div class="panel"><div class="panel-title">{current['company']}</div><div class="panel-sub">{ticker} · {current['exchange']} · Data through {current['date']} · Forecast horizon: {horizon_label}</div></div>''', unsafe_allow_html=True)
    st.markdown(f'''<div class="verdict {verdict_class}"><div class="verdict-label">Experimental model signal · {horizon_label}</div><div class="verdict-value">{recommendation}</div><div class="small">{('Decision Tree and Random Forest agree.' if models_agree else f"Models disagree: Decision Tree = {votes[0]}, Random Forest = {votes[1]}. HOLD is a cautious fallback, not a confident forecast.")} Logistic Regression separately predicts {results['logistic']['prediction']} on an UP/DOWN target.</div></div>''', unsafe_allow_html=True)
    st.caption("Educational experiment only — not financial advice. Historical test performance is limited and does not establish that this signal will be accurate.")
    if accuracy_below_baseline:
        st.warning("At least one model performs below its majority-class baseline. The Machine Learning tab shows each model's exact accuracy, baseline and difference.")

    # Immediate explanation of disagreement between the ML signal and market context.
    ma_deviation = current["ma5_ratio"]
    if recommendation == "BUY" and ma_deviation < 0:
        st.info(f"**Why BUY when the market looks bearish?** Decision Tree and Random Forest classify the current feature combination as BUY, but the price is {ma_deviation:.2%} below MA5. That is conflicting evidence, not something the app should explain away. The fitted tree's actual decision path and test performance are available in Machine Learning.")
    elif recommendation == "SELL" and ma_deviation > 0:
        st.info(f"**Why SELL when price is above MA5?** The tree and forest classify the current feature combination as SELL, while price is {ma_deviation:+.2%} above MA5. Treat this as a conflict between the model signal and market context; inspect the evaluation before drawing conclusions.")
    elif recommendation == "BUY":
        st.info("**Why BUY?** Decision Tree and Random Forest agree on the BUY class. This is the models' classification of the engineered features, not proof of future gains; inspect the historical test results and tree path below.")
    elif recommendation == "SELL":
        st.info("**Why SELL?** Decision Tree and Random Forest agree on the SELL class. This is an experimental classification, not proof of future losses; inspect the historical test results and tree path below.")
    else:
        st.info("**Why HOLD?** The three-class models disagree or classify the current features as HOLD. HOLD is a cautious display state, not a prediction that the price will remain unchanged.")

    # Four compact market cards, each with its interpretation (no repeated context table).
    cols = st.columns(4)
    ma_note = "Price is above its 5-session average." if current["ma5_ratio"] > 0 else "Price is below its 5-session average." if current["ma5_ratio"] < 0 else "Price is at its 5-session average."
    rsi_note = "Near neutral; not strongly overbought or oversold." if 40 <= current["rsi"] <= 60 else "Relatively overbought." if current["rsi"] > 70 else "Relatively oversold." if current["rsi"] < 30 else "Momentum is leaning away from neutral."
    labels = ["Current price", "Vs. MA5", "RSI (14)", "5-session volatility"]
    values = [f"{current['price']:.2f} {current['currency']}", f"{current['ma5_ratio']:+.2%}", f"{current['rsi']:.1f}", f"{current['volatility']:.2f}%"]
    notes = ["Latest adjusted close", ma_note, rsi_note, "Recent return variability; not a direction forecast"]
    for col, label, value, note in zip(cols, labels, values, notes):
        with col:
            metric_card(label, value, note)

    # The summary describes observable evidence and model outputs without inventing a reason.
    trend_history = current["history"]["Close"].tail(20).dropna()
    recent_trend_pct = float(trend_history.iloc[-1] / trend_history.iloc[0] - 1) if len(trend_history) >= 2 else 0.0
    trend_text = "has risen" if recent_trend_pct > 0.01 else "has fallen" if recent_trend_pct < -0.01 else "has been broadly flat"
    ma_conflict = current["ma5_ratio"] < 0 and recommendation == "BUY"
    if recommendation == "BUY":
        signal_meaning = "The Decision Tree and Random Forest produce a BUY class." if models_agree else "The tree and forest disagree, so HOLD is shown as the cautious fallback rather than a strong BUY."
    elif recommendation == "SELL":
        signal_meaning = "The Decision Tree and Random Forest produce a SELL class."
    else:
        signal_meaning = "The current three-class signal is HOLD; this is not proof that the price will remain stable."
    if ma_conflict:
        market_conflict_text = "This conflicts with the market snapshot: price is below its five-session average."
    elif current["ma5_ratio"] > 0 and recommendation == "SELL":
        market_conflict_text = "This conflicts with the market snapshot: price is above its five-session average."
    else:
        market_conflict_text = "The moving-average position is context, not a vote in the ML recommendation."
    st.markdown("#### What this result means")
    st.write(
        f"Over the last 20 available sessions, the price {trend_text} ({recent_trend_pct:+.2%} from first to last close). "
        f"{signal_meaning} {market_conflict_text} The models use MA5 ratio, momentum, volatility, RSI and daily return; "
        "these inputs describe recent behaviour, but they do not prove causation or guarantee future performance."
    )
    if ma_conflict:
        st.info("**Why BUY despite the bearish market context?** The tree and forest classify the engineered feature combination as BUY; the tree's actual decision path is available in Machine Learning → model evaluation. The falling trend and negative MA5 deviation are genuine counter-signals. Feature importance does not prove which feature caused the BUY result.")
    elif recommendation == "BUY" and recent_trend_pct < -0.01:
        st.info("**Conflicting evidence:** the model signal is BUY, but the recent 20-session price trend is negative. Treat the signal as experimental and inspect the model evaluation before drawing conclusions.")

    overview_tab, market_tab, ml_tab, ai_tab, chat_tab, globe_tab = st.tabs(["Overview", "Market", "Machine Learning", "AI Analyst", "Chatbot", "Global Pulse"])
    with overview_tab:
        st.markdown('<div class="panel"><div class="panel-title">Recent price trend</div><div class="panel-sub">Last 60 trading sessions with the 5-session moving average.</div></div>', unsafe_allow_html=True)
        hist = current["history"].tail(60).copy()
        hist["MA5"] = hist["Close"].rolling(5).mean()
        chart_data = hist[["Close", "MA5"]].rename(columns={"Close":"Price"}).reset_index()
        date_col = chart_data.columns[0]
        long = chart_data.melt(id_vars=date_col, var_name="Series", value_name="Value")
        price_chart = alt.Chart(long).mark_line().encode(
            x=alt.X(f"{date_col}:T", title="Trading date"),
            y=alt.Y("Value:Q", title=f"Price ({current['currency']})", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", title="Series"),
            tooltip=[alt.Tooltip(f"{date_col}:T", title="Date"), "Series:N", alt.Tooltip("Value:Q", format=".2f")]
        ).properties(height=350)
        st.altair_chart(price_chart, use_container_width=True)
        st.markdown("**What the market indicators say**")
        st.write(f"{ma_note} RSI is {rsi_note.lower()} Five-session momentum is **{current['momentum']:+.2%}**. These indicators describe recent behaviour; they do not independently confirm a future move.")

    with market_tab:
        st.markdown('<div class="panel"><div class="panel-title">Market context</div><div class="panel-sub">Interpretations are placed beside the measures they describe.</div></div>', unsafe_allow_html=True)
        st.write(f"**Price vs. MA5:** {ma_note}")
        st.write(f"**RSI:** {rsi_note}")
        st.write(f"**Momentum (5 sessions):** {current['momentum']:+.2%}")
        st.write(f"**Volatility (5 sessions):** {current['volatility']:.2f}% — recent variability, not a prediction of direction.")
        st.caption(f"Last available market-data date: {current['date']}. Chart currency: {current['currency']}.")

    comparison = pd.DataFrame([model_row(results["logistic"]), model_row(results["tree"]), model_row(results["forest"])])
    comparison_display = comparison.copy()
    pct_cols = ["Predicted class probability*", "Accuracy", "Balanced accuracy", "Precision", "Recall", "F1", "Baseline"]
    for col in pct_cols:
        comparison_display[col] = comparison_display[col].map(lambda x: f"{x:.1%}")
    with ml_tab:
        st.markdown(
            '<div class="panel"><div class="panel-title">Machine Learning predictions</div>'
            '<div class="panel-sub">Three models look at recent stock data in different ways.</div></div>',
            unsafe_allow_html=True,
        )
        model_cards = [
            ("Logistic Regression", results["logistic"], "Predicts whether the price may go UP or DOWN."),
            ("Decision Tree", results["tree"], "Uses a set of decision rules to suggest BUY, HOLD or SELL."),
            ("Random Forest", results["forest"], "Combines multiple decision trees to suggest BUY, HOLD or SELL."),
        ]
        cols = st.columns(3)
        for col, (name, result, explanation) in zip(cols, model_cards):
            with col:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown(f"**{name}**")
                st.markdown(f"### {result['prediction']}")
                st.caption(explanation)
                st.markdown('</div>', unsafe_allow_html=True)
        st.warning(
            "These predictions are experimental. In the current test, all three models scored below "
            "a simple majority-class baseline, so their predictions should not be treated as reliable forecasts."
        )
        with st.expander("How were the models tested? (optional)"):
            st.write(
                f"The models use five technical indicators calculated from historical data and a chronological "
                f"train/test split, with a {horizon}-trading-session gap to reduce data leakage. "
                "Logistic Regression predicts UP/DOWN; Decision Tree and Random Forest predict BUY/HOLD/SELL."
            )

    with ai_tab:
        st.markdown('<div class="panel"><div class="panel-title">AI explanation</div><div class="panel-sub">A concise interpretation of the predictions, uncertainty, and measured reliability.</div></div>', unsafe_allow_html=True)
        with st.spinner("Preparing the concise explanation..."):
            analysis = generate_ai_analysis(current, results, horizon_label, recommendation, models_agree)
        st.markdown(analysis)

    with chat_tab:
        st.markdown(
            '<div class="panel"><div class="panel-title">Ask your AI Financial Analyst</div>'
            '<div class="panel-sub">Ask questions about the stock, chart, indicators, or model predictions shown in this analysis.</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Answers use the current analysis as context. They are educational explanations, not investment advice.")

        # Keep separate chat history for each selected ticker and forecast horizon.
        chat_key = f"finance_chat_{ticker}_{horizon_label}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for message in st.session_state[chat_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask about this stock or its predictions…", key=f"chat_input_{ticker}_{horizon_label}")
        if question:
            st.session_state[chat_key].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            market_context = (
                f"Company: {current['company']} ({ticker}); exchange: {current['exchange']}; "
                f"price: {current['price']:.2f} {current['currency']}; data date: {current['date']}; "
                f"forecast horizon: {horizon_label}; price vs MA5: {current['ma5_ratio']:+.2%}; "
                f"5-session momentum: {current['momentum']:+.2%}; 5-session volatility: {current['volatility']:.2f}%; "
                f"RSI: {current['rsi']:.1f}. Logistic Regression predicts {results['logistic']['prediction']} (UP/DOWN); "
                f"Decision Tree predicts {results['tree']['prediction']} (BUY/HOLD/SELL); "
                f"Random Forest predicts {results['forest']['prediction']} (BUY/HOLD/SELL); "
                f"overall cautious signal: {recommendation}; tree/forest agree: {models_agree}. "
                f"Test accuracy vs majority baseline: Logistic Regression {results['logistic']['metrics']['accuracy']:.1%} vs {results['logistic']['baseline']:.1%}; "
                f"Decision Tree {results['tree']['metrics']['accuracy']:.1%} vs {results['tree']['baseline']:.1%}; "
                f"Random Forest {results['forest']['metrics']['accuracy']:.1%} vs {results['forest']['baseline']:.1%}."
            )
            history_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in st.session_state[chat_key][-8:]
            )
            try:
                if not CREWAI_AVAILABLE:
                    raise RuntimeError("CrewAI is not installed in this environment.")
                api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
                if not api_key:
                    raise RuntimeError("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")
                os.environ["GROQ_API_KEY"] = api_key
                chat_llm = LLM(model="groq/openai/gpt-oss-20b", temperature=0.2)
                chat_agent = Agent(
                    role="Financial analysis explainer",
                    goal="Answer the user's questions clearly using only the supplied stock analysis context.",
                    backstory="You are a careful tutor explaining an educational stock-analysis app in plain English. You never claim stock predictions are certain.",
                    llm=chat_llm,
                    verbose=False,
                    allow_delegation=False,
                )
                chat_task = Task(
                    description=f"""Answer the latest user question in simple, concise language. Use the supplied context and recent chat only. Do not invent news, prices, or facts. If the context does not answer the question, say so. Explain technical terms simply. Do not give personalized investment advice, and clearly state that the models are experimental and scored below their majority-class baselines.

CURRENT ANALYSIS:
{market_context}

RECENT CHAT:
{history_text}

LATEST QUESTION:
{question}""",
                    expected_output="A helpful plain-English answer grounded in the provided analysis.",
                    agent=chat_agent,
                )
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = Crew(agents=[chat_agent], tasks=[chat_task], process=Process.sequential, verbose=False).kickoff().raw
                    st.markdown(answer)
                st.session_state[chat_key].append({"role": "assistant", "content": str(answer)})
            except Exception as exc:
                error_message = f"I couldn't answer just now: {exc}"
                with st.chat_message("assistant"):
                    st.warning(error_message)
                st.session_state[chat_key].append({"role": "assistant", "content": error_message})

    with globe_tab:
        st.markdown(
            '<div class="panel"><div class="panel-title">Global Market Pulse</div>'
            '<div class="panel-sub">Explore recent index movements across selected global markets.</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Each marker represents a market index, not every individual trade. "
            "Colour shows the latest daily index return; marker size reflects the absolute daily move. "
            "This is a market snapshot, not a measure of total trading activity or financial advice."
        )
        with st.spinner("Loading recent global index data..."):
            pulse = fetch_global_market_pulse()
        if pulse.empty:
            st.warning("Global market data could not be loaded right now. Please try again later.")
        else:
            pulse["Daily change"] = pulse["daily_return"].map(lambda v: f"{v:+.2%}")
            pulse["Weekly change"] = pulse["weekly_return"].map(lambda v: f"{v:+.2%}")
            pulse["Marker size"] = pulse["daily_return"].abs().clip(lower=0.001) * 100
            fig = px.scatter_geo(
                pulse,
                lat="lat",
                lon="lon",
                color="daily_return",
                size="Marker size",
                size_max=24,
                hover_name="region",
                hover_data={
                    "exchange": True,
                    "ticker": True,
                    "Daily change": True,
                    "Weekly change": True,
                    "as_of": True,
                    "lat": False,
                    "lon": False,
                    "daily_return": False,
                    "weekly_return": False,
                    "Marker size": False,
                },
                projection="orthographic",
                color_continuous_scale=["#f87171", "#64748b", "#4ade80"],
                color_continuous_midpoint=0,
                title="Market momentum around the world",
            )
            fig.update_geos(
                showland=True, landcolor="#111827",
                showocean=True, oceancolor="#080d19",
                showcountries=True, countrycolor="#334155",
                showcoastlines=True, coastlinecolor="#475569",
                showframe=False, bgcolor="rgba(0,0,0,0)",
                projection_rotation=dict(lon=10, lat=15, roll=0),
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color="#f8fafc",
                margin=dict(l=0, r=0, t=55, b=0),
                height=520,
                coloraxis_colorbar=dict(title="Daily return", tickformat=".1%"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("#### Regional snapshot")
            snapshot = pulse[["region", "exchange", "last_close", "Daily change", "Weekly change", "as_of"]].copy()
            snapshot.columns = ["Market", "Exchange", "Latest index close", "Daily change", "5-session change", "Data date"]
            st.dataframe(snapshot, use_container_width=True, hide_index=True)
            st.caption(
                "Markets shown: S&P 500 (US), FTSE 100 (UK), DAX (Germany), Nifty 50 (India), "
                "Hang Seng (Hong Kong) and Nikkei 225 (Japan). Index levels are not directly comparable across markets. "
                "The globe colours and sizes describe recent index price changes; they do not indicate that a region is "
                "'flourishing' overall or that its trades are more profitable."
            )

    report = f"""AI FINANCIAL ANALYST — EDUCATIONAL REPORT
{current['company']} ({ticker}) · {current['exchange']}
Data through: {current['date']} | Horizon: {horizon_label}
{'='*66}

MODEL SIGNAL: {recommendation}
{('Decision Tree and Random Forest agree.' if models_agree else f"Tree ({votes[0]}) and Forest ({votes[1]}) disagree; HOLD is a cautious fallback.")}
Logistic Regression separately predicts {results['logistic']['prediction']} (UP/DOWN target).
Reliability caveat: experimental signal only; model agreement does not establish correctness.
Recent 20-session price change: {recent_trend_pct:+.2%}.
Signal/market conflict: {"BUY conflicts with a price below MA5." if recommendation == "BUY" and current["ma5_ratio"] < 0 else "SELL conflicts with a price above MA5." if recommendation == "SELL" and current["ma5_ratio"] > 0 else "No direct MA5 conflict flagged."}

MARKET SNAPSHOT
Price: {current['price']:.2f} {current['currency']}
Vs. MA5: {current['ma5_ratio']:+.2%} — {ma_note}
RSI (14): {current['rsi']:.1f} — {rsi_note}
Momentum (5 sessions): {current['momentum']:+.2%}
Volatility (5 sessions): {current['volatility']:.2f}%

MODEL EVALUATION (Accuracy / Balanced accuracy / Macro F1 / Majority baseline)
"""
    for key in ["logistic", "tree", "forest"]:
        r = results[key]
        target = "UP/DOWN" if key == "logistic" else "BUY/HOLD/SELL"
        gap = r["metrics"]["accuracy"] - r["baseline"]
        report += f"{r['model']} [{target}]: accuracy {r['metrics']['accuracy']:.1%}; balanced accuracy {r['metrics']['balanced_accuracy']:.1%}; macro F1 {r['metrics']['f1']:.1%}; majority baseline {r['baseline']:.1%}; gap {gap:+.1%} ({'above' if gap > 0 else 'below' if gap < 0 else 'at'} baseline).\n"
    report += f"\nGenerated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}\nEducational project only; not financial advice. Raw class probabilities are not calibrated forecasts.\n"
    st.download_button("Download analysis report", report, f"{ticker}_analysis_report.txt", "text/plain")
    with st.expander("Methodology details"):
        st.markdown("""
**Data:** 5 years of daily adjusted-price history from Yahoo Finance.

**Features:** MA5 ratio, 5-session momentum, 5-session volatility, RSI(14), and daily return.

**Validation:** chronological 80/20 train/test split with a purge gap equal to the forecast horizon, so training labels cannot use prices from the held-out test period. Hyperparameter tuning uses TimeSeriesSplit with the same gap on training data only. This is a per-ticker historical experiment, not a test of generalization to unseen companies.

**Metrics:** accuracy, balanced accuracy, macro precision/recall/F1, confusion matrices, and majority-class baseline. Logistic and three-class models use different targets, so their scores are not directly comparable.

**Limitations:** some models may perform below baseline; raw class probabilities are not calibrated; feature importance is not causation. A chronological split reduces leakage from overlapping forward labels but does not remove every source of market-data or model-selection bias.
""")
    st.caption("Python · Streamlit · yfinance · scikit-learn · Altair · CrewAI/Groq | Educational use only")
