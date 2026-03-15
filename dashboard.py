# dashboard.py
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────
LOG_FILE = Path("logs/llm-calls.jsonl")

# ── Helper to safely read jsonl ───────────────────────────────────────────
def load_metrics() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    
    records = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                log_entry = json.loads(line)
                if "llm_metrics" in log_entry:
                    metrics_str = log_entry["llm_metrics"]
                    metrics = json.loads(metrics_str)
                    metrics["log_timestamp"] = log_entry.get("asctime", "unknown")
                    metrics["log_level"] = log_entry.get("levelname", "INFO")
                    records.append(metrics)
            except json.JSONDecodeError:
                continue    # silent skip for now
    return records

# ── Streamlit app ─────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM Observability Dashboard", layout="wide")

st.title("LLM Observability – Session 13")
st.markdown("Filter calls & see basic success + performance summary")

# Load data once
with st.spinner("Reading logs..."):
    metrics_list = load_metrics()

if not metrics_list:
    st.warning("No logs yet. Run some LLM calls first.")
    st.stop()

df = pd.DataFrame(metrics_list)

# Make timestamp usable
if "log_timestamp" in df.columns:
    df["log_timestamp"] = pd.to_datetime(df["log_timestamp"], errors="coerce")

# ── Filters ───────────────────────────────────────────────────────────────
st.subheader("Filters")

col1, col2, col3 = st.columns(3)

with col1:
    all_providers = ["All"] + sorted(df["provider"].dropna().unique())
    selected_provider = st.selectbox("Provider", all_providers, index=0)

with col2:
    max_latency = float(df["latency_seconds"].max()) if "latency_seconds" in df.columns else 10.0
    min_latency_filter = st.slider(
        "Minimum latency (seconds)",
        0.0, max_latency, 0.0, step=0.1
    )

with col3:
    success_only = st.checkbox("Show success only", value=False)

# Apply filters
filtered_df = df.copy()

if selected_provider != "All":
    filtered_df = filtered_df[filtered_df["provider"] == selected_provider]

if "latency_seconds" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["latency_seconds"] >= min_latency_filter]

if success_only and "success" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["success"] == True]

# ── Summary KPIs ──────────────────────────────────────────────────────────
st.subheader("Summary")

if len(filtered_df) > 0:
    col_a, col_b, col_c = st.columns(3)
    
    total_calls = len(filtered_df)
    success_rate = filtered_df["success"].mean() * 100 if "success" in filtered_df else 0
    avg_latency = filtered_df["latency_seconds"].mean() if "latency_seconds" in filtered_df else 0
    
    col_a.metric("Total calls", f"{total_calls:,}")
    col_b.metric("Success rate", f"{success_rate:.1f}%")
    col_c.metric("Avg latency", f"{avg_latency:.3f} s")
else:
    st.info("No calls match the current filters.")

# ── Table ─────────────────────────────────────────────────────────────────
important_columns = [
    "log_timestamp", "provider", "model", "success",
    "latency_seconds", "total_tokens", "temperature",
    "chain_of_thought", "session_id"
]

available_cols = [c for c in important_columns if c in filtered_df.columns]

st.subheader(f"Filtered calls ({len(filtered_df)} records)")

if len(filtered_df) > 0:
    st.dataframe(
        filtered_df[available_cols].sort_values("log_timestamp", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No data after filtering.")

# ── Latency chart ─────────────────────────────────────────────────────────
st.subheader("Latency distribution (filtered)")

if "latency_seconds" in filtered_df.columns and len(filtered_df) >= 3:
    st.bar_chart(
        filtered_df["latency_seconds"].value_counts(bins=12).sort_index(),
        x_label="Latency (seconds)",
        y_label="Count",
    )
else:
    st.info("Not enough filtered data for chart.")

# Footer
st.markdown("---")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")