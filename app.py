import streamlit as st
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="VibraSafe Swarm Simulator", layout="wide")

st.title("🔴 VibraSafe Swarm – Factory Safety Digital Twin (Software Only)")
st.caption("Team: <Your Team Name>  |  College: <Your College Name>")

# ------------------ CONSTANTS ------------------
VEST_COUNT = 6

# ------------------ SESSION STATE INIT ------------------
if "status" not in st.session_state:
    st.session_state.status = ["OK"] * VEST_COUNT          # OK / WARN / ALERT
if "last_score" not in st.session_state:
    st.session_state.last_score = 0.0
if "active_alerts" not in st.session_state:
    st.session_state.active_alerts = 0
if "evacuated" not in st.session_state:
    st.session_state.evacuated = 0
if "score_history" not in st.session_state:
    st.session_state.score_history = []     # list of floats (overall anomaly)
if "time_step" not in st.session_state:
    st.session_state.time_step = 0

# ------------------ DATA GENERATION ------------------
def generate_data(level: str, n_samples: int = 200):
    """Synthetic vibration data for different hazard levels."""
    t = np.linspace(0, 1, n_samples)

    if level == "NORMAL":
        sig = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, n_samples)
    elif level == "WARN":
        sig = np.sin(2 * np.pi * 15 * t) + np.random.normal(0, 0.3, n_samples)
    else:  # MAJOR
        sig = 3 * np.sin(2 * np.pi * 40 * t) + np.random.normal(0, 0.6, n_samples)

    x = sig
    y = sig + np.random.normal(0, 0.1, n_samples)
    z = sig + np.random.normal(0, 0.1, n_samples)
    data = np.column_stack([x, y, z])
    return data

# ------------------ HEATMAP ------------------
def heatmap():
    xs, ys, zones, risks = [], [], [], []
    grid_x = ["Zone1", "Zone2", "Zone3"]
    grid_y = ["A", "B"]

    idx = 0
    for y in grid_y:
        for x in grid_x:
            xs.append(x)
            ys.append(y)
            if idx < VEST_COUNT:
                zones.append(f"Vest {idx+1}")
                state = st.session_state.status[idx]
                if state == "ALERT":
                    risks.append(0.9)
                elif state == "WARN":
                    risks.append(0.6)
                else:
                    risks.append(0.2)
            else:
                zones.append("EMPTY")
                risks.append(0.1)
            idx += 1

    df = pd.DataFrame({"x": xs, "y": ys, "zone": zones, "risk": risks})
    fig = px.density_heatmap(
        df,
        x="x",
        y="y",
        z="risk",
        color_continuous_scale="RdYlGn_r",
        range_color=(0, 1),
        labels={"x": "Factory Zone", "y": "Row", "risk": "Risk"},
        title="Factory Risk Heatmap (Simulated Swarm)"
    )
    fig.update_layout(height=400)
    return fig

# ------------------ CALLBACKS ------------------
def set_vest_state(i: int, state: str):
    st.session_state.status[i] = state

def run_hazard(level: str):
    data = generate_data(level)
    model = IsolationForest().fit(data)
    score = model.score_samples(data)[-1]

    st.session_state.last_score = float(score)
    st.session_state.time_step += 1
    st.session_state.score_history.append(
        {"t": st.session_state.time_step, "score": st.session_state.last_score}
    )

    # simple rule
    if level == "NORMAL":
        for i in range(VEST_COUNT):
            if st.session_state.status[i] != "ALERT":
                st.session_state.status[i] = "OK"
        st.session_state.active_alerts = 0
    elif level == "WARN":
        st.session_state.status[0] = "WARN"
        st.session_state.status[1] = "WARN"
        st.session_state.active_alerts = 2
        st.session_state.evacuated += 5
    else:  # MAJOR
        for i in range(4):
            st.session_state.status[i] = "ALERT"
        st.session_state.active_alerts = 4
        st.session_state.evacuated += 15

# ------------------ LAYOUT ------------------
st.subheader("Vest Controls (Simulated Nodes)")

cols = st.columns(3)
for i in range(VEST_COUNT):
    with cols[i % 3]:
        st.text(f"Vest {i+1}: {st.session_state.status[i]}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("OK", key=f"ok_{i}", on_click=set_vest_state, args=(i, "OK"))
        with c2:
            st.button("Warn", key=f"warn_{i}", on_click=set_vest_state, args=(i, "WARN"))
        with c3:
            st.button("Alert", key=f"alert_{i}", on_click=set_vest_state, args=(i, "ALERT"))

st.markdown("---")
st.subheader("Simulate Sensor Pattern + Run AI")

b1, b2, b3 = st.columns(3)
with b1:
    if st.button("✅ NORMAL RUN"):
        run_hazard("NORMAL")
with b2:
    if st.button("⚠️ SMALL ISSUE"):
        run_hazard("WARN")
with b3:
    if st.button("🚨 MAJOR HAZARD"):
        run_hazard("MAJOR")

# metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Latest Anomaly Score", f"{st.session_state.last_score:.3f}")
with m2:
    st.metric("Active Alerts (Vests)", st.session_state.active_alerts)
with m3:
    st.metric("Evacuated Workers (Simulated)", st.session_state.evacuated)

# charts
c1, c2 = st.columns([2, 1])
with c1:
    st.plotly_chart(heatmap(), use_container_width=True)

with c2:
    st.markdown("### Anomaly Trend")
    if st.session_state.score_history:
        df_hist = pd.DataFrame(st.session_state.score_history)
        fig_line = px.line(df_hist, x="t", y="score",
                           labels={"t": "Time Step", "score": "Anomaly Score"},
                           title="Overall Anomaly Score Over Time")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Press any simulation button to start anomaly trend.")
