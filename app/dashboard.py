"""Streamlit demo dashboard.

Run from the repo root:
    streamlit run app/dashboard.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from planner.vehicle import Vehicle
from planner.wind import WindField
from planner.noise import GroundGrid, THRESHOLD_DB
from planner.optimize import TransitionPlanner

st.set_page_config(page_title="VTOL Transition Planner", layout="wide",
                   page_icon="🛫")

MODE_COLOR = {"MC": "#D9480F", "FW": "#0B7285"}

st.title("Energy-Optimal VTOL Transition Planner")
st.caption("Given a mission and real wind at altitude: when should a VTOL "
           "switch between hover and cruise? Second objective: acoustic "
           "footprint on the ground.")

with st.sidebar:
    st.header("Mission")
    lat_a = st.number_input("Origin lat", value=12.935, format="%.4f")
    lon_a = st.number_input("Origin lon", value=77.535, format="%.4f")
    lat_b = st.number_input("Destination lat", value=12.985, format="%.4f")
    lon_b = st.number_input("Destination lon", value=77.610, format="%.4f")
    route_km = st.slider("Route length (km)", 3.0, 25.0, 8.5, 0.5)
    payload = st.slider("Payload (kg)", 0.0, 2.0, 0.5, 0.1)
    offline = st.checkbox("Offline (use cached wind)", value=False)
    st.header("Objective")
    lam = st.select_slider(
        "Energy ↔ Quiet",
        options=[0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        value=0.01)
    go_btn = st.button("Plan mission", type="primary", use_container_width=True)


@st.cache_resource(show_spinner="Fetching wind + building planner…")
def build(lat_a, lon_a, lat_b, lon_b, route_m, payload, offline):
    v = Vehicle(mass=5.0 + payload)
    wf = WindField.fetch(lat_a, lon_a, lat_b, lon_b, use_cache_only=offline)
    grid = GroundGrid(route_m)
    return v, wf, grid, TransitionPlanner(v, grid, wf, route_m)


v, wf, grid, planner = build(lat_a, lon_a, lat_b, lon_b,
                             route_km * 1000, payload, offline)

naive = planner.naive_plan()
opt = planner.solve(lam=lam)

c1, c2, c3, c4 = st.columns(4)
saved = 100 * (1 - opt.energy_wh / naive.energy_wh)
c1.metric("Mission energy", f"{opt.energy_wh:.1f} Wh",
          f"{saved:+.0f}% vs naive", delta_color="inverse")
c2.metric("Battery used", f"{100*opt.energy_wh/v.battery_wh:.0f}%")
c3.metric("Flight time", f"{opt.duration_s/60:.1f} min")
ln = grid.footprint(naive.steps)
lo = grid.footprint(opt.steps)
c4.metric(f"Area > {THRESHOLD_DB:.0f} dB",
          f"{grid.area_above_threshold_km2(lo):.2f} km²",
          f"{grid.area_above_threshold_km2(lo)-grid.area_above_threshold_km2(ln):+.2f} vs naive",
          delta_color="inverse")

left, right = st.columns([3, 2])

with left:
    st.subheader("Mission profile")
    fig = go.Figure()
    for plan, name, width, dash in ((naive, "naive", 2, "dot"),
                                    (opt, "optimized", 5, None)):
        xs = [s[0] / 1000 for s in plan.steps]
        alts = [s[1] for s in plan.steps]
        modes = [s[2] for s in plan.steps]
        for m in ("MC", "FW"):
            fig.add_trace(go.Scatter(
                x=[x if modes[i] == m else None for i, x in enumerate(xs)],
                y=[a if modes[i] == m else None for i, a in enumerate(alts)],
                mode="lines",
                line={"color": MODE_COLOR[m], "width": width,
                      "dash": dash or "solid"},
                name=f"{name} · {'hover' if m=='MC' else 'cruise'}",
                connectgaps=False))
    fig.update_layout(xaxis_title="distance (km)",
                      yaxis_title="altitude AGL (m)",
                      height=380, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ground noise footprint (optimized plan)")
    fig2 = go.Figure(go.Heatmap(
        x=grid.xs / 1000, y=grid.ys / 1000, z=lo,
        colorscale="Magma", zmin=35, zmax=float(max(ln.max(), lo.max())),
        colorbar={"title": "dB"}))
    fig2.add_contour(x=grid.xs / 1000, y=grid.ys / 1000, z=lo,
                     contours={"start": THRESHOLD_DB, "end": THRESHOLD_DB,
                               "size": 1, "coloring": "none"},
                     line={"color": "#4FC3D9", "width": 2},
                     showscale=False)
    fig2.update_layout(xaxis_title="along route (km)",
                       yaxis_title="cross-track (km)",
                       height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("Energy ↔ noise trade")
    plans, front = planner.pareto_sweep()
    fs = sorted(front, key=lambda p: p.noise_score)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=[p.noise_score for p in plans], y=[p.energy_wh for p in plans],
        mode="markers", marker={"color": "#9AA7B8", "size": 8},
        name="candidates"))
    fig3.add_trace(go.Scatter(
        x=[p.noise_score for p in fs], y=[p.energy_wh for p in fs],
        mode="lines+markers", line={"color": "#0B7285", "width": 2},
        name="Pareto front"))
    fig3.add_trace(go.Scatter(
        x=[naive.noise_score], y=[naive.energy_wh], mode="markers",
        marker={"symbol": "x", "size": 14, "color": "#D9480F"},
        name="naive"))
    fig3.add_trace(go.Scatter(
        x=[opt.noise_score], y=[opt.energy_wh], mode="markers",
        marker={"symbol": "star", "size": 16, "color": "#B08300"},
        name="current λ"))
    fig3.update_layout(xaxis_title="acoustic exposure (lower = quieter)",
                       yaxis_title="mission energy (Wh)",
                       height=380, margin=dict(t=10, b=10),
                       legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Wind aloft (live forecast)")
    rows = []
    for alt in (10, 80, 120, 180, 250):
        s, d = wf.wind(0.5, alt)
        rows.append({"altitude (m)": alt, "speed (m/s)": round(s, 1),
                     "from (°)": round(d)})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Transitions")
    for x_m, m_from, m_to in opt.transition_points_m:
        arrow = "hover → cruise" if m_to == "FW" else "cruise → hover"
        st.write(f"• **{x_m/1000:.1f} km**: {arrow}")
