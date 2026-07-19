# Energy-Optimal VTOL Transition Planner

**Given a mission and real forecast wind at altitude: when should a VTOL
aircraft switch between hover and cruise?** This planner answers that with
dynamic programming over altitude × position × flight mode, optimizing
mission energy with **acoustic footprint** as a second objective — and
validates its plans by flying them in **PX4 SITL**, the same autopilot
software that flies real aircraft.

## Why

- Nearly every open AI-drone project targets multirotors. The hard regime of
  real electric aviation — the **transition** between rotor-borne hover and
  wing-borne cruise — is still flown on fixed, hand-tuned schedules.
- Hover burns ~6–10× cruise power, and it's the loud mode. **Noise is the
  actual regulatory blocker for urban eVTOL**, so it belongs in the
  objective, not the appendix.
- Wind at altitude changes the answer every day. The planner pulls the
  live [Open-Meteo](https://open-meteo.com/) forecast (10–180 m levels)
  per mission.

## Architecture

```
mission + wind forecast ──► planner core (Python) ──► plan + Pareto front
                              │  vehicle.py   energy model (momentum theory + drag polar)
                              │  wind.py      Open-Meteo wind-at-altitude, cached
                              │  noise.py     source levels + propagation + ground grid
                              │  optimize.py  DP over (step × altitude × mode), λ-sweep
                              ▼
                    PX4 SITL validation (WSL2 + Gazebo, standard_vtol)
                              │  plan_export.py   planner output → mission params
                              │  fly_mission.py   MAVSDK upload + fly + log
                              │  parse_logs.py    ULog battery → measured Wh
                              ▼
                    app/dashboard.py  (Streamlit demo)
```

## Quickstart (planner only — no simulator needed)

```bash
pip install -r requirements.txt
python -m planner.make_figures     # figures/04..07: power curve, profile, Pareto, noise
streamlit run app/dashboard.py     # interactive demo
```

## SITL validation loop

Requires PX4-Autopilot + Gazebo (see PX4 docs, `make px4_sitl gz_standard_vtol`).

```bash
python3 -m sitl.plan_export              # solve with today's wind → plan_params.json
python3 sitl/fly_mission.py --plan naive       # fly baseline, ULog recorded
python3 sitl/fly_mission.py --plan optimized   # fly optimized plan
python3 sitl/parse_logs.py <naive.ulg> <optimized.ulg>   # figures/08
```

## Honest limitations

- SITL wind is a simple constant/turbulence model, not the forecast field.
  The claim is two rigorous links: *plans optimized against real forecast
  wind* + *energy model validated against PX4 SITL* — not one oversold
  end-to-end loop.
- The acoustic model is first-order (per-mode source level, spherical
  spreading, absorption; synthetic population grid). Source levels are
  parameters in `noise.py`, ballparked from small-eVTOL noise literature.
- Vehicle parameters describe PX4's `standard_vtol` (~5 kg quadplane) and
  are exposed in `vehicle.py` for calibration from flight logs.

## Why not reinforcement learning?

The discretized search space (~50 steps × 8 altitudes × 2 modes) is small
enough to solve *optimally* in milliseconds with dynamic programming. RL
would be slower and approximate. Learning belongs where physics is weakest —
the power model — which can be fit from SITL flight logs.
