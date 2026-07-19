"""Bridge: Tier 1 planner output -> SITL mission parameters.

Solves the demo mission with the real wind forecast, extracts the two
transition points and cruise altitude, scales them onto the (shorter) SITL
route, and writes sitl/plan_params.json which fly_mission.py reads.

Run from the repo root:  python3 -m sitl.plan_export
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from planner.vehicle import Vehicle
from planner.wind import WindField
from planner.noise import GroundGrid
from planner.optimize import TransitionPlanner

LAT_A, LON_A = 12.935, 77.535
LAT_B, LON_B = 12.985, 77.610
ROUTE_LEN_M = 8500.0
TRACK_DEG = 55.0
SITL_ROUTE_LEN_M = 2000.0

OUT = os.path.join(os.path.dirname(__file__), "plan_params.json")


def main():
    v = Vehicle()
    wf = WindField.fetch(LAT_A, LON_A, LAT_B, LON_B)
    grid = GroundGrid(ROUTE_LEN_M)
    planner = TransitionPlanner(v, grid, wf, ROUTE_LEN_M, TRACK_DEG)
    opt = planner.solve(lam=0.01)

    tps = opt.transition_points_m
    fw_start = next((x for x, _, m_to in tps if m_to == "FW"), ROUTE_LEN_M * .1)
    fw_end = next((x for x, _, m_to in reversed(tps) if m_to == "MC"),
                  ROUTE_LEN_M * .9)
    # cruise altitude: most common FW altitude
    fw_alts = [alt for _, alt, m in opt.steps if m == "FW"]
    cruise_alt = max(set(fw_alts), key=fw_alts.count) if fw_alts else 100.0

    scale = SITL_ROUTE_LEN_M / ROUTE_LEN_M
    params = {
        "optimized": {
            # clamp so the back-transition finishes with margin before the pad
            "transition_out_m": max(round(fw_start * scale, 1), 40.0),
            "transition_back_m": min(round(fw_end * scale, 1),
                                     SITL_ROUTE_LEN_M - 150.0),
            "cruise_alt_m": float(cruise_alt),
        },
        "naive": {
            "transition_out_m": 200.0,
            "transition_back_m": SITL_ROUTE_LEN_M - 200.0,
            "cruise_alt_m": 80.0,
        },
        "planner_prediction_wh": {
            "optimized": round(opt.energy_wh, 2),
            "naive": round(planner.naive_plan().energy_wh, 2),
        },
    }
    with open(OUT, "w") as f:
        json.dump(params, f, indent=2)
    print(json.dumps(params, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
