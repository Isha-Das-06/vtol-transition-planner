"""Fly a VTOL mission in PX4 SITL via MAVSDK and record the ULog.

Usage (inside WSL, with SITL already running):
    python3 sitl/fly_mission.py --plan optimized
    python3 sitl/fly_mission.py --plan naive

The mission is the demo route scaled onto the SITL world: takeoff,
transition to fixed-wing at a plan-dependent distance, cruise, transition
back, land vertically. The flight's ULog (with battery data) is what
parse_logs.py consumes afterwards.
"""

import argparse
import asyncio
import math

from mavsdk import System
from mavsdk.mission_raw import MissionItem

# MAVLink command ids we need (mission protocol)
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_VTOL_TAKEOFF = 84
MAV_CMD_NAV_VTOL_LAND = 85
MAV_CMD_DO_VTOL_TRANSITION = 3000
MAV_FRAME_GLOBAL_REL_ALT_INT = 6
MAV_FRAME_MISSION = 2
VTOL_STATE_FW = 4
VTOL_STATE_MC = 3

ROUTE_LEN_M = 2000.0   # scaled-down SITL route (full 8.5 km takes ages)
TRACK_DEG = 90.0       # due east


def offset_latlon(lat, lon, east_m, north_m=0.0):
    dlat = north_m / 111_111.0
    dlon = east_m / (111_111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def build_items(home_lat, home_lon, transition_out_m, transition_back_m,
                cruise_alt_m):
    """Mission: VTOL takeoff -> transition -> cruise WPs -> back-transition
    -> VTOL land."""
    items = []
    seq = 0

    def add(cmd, p1=0.0, p2=0.0, p3=0.0, p4=float("nan"),
            lat=0, lon=0, alt=0.0, frame=MAV_FRAME_GLOBAL_REL_ALT_INT,
            current=0):
        nonlocal seq
        items.append(MissionItem(
            seq, frame, cmd, current, 1,
            p1, p2, p3, p4,
            int(lat * 1e7), int(lon * 1e7), float(alt),
            0))  # mission_type 0
        seq += 1

    lat_t, lon_t = offset_latlon(home_lat, home_lon, transition_out_m)
    lat_c1, lon_c1 = offset_latlon(home_lat, home_lon, ROUTE_LEN_M * 0.5)
    lat_b, lon_b = offset_latlon(home_lat, home_lon, transition_back_m)
    lat_end, lon_end = offset_latlon(home_lat, home_lon, ROUTE_LEN_M)

    # Plain MC takeoff (NAV_VTOL_TAKEOFF would auto-transition immediately,
    # overriding the planner's schedule). Transitions are explicit DO items.
    add(MAV_CMD_NAV_TAKEOFF, lat=home_lat, lon=home_lon,
        alt=cruise_alt_m, current=1)
    add(MAV_CMD_NAV_WAYPOINT, lat=lat_t, lon=lon_t, alt=cruise_alt_m)
    add(MAV_CMD_DO_VTOL_TRANSITION, p1=VTOL_STATE_FW, frame=MAV_FRAME_MISSION)
    add(MAV_CMD_NAV_WAYPOINT, lat=lat_c1, lon=lon_c1, alt=cruise_alt_m)
    add(MAV_CMD_NAV_WAYPOINT, lat=lat_b, lon=lon_b, alt=cruise_alt_m)
    add(MAV_CMD_DO_VTOL_TRANSITION, p1=VTOL_STATE_MC, frame=MAV_FRAME_MISSION)
    add(MAV_CMD_NAV_VTOL_LAND, lat=lat_end, lon=lon_end, alt=0.0)
    return items


def load_plans():
    """Plan parameters come from sitl/plan_export.py (today's wind);
    fall back to sane defaults if it hasn't been run."""
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "plan_params.json")
    if os.path.exists(path):
        with open(path) as f:
            p = json.load(f)
        return {k: {kk: p[k][kk] for kk in
                    ("transition_out_m", "transition_back_m", "cruise_alt_m")}
                for k in ("naive", "optimized")}
    return {
        "naive": dict(transition_out_m=200.0,
                      transition_back_m=ROUTE_LEN_M - 200.0,
                      cruise_alt_m=80.0),
        "optimized": dict(transition_out_m=120.0,
                          transition_back_m=ROUTE_LEN_M - 350.0,
                          cruise_alt_m=120.0),
    }


PLANS = load_plans()


async def run(plan_name):
    plan = PLANS[plan_name]
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("waiting for connection…")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    print("connected; waiting for global position…")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break

    pos = None
    async for p in drone.telemetry.position():
        pos = p
        break
    home_lat, home_lon = pos.latitude_deg, pos.longitude_deg
    print(f"home: {home_lat:.6f}, {home_lon:.6f}")

    # lighter logging: default profile only (halves ULog size/IO)
    try:
        await drone.param.set_param_int("SDLOG_PROFILE", 1)
    except Exception as e:  # noqa: BLE001 - non-fatal
        print(f"param set warning: {e}")

    items = build_items(home_lat, home_lon, **plan)
    await drone.mission_raw.upload_mission(items)
    print(f"mission '{plan_name}' uploaded ({len(items)} items)", flush=True)

    await drone.action.arm()
    await drone.mission_raw.start_mission()
    print("mission started")

    async for progress in drone.mission_raw.mission_progress():
        print(f"  item {progress.current}/{progress.total}", flush=True)
        if progress.current == progress.total:
            break

    # Final descent takes ~60-90 s; the gz ground-contact glitch can swallow
    # the disarm event, so wait a fixed window instead of blocking on it.
    print("mission waypoints complete; allowing 90 s for descent…", flush=True)
    await asyncio.sleep(90)
    print("done. ULog is in PX4-Autopilot/build/px4_sitl_default/rootfs/log/",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=list(PLANS), default="optimized")
    args = ap.parse_args()
    asyncio.run(run(args.plan))
