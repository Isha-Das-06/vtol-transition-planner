"""Real wind-at-altitude from the Open-Meteo forecast API.

Free, no API key. We pull hourly wind speed/direction at 10/80/120/180 m
above ground for sample points along the route, cache the raw JSON to disk
(offline-demo insurance), and expose an interpolated lookup:

    wf = WindField.fetch(lat_a, lon_a, lat_b, lon_b)
    speed_ms, dir_deg = wf.wind(frac_along_route, alt_m)

Above 180 m we extrapolate with a logarithmic profile — standard practice
for the surface layer and honest enough for a 40-300 m planning band.
"""

import json
import math
import os
import time

import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wind_cache")
LEVELS_M = [10.0, 80.0, 120.0, 180.0]
_API = "https://api.open-meteo.com/v1/forecast"
_VARS = ",".join(
    f"wind_speed_{int(z)}m,wind_direction_{int(z)}m" for z in LEVELS_M
)


class WindField:
    def __init__(self, samples):
        # samples: list of dicts {frac, lat, lon, speeds[4] (m/s), dirs[4] (deg)}
        self.samples = samples

    # ---------- construction ----------
    @classmethod
    def fetch(cls, lat_a, lon_a, lat_b, lon_b, n_samples=5, hour_offset=0,
              use_cache_only=False):
        """Sample the forecast at n points along the great-circle route.
        Falls back to the newest cache file if the network is down."""
        key = f"{lat_a:.3f}_{lon_a:.3f}_{lat_b:.3f}_{lon_b:.3f}"
        cache_path = os.path.join(CACHE_DIR, f"wind_{key}.json")

        if not use_cache_only:
            try:
                lats = [lat_a + (lat_b - lat_a) * i / (n_samples - 1) for i in range(n_samples)]
                lons = [lon_a + (lon_b - lon_a) * i / (n_samples - 1) for i in range(n_samples)]
                r = requests.get(_API, params={
                    "latitude": ",".join(f"{x:.4f}" for x in lats),
                    "longitude": ",".join(f"{x:.4f}" for x in lons),
                    "hourly": _VARS,
                    "wind_speed_unit": "ms",
                    "forecast_days": 2,
                }, timeout=15)
                r.raise_for_status()
                raw = r.json()
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({"fetched_at": time.time(), "raw": raw}, f)
            except Exception as e:  # noqa: BLE001 - any network failure -> cache
                print(f"[wind] fetch failed ({e}); trying cache")

        if not os.path.exists(cache_path):
            # last resort: any cache file at all
            cand = sorted(
                (os.path.join(CACHE_DIR, p) for p in os.listdir(CACHE_DIR))
                if os.path.isdir(CACHE_DIR) else [],
                key=os.path.getmtime, reverse=True)
            if not cand:
                raise RuntimeError("no wind data and no cache — run once online")
            cache_path = cand[0]

        with open(cache_path) as f:
            raw = json.load(f)["raw"]
        if isinstance(raw, dict):
            raw = [raw]

        # pick the forecast hour closest to now + offset
        samples = []
        for i, loc in enumerate(raw):
            hourly = loc["hourly"]
            times = hourly["time"]
            now_iso = time.strftime("%Y-%m-%dT%H:00", time.localtime())
            try:
                idx = times.index(now_iso) + hour_offset
            except ValueError:
                idx = 0
            idx = max(0, min(idx, len(times) - 1))
            speeds = [hourly[f"wind_speed_{int(z)}m"][idx] for z in LEVELS_M]
            dirs = [hourly[f"wind_direction_{int(z)}m"][idx] for z in LEVELS_M]
            samples.append({
                "frac": i / (len(raw) - 1) if len(raw) > 1 else 0.0,
                "lat": loc["latitude"], "lon": loc["longitude"],
                "speeds": speeds, "dirs": dirs,
            })
        return cls(samples)

    # ---------- lookup ----------
    def wind(self, frac, alt_m):
        """Interpolated (speed m/s, direction deg FROM) at a route fraction
        and altitude above ground."""
        frac = min(max(frac, 0.0), 1.0)
        # nearest two route samples
        s = self.samples
        for j in range(len(s) - 1):
            if s[j]["frac"] <= frac <= s[j + 1]["frac"]:
                t = ((frac - s[j]["frac"]) /
                     (s[j + 1]["frac"] - s[j]["frac"] or 1.0))
                a, b = s[j], s[j + 1]
                break
        else:
            a = b = s[-1]
            t = 0.0
        sp_a = _interp_alt(a["speeds"], alt_m)
        sp_b = _interp_alt(b["speeds"], alt_m)
        di_a = _interp_alt(a["dirs"], alt_m, angular=True)
        di_b = _interp_alt(b["dirs"], alt_m, angular=True)
        speed = sp_a + (sp_b - sp_a) * t
        ddir = ((di_b - di_a + 180) % 360) - 180
        return speed, (di_a + ddir * t) % 360

    def headwind_component(self, frac, alt_m, track_deg):
        """Positive = headwind along the given ground track."""
        speed, wdir = self.wind(frac, alt_m)
        # wind FROM wdir; headwind when wind blows against the track
        return speed * math.cos(math.radians(wdir - track_deg))


def _interp_alt(vals, alt_m, angular=False):
    """Interpolate across LEVELS_M; log-profile extrapolation above the top."""
    zs = LEVELS_M
    if alt_m <= zs[0]:
        return vals[0]
    if alt_m >= zs[-1]:
        if angular:
            return vals[-1]
        # log wind profile anchored on the top two levels
        z1, z2 = zs[-2], zs[-1]
        v1, v2 = vals[-2], vals[-1]
        if v2 <= 0 or v1 <= 0 or v2 == v1:
            return vals[-1]
        alpha = math.log(v2 / v1) / math.log(z2 / z1)
        return v2 * (alt_m / z2) ** alpha
    for k in range(len(zs) - 1):
        if zs[k] <= alt_m <= zs[k + 1]:
            t = (alt_m - zs[k]) / (zs[k + 1] - zs[k])
            if angular:
                d = ((vals[k + 1] - vals[k] + 180) % 360) - 180
                return (vals[k] + d * t) % 360
            return vals[k] + (vals[k + 1] - vals[k]) * t
    return vals[-1]


if __name__ == "__main__":
    # demo route: ~8.5 km across Bengaluru
    wf = WindField.fetch(12.935, 77.535, 12.985, 77.610)
    for alt in (10, 50, 120, 180, 250):
        s, d = wf.wind(0.5, alt)
        print(f"alt {alt:4d} m : {s:5.1f} m/s from {d:5.1f} deg")
