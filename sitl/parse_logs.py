"""Extract energy usage from PX4 SITL ULog files and chart the comparison.

Usage (inside WSL):
    python3 sitl/parse_logs.py <naive.ulg> <optimized.ulg>

Produces figures/08_energy_comparison.png: cumulative energy traces plus
total-Wh bars, the 'proof' chart of the project.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def energy_trace(path):
    """Return (t_s, cum_wh) from a ULog, truncated at touchdown.

    The gz ground-contact glitch can let the model clip through the terrain
    after landing; everything past the moment the vehicle first returns to
    ground level (after having been airborne) is discarded.
    """
    ulog = ULog(path, ["battery_status", "vehicle_local_position"])
    bat = next(d for d in ulog.data_list if d.name == "battery_status")
    t = bat.data["timestamp"] / 1e6
    v = bat.data["voltage_v"]
    # SITL's battery reports -1 in current_a; the real draw is in
    # current_average_a. Prefer whichever actually varies.
    i = bat.data["current_a"]
    if np.ptp(i) < 0.01 and "current_average_a" in bat.data:
        i = bat.data["current_average_a"]
    p = v * np.clip(i, 0.0, None)               # W

    # touchdown detection from local position (NED: z negative = airborne)
    t_end = t[-1]
    try:
        lp = next(d for d in ulog.data_list
                  if d.name == "vehicle_local_position")
        tz = lp.data["timestamp"] / 1e6
        z = lp.data["z"]
        airborne = z < -3.0                     # ever higher than 3 m
        if airborne.any():
            i_up = int(np.argmax(airborne))
            after = np.where((tz > tz[i_up]) & (z > -0.5))[0]
            if after.size:
                t_end = tz[after[0]]
    except StopIteration:
        pass

    keep = t <= t_end
    t, p = t[keep], p[keep]
    dt = np.diff(t, prepend=t[0])
    cum_wh = np.cumsum(p * dt) / 3600.0
    return t - t[0], cum_wh


def main(naive_path, opt_path):
    os.makedirs(FIGDIR, exist_ok=True)
    tn, en = energy_trace(naive_path)
    to, eo = energy_trace(opt_path)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax1.plot(tn, en, color="#9AA7B8", lw=2, label=f"naive ({en[-1]:.1f} Wh)")
    ax1.plot(to, eo, color="#0B7285", lw=2.4,
             label=f"optimized ({eo[-1]:.1f} Wh)")
    ax1.set_xlabel("flight time (s)")
    ax1.set_ylabel("cumulative energy (Wh)")
    ax1.set_title("Measured in PX4 SITL: cumulative battery energy")
    ax1.legend(frameon=False)

    saved = 100 * (1 - eo[-1] / en[-1])
    ax2.bar(["naive", "optimized"], [en[-1], eo[-1]],
            color=["#9AA7B8", "#0B7285"], width=.6)
    ax2.set_ylabel("total energy (Wh)")
    ax2.set_title(f"{saved:.0f}% saved")
    for x, y in enumerate([en[-1], eo[-1]]):
        ax2.text(x, y, f"{y:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    out = os.path.join(FIGDIR, "08_energy_comparison.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — naive {en[-1]:.1f} Wh, optimized {eo[-1]:.1f} Wh, "
          f"saved {saved:.0f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: parse_logs.py <naive.ulg> <optimized.ulg>")
    main(sys.argv[1], sys.argv[2])
