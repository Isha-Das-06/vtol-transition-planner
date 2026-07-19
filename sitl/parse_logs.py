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
    """Return (t_s, cum_wh, vtol_mode_series) from a ULog."""
    ulog = ULog(path, ["battery_status", "vehicle_status"])
    bat = next(d for d in ulog.data_list if d.name == "battery_status")
    t = bat.data["timestamp"] / 1e6
    v = bat.data["voltage_v"]
    i = bat.data["current_a"]
    p = v * i                                   # W
    dt = np.diff(t, prepend=t[0])
    cum_wh = np.cumsum(p * dt) / 3600.0
    t0 = t[0]

    mode = None
    try:
        vs = next(d for d in ulog.data_list if d.name == "vehicle_status")
        mode = (vs.data["timestamp"] / 1e6 - t0,
                vs.data["vehicle_type"])        # 1=FW? PX4: 1 fixed wing, 2 MC
    except StopIteration:
        pass
    return t - t0, cum_wh, mode


def main(naive_path, opt_path):
    os.makedirs(FIGDIR, exist_ok=True)
    tn, en, _ = energy_trace(naive_path)
    to, eo, _ = energy_trace(opt_path)

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
