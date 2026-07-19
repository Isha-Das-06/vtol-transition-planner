"""Generate the core figures for the pitch (shot-list items 4-7).

Run:  python -m planner.make_figures          (from the repo root)
Writes PNGs into figures/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .vehicle import Vehicle
from .wind import WindField
from .noise import GroundGrid, THRESHOLD_DB
from .optimize import TransitionPlanner, ALTS

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# demo mission: ~8.5 km eastbound across a city
LAT_A, LON_A = 12.935, 77.535
LAT_B, LON_B = 12.985, 77.610
ROUTE_LEN_M = 8500.0
TRACK_DEG = 55.0

MODE_COLOR = {"MC": "#D9480F", "FW": "#0B7285"}


def fig_power_curve(v: Vehicle):
    speeds = np.linspace(0, 26, 200)
    p_fw = [v.cruise_power(s, 80.0) for s in speeds]
    p_mc = v.hover_power(80.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(speeds, p_fw, color="#0B7285", lw=2.2, label="Fixed-wing (drag polar)")
    ax.axhline(p_mc, color="#D9480F", lw=2.2, ls="--",
               label=f"Hover (momentum theory) ≈ {p_mc:.0f} W")
    ax.axvline(v.stall_speed, color="grey", lw=1, ls=":")
    ax.annotate("stall", (v.stall_speed, ax.get_ylim()[1] * 0.55),
                rotation=90, va="top", ha="right", color="grey")
    best = min(range(len(speeds)), key=lambda i: p_fw[i] / max(speeds[i], .1))
    ax.scatter([speeds[best]], [p_fw[best]], zorder=5, color="#0B7285")
    ax.annotate(f"best range ≈ {speeds[best]:.0f} m/s",
                (speeds[best], p_fw[best]), xytext=(8, -14),
                textcoords="offset points", fontsize=9)
    ax.set_xlabel("airspeed (m/s)")
    ax.set_ylabel("electrical power (W)")
    ax.set_title("Why transition timing matters: hover vs cruise power")
    ax.legend(frameon=False)
    ax.set_ylim(0, p_mc * 1.35)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "04_power_curve.png"), dpi=160)
    plt.close(fig)


def fig_plans(planner, naive, opt, wf):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})
    for plan, ls, label in ((naive, "--", "naive (fixed schedule)"),
                            (opt, "-", "optimized")):
        xs = [s[0] / 1000 for s in plan.steps]
        alts = [s[1] for s in plan.steps]
        ax1.plot(xs, alts, ls, color="#555", lw=1.2, alpha=.6)
        for i in range(len(plan.steps) - 1):
            ax1.plot(xs[i:i+2], alts[i:i+2],
                     color=MODE_COLOR[plan.steps[i+1][2]],
                     lw=3 if ls == "-" else 1.8,
                     alpha=1.0 if ls == "-" else 0.45)
        for x_m, m_from, m_to in plan.transition_points_m:
            if ls == "-":
                ax1.axvline(x_m / 1000, color="#D9480F", lw=.8, ls=":", alpha=.7)
    ax1.set_ylabel("altitude AGL (m)")
    ax1.set_title(
        f"Mission profile — naive {naive.energy_wh:.1f} Wh vs "
        f"optimized {opt.energy_wh:.1f} Wh "
        f"({100*(1-opt.energy_wh/naive.energy_wh):.0f}% saved)")
    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([], [], color="#D9480F", lw=3, label="rotor-borne (MC)"),
        Line2D([], [], color="#0B7285", lw=3, label="wing-borne (FW)"),
        Line2D([], [], color="#555", lw=1.5, ls="--", alpha=.6, label="naive plan"),
    ], frameon=False, loc="lower center", ncol=3)

    # wind barbs along route at cruise band
    xs = np.linspace(0, ROUTE_LEN_M, 18)
    for x in xs:
        sp, wd = wf.wind(x / ROUTE_LEN_M, 120.0)
        head = wf.headwind_component(x / ROUTE_LEN_M, 120.0, TRACK_DEG)
        ax2.arrow(x / 1000, 0, 0.0001, -head, head_width=.12,
                  head_length=.25, length_includes_head=True,
                  color="#B08300" if head > 0 else "#2B7A3D", alpha=.8)
    ax2.axhline(0, color="grey", lw=.5)
    ax2.set_ylabel("headwind (m/s)\n(down = tailwind)")
    ax2.set_xlabel("distance along route (km)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "05_mission_profile.png"), dpi=160)
    plt.close(fig)


def fig_pareto(plans, front, naive):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter([p.noise_score for p in plans], [p.energy_wh for p in plans],
               color="#9AA7B8", s=30, label="candidate plans (λ sweep)")
    fs = sorted(front, key=lambda p: p.noise_score)
    ax.plot([p.noise_score for p in fs], [p.energy_wh for p in fs],
            "-o", color="#0B7285", lw=2, ms=6, label="Pareto front")
    ax.scatter([naive.noise_score], [naive.energy_wh], marker="X", s=120,
               color="#D9480F", zorder=5, label="naive fixed schedule")
    for p, tag in ((fs[0], "quietest"), (fs[-1], "most efficient")):
        ax.annotate(tag, (p.noise_score, p.energy_wh), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color="#0B7285")
    ax.set_xlabel("acoustic exposure score (pop-weighted, lower = quieter)")
    ax.set_ylabel("mission energy (Wh)")
    ax.set_title("The trade: energy vs acoustic footprint")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "06_pareto_front.png"), dpi=160)
    plt.close(fig)


def fig_noise_maps(grid, naive, opt):
    def path_of(plan):
        return [(s[0], s[1], s[2]) for s in plan.steps]
    ln = grid.footprint(path_of(naive))
    lo = grid.footprint(path_of(opt))
    vmax = max(ln.max(), lo.max())
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for ax, lmax, plan, title in (
            (axes[0], ln, naive, "naive fixed schedule"),
            (axes[1], lo, opt, "optimized (noise-aware)")):
        pc = ax.pcolormesh(grid.xs / 1000, grid.ys / 1000, lmax,
                           cmap="magma", vmin=35, vmax=vmax, shading="auto")
        ax.contour(grid.xs / 1000, grid.ys / 1000, grid.pop,
                   levels=4, colors="white", linewidths=.5, alpha=.5)
        ax.contour(grid.xs / 1000, grid.ys / 1000, lmax,
                   levels=[THRESHOLD_DB], colors="#4FC3D9", linewidths=1.6)
        ax.set_title(f"{title} — area >{THRESHOLD_DB:.0f} dB: "
                     f"{grid.area_above_threshold_km2(lmax):.2f} km², "
                     f"exposure {grid.exposure_score(lmax):.0f}", fontsize=10)
        ax.set_ylabel("cross-track (km)")
    axes[1].set_xlabel("along route (km)")
    cb = fig.colorbar(pc, ax=axes, shrink=.85, label="max ground level (dB)")
    fig.savefig(os.path.join(FIGDIR, "07_noise_footprint.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    v = Vehicle()
    print("[1/5] power curve")
    fig_power_curve(v)

    print("[2/5] wind")
    wf = WindField.fetch(LAT_A, LON_A, LAT_B, LON_B)

    print("[3/5] solving plans")
    grid = GroundGrid(ROUTE_LEN_M)
    planner = TransitionPlanner(v, grid, wf, ROUTE_LEN_M, TRACK_DEG)
    naive = planner.naive_plan()
    plans, front = planner.pareto_sweep()
    opt = front[len(front) // 2] if front else plans[0]

    print(f"    naive     : {naive.energy_wh:6.1f} Wh  noise {naive.noise_score:8.1f}")
    for p in front:
        print(f"    lam={p.lam:<6} : {p.energy_wh:6.1f} Wh  noise {p.noise_score:8.1f}")

    print("[4/5] profile + pareto figures")
    fig_plans(planner, naive, opt, wf)
    fig_pareto(plans, front, naive)

    print("[5/5] noise maps")
    fig_noise_maps(grid, naive, opt)
    print(f"done -> {os.path.abspath(FIGDIR)}")


if __name__ == "__main__":
    main()
