"""The planner: dynamic programming over (route step, altitude, mode).

State graph:
  - route discretized into N steps of ds meters
  - altitude levels ALTS (m AGL)
  - mode in {MC, FW}
Edges: advance one step while (holding | climbing | descending) one altitude
level, optionally switching mode (which prices a transition burst).
Edge cost = energy_Wh + lam * noise_cost. Sweeping lam yields the Pareto
front between total energy and acoustic exposure.

Takeoff/landing are pinned: start MC at ALTS[0] on step 0, end MC at
ALTS[0] on the last step (you must be rotor-borne to land vertically).
"""

from dataclasses import dataclass, field
import math

import numpy as np

from .vehicle import Vehicle
from .noise import GroundGrid

ALTS = [40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 180.0, 220.0]
MODES = ["MC", "FW"]
MC_SPEED = 6.0     # m/s ground speed in multicopter mode (conservative)


@dataclass
class Plan:
    steps: list = field(default_factory=list)  # (x_m, alt_m, mode)
    energy_wh: float = 0.0
    noise_score: float = 0.0
    duration_s: float = 0.0
    lam: float = 0.0

    @property
    def transition_points_m(self):
        pts = []
        for i in range(1, len(self.steps)):
            if self.steps[i][2] != self.steps[i - 1][2]:
                pts.append((self.steps[i][0], self.steps[i - 1][2],
                            self.steps[i][2]))
        return pts


class TransitionPlanner:
    def __init__(self, vehicle: Vehicle, grid: GroundGrid, wind_field,
                 route_len_m: float, track_deg: float = 90.0, n_steps: int = 50):
        self.v = vehicle
        self.grid = grid
        self.wf = wind_field
        self.L = route_len_m
        self.track = track_deg
        self.N = n_steps
        self.ds = route_len_m / n_steps
        self._noise_cache = {}
        self._precompute_noise()

    def _precompute_noise(self):
        for i in range(self.N):
            x0, x1 = i * self.ds, (i + 1) * self.ds
            for ai, alt in enumerate(ALTS):
                for mode in MODES + ["TRANSITION"]:
                    self._noise_cache[(i, ai, mode)] = \
                        self.grid.edge_noise_cost(x0, x1, alt, mode)

    # ---- edge physics ----
    def _edge(self, i, ai_from, ai_to, mode_from, mode_to):
        """Energy (Wh), noise cost, duration (s) to advance step i."""
        alt = ALTS[ai_to]
        frac = (i + 0.5) / self.N
        head = self.wf.headwind_component(frac, alt, self.track)

        energy = 0.0
        noise = 0.0
        dur = 0.0

        if mode_to == "FW":
            vg = max(self.v.best_airspeed - head, 2.0)  # ground speed
            t = self.ds / vg
            energy += self.v.cruise_power(self.v.best_airspeed, alt) * t / 3600.0
        else:
            vg = MC_SPEED
            t = self.ds / vg
            energy += self.v.hover_power(alt) * t / 3600.0
        dur += t
        noise += self._noise_cache[(i, ai_to, mode_to)]

        # climb cost between levels
        energy += self.v.climb_energy_wh(ALTS[ai_to] - ALTS[ai_from])

        # mode switch: transition burst, priced + extra noise at this cell
        if mode_from != mode_to:
            energy += self.v.transition_energy_wh(alt)
            dur += self.v.transition_duration_s
            noise += self._noise_cache[(i, ai_to, "TRANSITION")]

        return energy, noise, dur

    # ---- DP solve ----
    def solve(self, lam: float = 0.0) -> Plan:
        NA, NM = len(ALTS), len(MODES)
        INF = float("inf")
        cost = np.full((self.N + 1, NA, NM), INF)
        e_acc = np.zeros_like(cost)
        n_acc = np.zeros_like(cost)
        t_acc = np.zeros_like(cost)
        parent = np.full((self.N + 1, NA, NM, 2), -1, dtype=int)

        cost[0, 0, 0] = 0.0  # start: lowest altitude, MC

        for i in range(self.N):
            for ai in range(NA):
                for mi, mode in enumerate(MODES):
                    c0 = cost[i, ai, mi]
                    if not math.isfinite(c0):
                        continue
                    for aj in (ai - 1, ai, ai + 1):
                        if not 0 <= aj < NA:
                            continue
                        for mj, mode_to in enumerate(MODES):
                            e, nz, du = self._edge(i, ai, aj, mode, mode_to)
                            c = c0 + e + lam * nz
                            if c < cost[i + 1, aj, mj]:
                                cost[i + 1, aj, mj] = c
                                e_acc[i + 1, aj, mj] = e_acc[i, ai, mi] + e
                                n_acc[i + 1, aj, mj] = n_acc[i, ai, mi] + nz
                                t_acc[i + 1, aj, mj] = t_acc[i, ai, mi] + du
                                parent[i + 1, aj, mj] = (ai, mi)

        # goal: back at lowest altitude in MC for vertical landing
        gi, gm = 0, 0
        if not math.isfinite(cost[self.N, gi, gm]):
            raise RuntimeError("no feasible plan — check ALTS/step counts")

        # backtrack
        steps = []
        ai, mi = gi, gm
        for i in range(self.N, 0, -1):
            steps.append((i * self.ds, ALTS[ai], MODES[mi]))
            ai, mi = parent[i, ai, mi]
        steps.append((0.0, ALTS[ai], MODES[mi]))
        steps.reverse()

        return Plan(steps=steps,
                    energy_wh=float(e_acc[self.N, gi, gm]),
                    noise_score=float(n_acc[self.N, gi, gm]),
                    duration_s=float(t_acc[self.N, gi, gm]),
                    lam=lam)

    def naive_plan(self, transition_dist_m: float = 500.0,
                   cruise_alt: float = 80.0) -> Plan:
        """Baseline: fixed transition distance and altitude, like a default
        mission planner would fly. Same edge pricing, no optimization."""
        ci = ALTS.index(cruise_alt)
        plan_states = []
        ai, mi = 0, 0
        for i in range(self.N):
            x_mid = (i + 0.5) * self.ds
            in_cruise_band = transition_dist_m < x_mid < self.L - transition_dist_m
            steps_left = self.N - (i + 1)
            if steps_left <= ai:                 # out of runway: descend to land
                aj = max(ai - 1, 0)
            elif ai < ci:                        # climb to cruise altitude
                aj = ai + 1
            else:
                aj = ai
            mj = 1 if (in_cruise_band and aj == ci) else 0
            plan_states.append((i, ai, aj, mi, mj))
            ai, mi = aj, mj

        # force MC at the end
        plan = Plan(lam=-1.0)
        e = nz = du = 0.0
        steps = [(0.0, ALTS[0], "MC")]
        for (i, ai, aj, mi, mj) in plan_states:
            de, dn, dt = self._edge(i, ai, aj, MODES[mi], MODES[mj])
            e, nz, du = e + de, nz + dn, du + dt
            steps.append(((i + 1) * self.ds, ALTS[aj], MODES[mj]))
        plan.steps, plan.energy_wh, plan.noise_score, plan.duration_s = \
            steps, e, nz, du
        return plan

    def pareto_sweep(self, lams=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
        plans = [self.solve(lam) for lam in lams]
        # keep only non-dominated
        front = []
        for p in plans:
            if not any(q.energy_wh <= p.energy_wh and
                       q.noise_score < p.noise_score for q in plans if q is not p):
                front.append(p)
        return plans, front
