"""First-order acoustic footprint model.

Each flight mode gets a source sound level (dB re 20 uPa at 1 m).
Propagation to the ground: spherical spreading (-20 log10 r) plus a linear
atmospheric-absorption term. The ground is a grid with a synthetic
population-density layer (Gaussian 'neighborhood' blobs). The score is
population-weighted exceedance above an annoyance threshold.

Ballpark source levels follow small-eVTOL noise literature: rotor-borne
flight is far louder than wing-borne cruise; transition is the worst case
because both propulsion sets run. These are parameters, not claims.
"""

import numpy as np

SOURCE_DB_AT_1M = {"MC": 100.0, "TRANSITION": 103.0, "FW": 88.0}
ATM_ABSORPTION_DB_PER_M = 0.005
THRESHOLD_DB = 55.0            # 'annoyance' threshold at ground level


class GroundGrid:
    """Rectangular grid under the route. x along-route (m), y cross-route (m)."""

    def __init__(self, route_len_m, half_width_m=1500.0, nx=120, ny=60, seed=7):
        self.xs = np.linspace(0.0, route_len_m, nx)
        self.ys = np.linspace(-half_width_m, half_width_m, ny)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)   # (ny, nx)
        rng = np.random.default_rng(seed)
        # synthetic population density: a handful of neighborhoods
        pop = np.zeros_like(self.X)
        n_blobs = 6
        for _ in range(n_blobs):
            cx = rng.uniform(0.1, 0.9) * route_len_m
            cy = rng.uniform(-0.6, 0.6) * half_width_m
            sx = rng.uniform(400.0, 1200.0)
            sy = rng.uniform(300.0, 900.0)
            amp = rng.uniform(500.0, 3000.0)  # people per cell-ish, arbitrary units
            pop += amp * np.exp(-(((self.X - cx) / sx) ** 2 +
                                  ((self.Y - cy) / sy) ** 2))
        self.pop = pop
        self.cell_area_km2 = ((self.xs[1] - self.xs[0]) *
                              (self.ys[1] - self.ys[0])) / 1e6

    def level_from_point(self, x_m, alt_m, mode):
        """Max ground SPL field (dB) from the aircraft at (x_m, y=0, alt_m)."""
        r = np.sqrt((self.X - x_m) ** 2 + self.Y ** 2 + alt_m ** 2)
        r = np.maximum(r, 1.0)
        return (SOURCE_DB_AT_1M[mode]
                - 20.0 * np.log10(r)
                - ATM_ABSORPTION_DB_PER_M * r)

    def footprint(self, path):
        """path: iterable of (x_m, alt_m, mode). Returns the LAmax ground
        field: per-cell maximum level over the whole flight."""
        lmax = np.full_like(self.X, -np.inf)
        for x_m, alt_m, mode in path:
            np.maximum(lmax, self.level_from_point(x_m, alt_m, mode), out=lmax)
        return lmax

    def exposure_score(self, lmax):
        """Population-weighted exceedance above THRESHOLD_DB (arbitrary units,
        comparable across plans)."""
        excess = np.clip(lmax - THRESHOLD_DB, 0.0, None)
        return float((excess * self.pop).sum() / 1e4)

    def area_above_threshold_km2(self, lmax):
        return float((lmax > THRESHOLD_DB).sum() * self.cell_area_km2)

    # ---- fast per-edge cost used inside the optimizer ----
    def edge_noise_cost(self, x0_m, x1_m, alt_m, mode, n_pts=3):
        """Approximate pop-weighted noise cost of flying one segment.
        Uses a few sample points; cheap enough for the DP loop."""
        cost = 0.0
        for i in range(n_pts):
            x = x0_m + (x1_m - x0_m) * (i + 0.5) / n_pts
            lvl = self.level_from_point(x, alt_m, mode)
            excess = np.clip(lvl - THRESHOLD_DB, 0.0, None)
            cost += float((excess * self.pop).sum())
        return cost / n_pts / 1e4


if __name__ == "__main__":
    g = GroundGrid(8500.0)
    quiet = g.footprint([(x, 120.0, "FW") for x in np.linspace(0, 8500, 40)])
    loud = g.footprint([(x, 60.0, "MC") for x in np.linspace(0, 8500, 40)])
    print(f"cruise@120m : score {g.exposure_score(quiet):8.1f}, "
          f"area>55dB {g.area_above_threshold_km2(quiet):5.2f} km^2")
    print(f"hover@60m   : score {g.exposure_score(loud):8.1f}, "
          f"area>55dB {g.area_above_threshold_km2(loud):5.2f} km^2")
