"""Vehicle energy model for a PX4 standard_vtol-class quadplane.

Point-mass physics with three regimes:
  - MC (multicopter / hover): momentum-theory rotor power
  - FW (fixed-wing / cruise): drag polar * airspeed / propulsive efficiency
  - TRANSITION: short high-power burst where both propulsion sets run

Parameters are chosen to describe the PX4 `standard_vtol` Gazebo airframe
(~5 kg quadplane). They are deliberately exposed as module constants so the
SITL log calibration (Build F) can overwrite them.
"""

from dataclasses import dataclass
import math

G = 9.81          # m/s^2
RHO0 = 1.225      # sea-level air density, kg/m^3


@dataclass
class Vehicle:
    mass: float = 5.0            # kg, standard_vtol all-up weight
    n_lift_rotors: int = 4
    rotor_radius: float = 0.17   # m
    figure_of_merit: float = 0.70  # hover efficiency vs ideal momentum theory
    wing_area: float = 0.45      # m^2
    aspect_ratio: float = 6.5
    oswald_e: float = 0.85
    cd0: float = 0.045           # parasitic drag coefficient
    prop_eta_fw: float = 0.75    # forward-flight propulsive efficiency
    avionics_w: float = 15.0     # constant systems draw, W
    battery_wh: float = 133.2    # 4S 9000 mAh class pack
    transition_duration_s: float = 12.0
    transition_power_factor: float = 1.25  # x hover power during transition
    climb_eta: float = 0.65      # efficiency of converting battery to altitude
    stall_speed: float = 11.0    # m/s, floor for FW airspeed
    best_airspeed: float = 16.0  # m/s, near best-range airspeed

    # ---- atmosphere ----
    @staticmethod
    def air_density(alt_m: float) -> float:
        """ISA density, good enough below 2 km."""
        return RHO0 * (1.0 - 2.2558e-5 * alt_m) ** 4.2559

    @property
    def disk_area(self) -> float:
        return self.n_lift_rotors * math.pi * self.rotor_radius ** 2

    @property
    def induced_k(self) -> float:
        return 1.0 / (math.pi * self.oswald_e * self.aspect_ratio)

    # ---- regime power models (Watts, electrical) ----
    def hover_power(self, alt_m: float = 0.0) -> float:
        """Momentum theory: P_ideal = W^1.5 / sqrt(2*rho*A), degraded by FoM."""
        w = self.mass * G
        rho = self.air_density(alt_m)
        p_ideal = w ** 1.5 / math.sqrt(2.0 * rho * self.disk_area)
        return p_ideal / self.figure_of_merit + self.avionics_w

    def cruise_power(self, airspeed: float, alt_m: float = 0.0) -> float:
        """Drag polar: P = D*V/eta. Below stall this model is invalid: penalize."""
        v = max(airspeed, 0.1)
        if v < self.stall_speed:
            # Blend toward hover cost: wing can't carry the weight yet.
            deficit = (self.stall_speed - v) / self.stall_speed
            return self.cruise_power(self.stall_speed, alt_m) + deficit * self.hover_power(alt_m)
        rho = self.air_density(alt_m)
        w = self.mass * G
        q = 0.5 * rho * v * v
        cl = w / (q * self.wing_area)
        cd = self.cd0 + self.induced_k * cl * cl
        drag = q * self.wing_area * cd
        return drag * v / self.prop_eta_fw + self.avionics_w

    def transition_energy_wh(self, alt_m: float = 0.0) -> float:
        """One transition event (either direction): a fixed-duration burst
        at transition_power_factor x hover power."""
        p = self.hover_power(alt_m) * self.transition_power_factor
        return p * self.transition_duration_s / 3600.0

    def climb_energy_wh(self, dalt_m: float) -> float:
        """Extra energy to gain altitude (potential energy / efficiency).
        Descent recovers nothing (conservative)."""
        if dalt_m <= 0:
            return 0.0
        return self.mass * G * dalt_m / self.climb_eta / 3600.0

    def power(self, mode: str, airspeed: float = 0.0, alt_m: float = 0.0) -> float:
        """Unified entry point. mode in {'MC', 'FW'}."""
        if mode == "MC":
            # Slow forward flight in MC costs roughly hover power.
            return self.hover_power(alt_m)
        if mode == "FW":
            return self.cruise_power(airspeed, alt_m)
        raise ValueError(f"unknown mode {mode!r}")


if __name__ == "__main__":
    v = Vehicle()
    print(f"disk area           : {v.disk_area:6.3f} m^2")
    print(f"hover power @ 50 m  : {v.hover_power(50):6.0f} W")
    print(f"cruise power @ 16 m/s: {v.cruise_power(16.0, 50):5.0f} W")
    print(f"hover/cruise ratio  : {v.hover_power(50)/v.cruise_power(16.0,50):6.1f}x")
    print(f"transition energy   : {v.transition_energy_wh(50)*3600:6.0f} J "
          f"({v.transition_energy_wh(50):.2f} Wh)")
    print(f"battery             : {v.battery_wh:.0f} Wh")
