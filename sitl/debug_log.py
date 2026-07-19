"""Quick ULog inspection: battery sign conventions and altitude profile."""
import sys

import numpy as np
from pyulog import ULog

ulog = ULog(sys.argv[1], ["battery_status", "vehicle_local_position"])
bat = next(d for d in ulog.data_list if d.name == "battery_status")
t = bat.data["timestamp"] / 1e6
v = bat.data["voltage_v"]
i = bat.data["current_a"]
print(f"battery samples: {len(t)}, t {t[0]:.0f}..{t[-1]:.0f}s")
print(f"voltage: min {v.min():.2f} max {v.max():.2f}")
print(f"current: min {i.min():.2f} max {i.max():.2f} mean {i.mean():.2f}")
for k in bat.data.keys():
    if "current" in k or "discharg" in k:
        arr = bat.data[k]
        try:
            print(f"  field {k}: min {np.min(arr):.3f} max {np.max(arr):.3f}")
        except Exception:
            pass

lp = next(d for d in ulog.data_list if d.name == "vehicle_local_position")
tz = lp.data["timestamp"] / 1e6
z = lp.data["z"]
print(f"z (NED down): min {z.min():.1f} max {z.max():.1f}")
airborne = z < -3.0
if airborne.any():
    i_up = int(np.argmax(airborne))
    after = np.where((tz > tz[i_up]) & (z > -0.5))[0]
    print(f"first airborne t={tz[i_up]-tz[0]:.0f}s ; "
          f"touchdown t={(tz[after[0]]-tz[0]) if after.size else -1:.0f}s "
          f"of {tz[-1]-tz[0]:.0f}s total")
