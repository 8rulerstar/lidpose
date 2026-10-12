"""Recompute the README measurement table from recorded runs (head_marks_run*.npz).

Segments are fixed here so the numbers can be reproduced:
  run4 0-15 s   head still, lid swung 83-133
  run5 0-19 s   head still, lid swung 99-133 (a nod starts at 20 s)
  run7 14-22 s  head still, lid and laptop distance changed together
  run7 37-42 s  head still, lid left at 117 after starting at 123

    uv run python analysis/table.py

It needs the recordings, which are not published (they hold face landmarks), so only
the author can run it. It uses a lid slope of 1 whatever LIDPOSE_SLOPE says.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import head  # noqa: E402

head.SLOPE, head.DIST_GAIN = 1.0, 0.0


def series(path):
    d = np.load(path)
    t, lid, marks = d["t"], d["lid"].astype(float), d["marks"]
    pnp = head.PnPPitch()
    pitch = np.array([np.nan if (v := pnp(m)) is None else v for m in marks])
    ok = ~np.isnan(pitch)
    i = np.where(ok)[0][0]
    camera = pitch - pitch[i]
    return t[ok], lid[ok], camera[ok], head.corrected(camera, lid, lid[i])[ok]


def main():
    if not Path("head_marks_run4.npz").exists():
        sys.exit("needs the unpublished recordings head_marks_run4/5/7.npz")
    print("note: these recordings hold no calibration; using calib.npz, which must be the one they were recorded with")
    runs = {n: series(f"head_marks_run{n}.npz") for n in (4, 5, 7)}
    for n, a, b in ((4, 0, 15), (5, 0, 19), (7, 14, 22)):
        t, lid, camera, desk = runs[n]
        m = (t >= a) & (t < b)
        slope = -np.polyfit(lid[m], camera[m], 1)[0]  # the raw slope is negative; report its magnitude
        print(
            f"run{n} {a}-{b} s: lid {lid[m].min():.0f}-{lid[m].max():.0f}, camera SD {camera[m].std():.1f}, "
            f"corrected SD {desk[m].std():.1f}, camera pitch per lid degree {slope:.2f}"
        )
    t, lid, camera, desk = runs[7]
    m = (t >= 37) & (t < 42)
    print(f"run7 37-42 s still at lid {lid[m].mean():.0f}: corrected mean {desk[m].mean():+.1f}, SD {desk[m].std():.1f}")


if __name__ == "__main__":
    main()
