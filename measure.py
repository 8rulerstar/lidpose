"""How does the camera pitch follow the lid angle?

The x-axis slope comes out negative (about -1); LIDPOSE_SLOPE takes its magnitude.

Prop the phone with board.png so it stays still in view, then slowly open and close
the lid for the whole run. Each frame records the lid angle and the camera rotation
relative to the first frame; a line is fitted to pitch change against lid change.
"""

import sys
import time

from pathlib import Path

import cv2
import numpy as np

from board import CAMERA, CAMERA_HELP, detect
from lid import Lid

try:
    SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 40
except ValueError:
    sys.exit("usage: measure.py [seconds]")
if not 0 < SECONDS < 3600:
    sys.exit("seconds must be between 0 and 3600")

if not Path("calib.npz").exists():
    sys.exit("calib.npz not found: run calibrate.py first")
c = np.load("calib.npz")
K, dist = c["K"], c["dist"]
try:
    lid = Lid()
except RuntimeError as e:
    sys.exit(f"{e}; see Requirements in README.md")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    sys.exit(f"camera {CAMERA} unavailable: {CAMERA_HELP}")
rows, R0, misses = [], None, 0
end = time.time() + SECONDS
while time.time() < end:
    ok, frame = cap.read()
    if not ok:
        misses += 1
        if misses > 100:
            sys.exit(f"camera {CAMERA} sent no frames: {CAMERA_HELP}")
        continue
    misses = 0
    if frame.shape[1::-1] != tuple(c["size"]):
        sys.exit(f"camera gives {frame.shape[1]}x{frame.shape[0]} but calib.npz is for {c['size'][0]}x{c['size'][1]}: recalibrate with this camera")
    try:
        angle = lid.angle()
    except RuntimeError as e:  # lid closed or an odd reading
        sys.exit(f"lid sensor unreadable: {e}")
    found = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if found:
        ok, rvec, _ = cv2.solvePnP(found[0], found[1], K, dist)
        if ok:
            R = cv2.Rodrigues(rvec)[0]
            R0 = R if R0 is None else R0
            rel = cv2.Rodrigues(R @ R0.T)[0].ravel()  # camera rotation since the first frame, camera axes
            rows.append((angle, *np.degrees(rel)))
    cv2.putText(frame, f"lid {angle}  frames {len(rows)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("measure", frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
lid.close()

if len(rows) < 10:
    sys.exit(f"the board was seen in only {len(rows)} frames; keep it in view and run again")
a = np.array(rows)
np.savetxt("measure.csv", a, delimiter=",", header="lid,rx,ry,rz", comments="", fmt="%.3f")
if np.ptp(a[:, 0]) < 20:
    sys.exit(f"the lid moved only {np.ptp(a[:, 0]):.0f} degrees; open and close it further (20 or more) and run again")
lid_change = a[:, 0] - a[0, 0]
print(f"{len(a)} frames, lid {a[:, 0].min():.0f} to {a[:, 0].max():.0f} deg")
for i, axis in enumerate("xyz", start=1):
    slope, offset = np.polyfit(lid_change, a[:, i], 1)
    resid = a[:, i] - (slope * lid_change + offset)
    print(f"rotation about camera {axis}: slope {slope:+.3f} deg per lid deg, residual SD {resid.std():.2f} deg")
x_slope = np.polyfit(lid_change, a[:, 1], 1)[0]
print(f"to use it: LIDPOSE_SLOPE={abs(x_slope):.2f}")
