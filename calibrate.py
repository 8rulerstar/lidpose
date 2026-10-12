"""Estimate the webcam intrinsics from views of the board.

Show board.png full screen on a phone and move it around in front of the camera:
near and far, tilted left, right, up and down. Frames are taken automatically.
"""

import sys
import time

import cv2
import numpy as np

from board import CAMERA, CAMERA_HELP, detect

VIEWS, GAP_S = 25, 0.7

cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    sys.exit(f"camera {CAMERA} unavailable: {CAMERA_HELP}")
obj_pts, img_pts, last, size = [], [], 0.0, None
misses = 0
while len(obj_pts) < VIEWS:
    ok, frame = cap.read()
    if not ok:
        misses += 1
        if misses > 100:
            sys.exit("camera opened but sent no frames")
        continue
    misses = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    found = detect(gray)
    if found and time.time() - last > GAP_S:
        obj_pts.append(found[0])
        img_pts.append(found[1])
        last = time.time()
        print(f"view {len(obj_pts)}/{VIEWS}: {len(found[0])} corners", flush=True)
    if found:
        for p in found[1].reshape(-1, 2):
            cv2.circle(frame, tuple(int(v) for v in p), 4, (0, 255, 0), -1)
    cv2.putText(frame, f"{len(obj_pts)}/{VIEWS}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("calibrate", frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
fov = np.degrees(2 * np.arctan2(np.array(size) / 2, [K[0, 0], K[1, 1]]))
print(f"image {size}, reprojection RMS {rms:.2f} px, fx {K[0, 0]:.0f} fy {K[1, 1]:.0f}, FOV {fov[0]:.1f} x {fov[1]:.1f} deg")
np.savez("calib.npz", K=K, dist=dist, size=size, rms=rms)
print("wrote calib.npz")
