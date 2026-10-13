"""How much of a lid tilt does Apple Vision's face pitch follow, next to lidtilt's solved pitch?

Adding the lid angle back one to one only works if the pitch follows the camera tilt one
to one. This records Vision's face pitch (VNDetectFaceRectanglesRequest) and the solved
pitch from head.py on the same camera frames and fits each against the lid angle, using
only frames where both found a face.

Run from the repository root after calibrate.py. Hold your head still, look at the
screen, and swing the lid slowly between about 100 and 130 degrees a few times. q to stop.

    uv run --extra vision python analysis/vision_slope.py                  # record to vision_runN.npy (next free N), then fit
    uv run --extra vision python analysis/vision_slope.py vision_run1.npy  # fit an existing recording
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import head  # noqa: E402

import Vision  # noqa: E402
from Foundation import NSData  # noqa: E402


def vision_pitch(frame_bgr):
    """Face pitch in degrees from VNDetectFaceRectanglesRequest, or None."""
    ok, jpg = cv2.imencode(".jpg", frame_bgr)
    data = NSData.dataWithBytes_length_(jpg.tobytes(), len(jpg))
    request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    if not handler.performRequests_error_([request], None)[0]:
        return None
    faces = request.results() or []
    if not faces or faces[0].pitch() is None:
        return None
    return float(np.degrees(faces[0].pitch().floatValue()))


def record(path):
    from board import CAMERA, CAMERA_HELP
    from lid import Lid

    if head.CALIB is None:
        sys.exit("calib.npz not found: run calibrate.py first, from the repository root")
    lid, detector, cap = Lid(), head.landmarker(), cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        sys.exit(f"camera {CAMERA} unavailable: {CAMERA_HELP}")
    head.head_pitch.pnp = head.PnPPitch()
    rows, t0 = [], time.time()
    print("hold your head still and swing the lid slowly; q to stop")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if frame.shape[1::-1] != tuple(head.CALIB["size"]):
                sys.exit("the camera resolution differs from calib.npz: recalibrate with this camera")
            try:
                angle = lid.angle()
            except RuntimeError:
                continue
            v = vision_pitch(frame)
            found = head.head_pitch(detector, frame)
            pnp = None if found is None else found[2]
            rows.append((time.time() - t0, angle, np.nan if v is None else v, np.nan if pnp is None else pnp))
            text = f"lid {angle}  vision {'-' if v is None else f'{v:+.1f}'}  pnp {'-' if pnp is None else f'{pnp:+.1f}'}"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("vision_slope", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        lid.close()
        np.save(path, np.array(rows, dtype=float))
        print(f"saved {len(rows)} frames to {path}")


def fit(path):
    d = np.load(path)
    t, lid, vision, pnp = d.T
    print(f"{path}: {len(t)} frames, lid {np.nanmin(lid):.0f} to {np.nanmax(lid):.0f}")
    if np.nanmax(lid) - np.nanmin(lid) < 20:
        sys.exit("the lid moved less than 20 degrees: swing it further and record again")
    both = ~np.isnan(vision) & ~np.isnan(pnp)  # compare on the same frames
    print(f"frames with a face: vision {(~np.isnan(vision)).sum()}, solved {(~np.isnan(pnp)).sum()}, both {both.sum()}")
    if both.sum() < 20:
        sys.exit("too few frames where both found a face (no calib.npz, or no face in view?)")
    for name, y in (("vision", vision), ("pnp", pnp)):
        m = both
        slope, icpt = np.polyfit(lid[m], y[m], 1)
        resid = y[m] - (slope * lid[m] + icpt)
        print(f"{name}: {slope:+.2f} degrees per lid degree (residual SD {resid.std():.1f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fit(sys.argv[1])
    else:
        n = 1
        while Path(f"vision_run{n}.npy").exists():
            n += 1
        out = f"vision_run{n}.npy"  # never overwrite an earlier run
        record(out)
        fit(out)
