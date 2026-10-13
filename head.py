"""Head pitch from the webcam, before and after correcting for the lid angle.

The webcam sits in the lid, so opening the lid tilts the camera about one degree per
lid degree (the face's direction from the camera: about 0.99 per degree while the lid
swung in runs 2 to 4; the solved
head pitch: 0.97 to 1.09 per degree in three runs by one person, see analysis/table.py) and a still head
appears to pitch by the same amount. Head pitch is solved from all face landmarks with the calibrated camera;
adding back the lid change since the start gives pitch relative to the desk.
Both are relative: zero is whatever posture you hold when the script starts, not an
upright posture.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from board import CAMERA, CAMERA_HELP
from lid import Lid

MODEL = Path("models/face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
def _env_float(name, default):
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        value = float("nan")
    if not np.isfinite(value):
        raise SystemExit(f"{name} must be a finite number")
    return value


SLOPE = _env_float("LIDTILT_SLOPE", 1.0)  # camera pitch degrees per lid degree
if SLOPE <= 0:
    raise SystemExit("LIDTILT_SLOPE takes the size of measure.py's slope, a positive number such as 0.99")
# Experimental, off by default. The pitch reads too high as the face nears the camera:
# +10 degrees at 0.65 of the starting distance with the head still and the laptop pulled
# closer (one run, one person), fitted there as -27.5 * log(distance ratio). It also
# shrinks a real lean toward the screen, and its size depends on where the face sits in
# the frame, so it is not a constant to trust yet. Try LIDTILT_DIST_GAIN=-27.5.
DIST_GAIN = _env_float("LIDTILT_DIST_GAIN", 0.0)


def landmarker():
    if not MODEL.exists():
        MODEL.parent.mkdir(exist_ok=True)
        print(f"downloading the face landmarker model to {MODEL} ...", flush=True)
        part = MODEL.with_suffix(".part")
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response, open(part, "wb") as out:
            out.write(response.read())
        part.rename(MODEL)  # a cut-off download never leaves a broken model behind
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


CALIB = np.load("calib.npz") if Path("calib.npz").exists() else None
TEMPLATE_FRAMES = 15  # the first frames with a face set the face shape and the neutral posture


class PnPPitch:
    """Head pitch from all 478 landmarks, solved with the calibrated camera.

    MediaPipe's own pitch follows only part of a camera tilt (0.32 to 0.45 per degree
    over runs 1 to 3). Solving the pose with the real intrinsics against the user's own face
    shape follows it at 0.97 to 1.09 per degree, close enough to add the lid angle back one
    to one; a few degrees can remain.
    """

    def __init__(self, calib=None):
        self.calib = calib if calib is not None else CALIB
        self.frames, self.template, self.prev = [], None, None

    def __call__(self, marks):
        if self.calib is None:
            return None
        if self.template is None:
            self.frames.append(marks)
            if len(self.frames) < TEMPLATE_FRAMES:
                return None
            t = np.median(self.frames, axis=0).astype(np.float64)
            self.template = np.ascontiguousarray(t - t.mean(0))
        image = np.ascontiguousarray(marks[:, :2], dtype=np.float64)
        if self.prev is None:  # EPnP alone was 25 degrees off in one run; refine it like later frames
            ok, rvec, tvec = cv2.solvePnP(self.template, image, self.calib["K"], self.calib["dist"], flags=cv2.SOLVEPNP_EPNP)
            self.prev = (rvec, tvec)
        ok, rvec, tvec = cv2.solvePnP(
            self.template, image, self.calib["K"], self.calib["dist"],
            rvec=self.prev[0].copy(), tvec=self.prev[1].copy(), useExtrinsicGuess=True,
        )
        if not ok:
            return None
        self.prev = (rvec, tvec)
        self.distance = float(np.ravel(tvec)[2])
        # Tilt of the face's vertical axis toward the camera. An Euler pitch mixes in a
        # head turn (26 degrees at a 55 degree turn in one run; this cut it to 16).
        down = cv2.Rodrigues(rvec)[0][:, 1]
        return float(np.degrees(np.arctan2(down[2], np.hypot(down[0], down[1]))))


def head_pitch(detector, frame_bgr):
    """(head pitch from the landmarker in degrees, positive = chin down; nose y in pixels), or None."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(image)
    if not result.facial_transformation_matrixes:
        return None
    rotation = np.array(result.facial_transformation_matrixes[0])[:3, :3]
    pitch, _, _ = cv2.RQDecomp3x3(rotation)[0]
    h, w = frame_bgr.shape[:2]
    marks = result.face_landmarks[0]
    head_pitch.last = np.array([[m.x * w, m.y * h, m.z * w] for m in marks], dtype=np.float32)
    return pitch, marks[1].y * h, head_pitch.pnp(head_pitch.last)


def corrected(pitch, lid_angle, lid_start, distance_ratio=1.0):
    """Pitch relative to the desk, zero at the starting posture: add back how far the lid
    has tilted the camera since the start (SLOPE per lid degree; measured 0.97 to 1.09), and
    remove the reading's drift with distance to the camera."""
    if not DIST_GAIN:  # skip the log: a flipped pose can give a ratio <= 0
        return pitch + SLOPE * (lid_angle - lid_start)
    return pitch + SLOPE * (lid_angle - lid_start) - DIST_GAIN * np.log(distance_ratio)


if __name__ == "__main__":
    if CALIB is None:
        sys.exit("calib.npz not found: run calibrate.py first (see board.py for the board)")
    try:
        lid = Lid()
    except RuntimeError as e:
        sys.exit(f"{e}; see Requirements in README.md for Macs without the sensor")
    try:
        detector = landmarker()
    except OSError as e:
        sys.exit(f"could not download the face landmarker model ({e}); check the network and run again")
    cap = cv2.VideoCapture(CAMERA)
    head_pitch.pnp, lid_start, pnp_start = PnPPitch(), None, None
    if not cap.isOpened():
        sys.exit(f"camera {CAMERA} unavailable: {CAMERA_HELP}")
    print("q to quit; readings go to head.csv")
    log = open("head.csv", "w")
    log.write("t,lid,camera_pitch,nose_y,pnp_pitch\n")
    t0 = time.time()
    record = {"t": [], "lid": [], "marks": []}  # all landmarks, for trying pose methods offline
    try:
        misses = lid_misses = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                misses += 1
                if misses > 100:
                    sys.exit("camera stopped sending frames")
                continue
            misses = 0
            if frame.shape[1::-1] != tuple(CALIB["size"]):
                sys.exit(f"camera gives {frame.shape[1]}x{frame.shape[0]} but calib.npz is for {CALIB['size'][0]}x{CALIB['size'][1]}: recalibrate with this camera")
            try:
                angle = lid.angle()
                lid_misses = 0
            except RuntimeError as e:  # lid closed or an odd reading: skip the frame, keep the window alive
                lid_misses += 1
                if lid_misses > 300:
                    sys.exit(f"lid sensor unreadable for 300 frames: {e}")
                cv2.putText(frame, "lid sensor unreadable", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                cv2.imshow("lidtilt", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            found = head_pitch(detector, frame)
            pitch = None if found is None else found[0]
            nose_y = "" if found is None else f"{found[1]:.1f}"
            pnp = None if found is None else found[2]
            if found is not None:
                record["t"].append(time.time() - t0)
                record["lid"].append(angle)
                record["marks"].append(head_pitch.last)
            log.write(f"{time.time() - t0:.3f},{angle},{'' if pitch is None else f'{pitch:.2f}'},{nose_y},{'' if pnp is None else f'{pnp:.2f}'}\n")
            lines = [f"lid {angle} deg"]
            if pitch is not None:
                if pnp is None:
                    lines += ["hold still: learning your face"]
                else:
                    if lid_start is None:
                        lid_start, pnp_start, dist_start = angle, pnp, head_pitch.pnp.distance
                    ratio = head_pitch.pnp.distance / dist_start
                    lines += [
                        f"head pitch raw {pnp - pnp_start:+.1f} (vs start, + = down)",
                        f"head pitch     {corrected(pnp - pnp_start, angle, lid_start, ratio):+.1f} (lid removed)",
                    ]
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (20, 50 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
            cv2.imshow("lidtilt", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:  # keep the recording however the loop ends (q, Ctrl+C, an error)
        log.close()
        if record["t"]:
            np.savez_compressed(
                "head_marks.npz", t=record["t"], lid=record["lid"], marks=np.stack(record["marks"]),
                size=CALIB["size"], K=CALIB["K"], dist=CALIB["dist"],
            )
        cap.release()
        cv2.destroyAllWindows()
        lid.close()
