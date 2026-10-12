"""Render a demo GIF from a recorded run (no camera image: only the face landmarks).

    uv run python demo/make_gif.py head_marks.npz demo/lidpose.gif --start 3 --end 23

head_marks.npz is what head.py saves when you quit it.
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import head  # noqa: E402

W, H = 960, 540
BG, GRID, TEXT, DIM = (24, 24, 28), (60, 60, 68), (235, 235, 240), (140, 140, 150)
RED, GREEN, BLUE = (80, 90, 240), (110, 210, 120), (230, 170, 90)  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONN = mp.tasks.vision.FaceLandmarksConnections
TESS = [(c.start, c.end) for c in CONN.FACE_LANDMARKS_TESSELATION]
CONTOURS = [(c.start, c.end) for c in CONN.FACE_LANDMARKS_CONTOURS]


def series(path):
    d = np.load(path)
    t, lid, marks = d["t"], d["lid"].astype(float), d["marks"]
    if "K" in d:  # the camera calibration saved with the recording
        calib = {"K": d["K"], "dist": d["dist"]}
    elif head.CALIB is not None and tuple(head.CALIB["size"]) == tuple(d["size"]):
        calib = head.CALIB  # older recordings: assume the current calib.npz is the one used
        print("warning: the recording has no calibration; using calib.npz, which must be the one it was recorded with")
    else:
        sys.exit("the recording has no camera calibration and calib.npz does not match its resolution")
    pnp, out = head.PnPPitch(calib), []
    for m in marks:
        v = pnp(m)
        out.append((np.nan, np.nan) if v is None else (v, pnp.distance))
    pitch, dist = np.array(out).T
    ok = ~np.isnan(pitch)
    if not ok.any():
        sys.exit(f"{path} has no frame with a solved head pitch")
    i = np.where(ok)[0][0]
    camera = pitch - pitch[i]
    desk = head.corrected(camera, lid, lid[i], dist / dist[i])
    return t[ok] - t[i], lid[ok], camera[ok], desk[ok], marks[ok], tuple(d["size"])


def text(img, s, xy, scale=0.6, color=TEXT, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def draw_face(img, marks, view, box):
    """Draw landmarks inside box, mapping the webcam region `view` (x, y, w, h in pixels)."""
    x0, y0, w, h = box
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), GRID, 1)
    vx, vy, vw, vh = view
    p = (marks[:, :2] - [vx, vy]) / [vw, vh] * [w, h] + [x0, y0]
    for a, b in TESS:
        cv2.line(img, tuple(p[a].astype(int)), tuple(p[b].astype(int)), (70, 70, 80), 1, cv2.LINE_AA)
    for a, b in CONTOURS:
        cv2.line(img, tuple(p[a].astype(int)), tuple(p[b].astype(int)), TEXT, 1, cv2.LINE_AA)
    text(img, "webcam view, cropped (face landmarks only)", (x0, y0 + h + 22), 0.5, DIM)


def draw_laptop(img, angle, origin):
    ox, oy = origin
    base = 150
    cv2.line(img, (ox, oy), (ox + base, oy), DIM, 6, cv2.LINE_AA)
    a = np.radians(angle)  # lid angle measured from the base, which points right
    tip = (int(ox + 130 * np.cos(a)), int(oy - 130 * np.sin(a)))
    cv2.line(img, (ox, oy), tip, BLUE, 6, cv2.LINE_AA)
    cv2.circle(img, (int(ox + 115 * np.cos(a)), int(oy - 115 * np.sin(a))), 6, TEXT, -1, cv2.LINE_AA)
    text(img, f"lid {angle:.0f} deg", (ox + 40, oy + 32), 0.7, BLUE, 2)


def draw_plot(img, t, camera, desk, now, box, span):
    x0, y0, w, h = box
    lo, hi = -40, 40
    to_y = lambda v: int(y0 + (hi - np.clip(v, lo, hi)) / (hi - lo) * h)
    to_x = lambda s: int(x0 + s / span * w)
    for v in (-30, -15, 0, 15, 30):
        cv2.line(img, (x0, to_y(v)), (x0 + w, to_y(v)), GRID if v else DIM, 1)
        text(img, f"{v:+d}", (x0 - 42, to_y(v) + 5), 0.45, DIM)
    for s in range(0, int(span) + 1, 5):
        text(img, f"{s}s", (to_x(s) - 8, y0 + h + 18), 0.4, DIM)
    cv2.line(img, (x0 + 330, y0 - 17), (x0 + 355, y0 - 17), RED, 2)
    text(img, "head pitch, raw", (x0 + 362, y0 - 12), 0.45, RED)
    cv2.line(img, (x0 + 540, y0 - 17), (x0 + 565, y0 - 17), GREEN, 2)
    text(img, "head pitch, lid removed", (x0 + 572, y0 - 12), 0.45, GREEN)
    m = t <= now
    for values, color in ((camera, RED), (desk, GREEN)):
        pts = np.c_[[to_x(s) for s in t[m]], [to_y(v) for v in values[m]]].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
    text(img, "degrees vs start (+ = head down)", (x0, y0 - 12), 0.5, DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--fps", type=float, default=12)
    args = ap.parse_args()
    if not 0 < args.fps <= 60:
        sys.exit("--fps must be between 0 and 60")
    if not Path(args.run).exists():
        sys.exit(f"{args.run} not found: head.py writes head_marks.npz when you quit it")

    t, lid, camera, desk, marks, size = series(args.run)
    keep = (t >= args.start) & (t <= args.end)
    if keep.sum() < 2:
        sys.exit(f"no frames between {args.start} and {args.end} s; the recording is {t[-1]:.1f} s long")
    t, lid, camera, desk, marks = t[keep] - args.start, lid[keep], camera[keep], desk[keep], marks[keep]
    span = t[-1]
    face_h = np.median(np.ptp(marks[:, :, 1], axis=1))
    centre = np.median(marks[:, :, :2].reshape(-1, 2), axis=0)
    vh = face_h * 3  # room for the face to slide as the lid moves
    vw = vh * 384 / 216
    view = (centre[0] - vw / 2, centre[1] - vh / 2, vw, vh)
    frames = []
    for now in np.arange(0, span, 1 / args.fps):
        i = max(np.searchsorted(t, now, side="right") - 1, 0)  # never wrap to the last frame
        img = np.full((H, W, 3), BG, np.uint8)
        text(img, "Opening the lid tilts the webcam. lidpose subtracts the tilt.", (30, 40), 0.75, TEXT, 2)
        draw_face(img, marks[i], view, (30, 70, 384, 216))
        draw_laptop(img, lid[i], (520, 250))
        text(img, f"raw pitch    {camera[i]:+5.1f}", (720, 150), 0.65, RED, 2)
        text(img, f"head pitch   {desk[i]:+5.1f}", (720, 190), 0.65, GREEN, 2)
        text(img, "red: as the webcam sees it", (720, 215), 0.45, DIM)
        text(img, "green: lid tilt removed", (720, 235), 0.45, DIM)
        draw_plot(img, t, camera, desk, now, (80, 345, 850, 170), span)
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).quantize(colors=64))
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"{args.out}: {len(frames)} frames, {span:.1f} s")


if __name__ == "__main__":
    main()
