"""ChArUco board shared by calibration and measurement. Square size is arbitrary:
rotations do not depend on it, so the board can be shown on any screen."""

import os

import cv2

try:  # built-in camera index; an iPhone (Continuity Camera) may take 0
    CAMERA = int(os.environ.get("LIDTILT_CAMERA", 0))
    if CAMERA < 0:
        raise ValueError
except ValueError:
    raise SystemExit("LIDTILT_CAMERA must be a camera index such as 0 or 1") from None
CAMERA_HELP = (
    "allow the camera for Terminal in System Settings > Privacy & Security > Camera, quit and reopen "
    "Terminal, and run again; if an iPhone took camera 0 (Continuity Camera), set LIDTILT_CAMERA=1"
)

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((5, 7), 1.0, 0.7, DICT)
DETECTOR = cv2.aruco.CharucoDetector(BOARD)


def detect(gray):
    """Return (object points, image points) of the visible chessboard corners, or None."""
    corners, ids, _, _ = DETECTOR.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj, img = BOARD.matchImagePoints(corners, ids)
    return obj, img


if __name__ == "__main__":
    cv2.imwrite("board.png", BOARD.generateImage((1000, 1400), marginSize=40))
    print("wrote board.png")
