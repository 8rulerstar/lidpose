"""Read the MacBook lid angle sensor (Apple sensor hub, HID orientation usage)."""

import hid

VENDOR, PRODUCT = 0x05AC, 0x8104
USAGE_PAGE, USAGE = 0x20, 0x8A


class Lid:
    def __init__(self):
        found = [
            d
            for d in hid.enumerate(VENDOR, PRODUCT)
            if d.get("usage_page") == USAGE_PAGE and d.get("usage") == USAGE
        ]
        if not found:
            raise RuntimeError("lid angle sensor not found on this Mac")
        self.dev = hid.device()
        try:
            self.dev.open_path(found[0]["path"])
        except OSError as e:
            raise RuntimeError(f"could not open the lid angle sensor: {e}") from e

    def angle(self) -> int:
        """Degrees between the display and the base, in whole degrees."""
        try:
            r = self.dev.get_feature_report(1, 8)
        except OSError as e:  # hidapi reports read failures as OSError
            raise RuntimeError(f"lid sensor read failed: {e}") from e
        if len(r) < 3:
            raise RuntimeError(f"unexpected lid sensor report: {list(r)}")
        angle = r[1] | (r[2] << 8)
        if not 0 < angle <= 180:
            raise RuntimeError(f"lid sensor gave {angle} degrees; this Mac may report it differently")
        return angle

    def close(self):
        self.dev.close()


if __name__ == "__main__":
    lid = Lid()
    print(lid.angle())
    lid.close()
