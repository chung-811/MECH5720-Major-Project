import argparse
import json
from pathlib import Path

import numpy as np

import real

# one calibration json per camera, everything in raw sensor orientation and in pixels of the image
# real.load_dng returns (half size for a colour sensor):
#   hub [row, col]           rotation centre
#   radius_px [r0, r1]       rings to deblur
#   arc_deg [a0, a1]         angles to deblur, counter-clockwise from +col as displayed: the blade and its smear
#   readout_s                first to last line, 0 for global shutter
#   readout_axis             0 = sensor lines are image rows
#   monitor {black, tau, t0} black level, LCD lag (s), code timing (s), from `monitor`


def load(path):
    return json.loads(Path(path).read_text())


def save(path, calib):
    Path(path).write_text(json.dumps(calib, indent=2) + "\n")
    print(f"{path}: {json.dumps(calib)}")


def init(a):
    save(a.calib, dict(hub=a.hub, radius_px=a.radius, arc_deg=a.arc, readout_s=a.readout, readout_axis=a.axis,
                       monitor=dict(black=0.0, tau=0.0, t0=0.0)))


def hub(a):
    c = load(a.calib)
    f0, f1 = real.load_dng(a.still0), real.load_dng(a.still1)
    r = np.arange(c["radius_px"][0], c["radius_px"][1] + 1.0)
    th = np.arange(*np.radians(c["arc_deg"]), 1 / r[-1])
    found = real.find_hub(f0, f1, np.radians(a.deg), c["hub"], r, th)
    print(f"hub moved {np.hypot(found[0] - c['hub'][0], found[1] - c['hub'][1]):.1f} px from the guess")
    c["hub"] = [round(float(v), 2) for v in found]
    save(a.calib, c)


def monitor(a):
    """black level, LCD lag and timing of the monitor from the leading edge of a plain blade"""
    c = load(a.calib[0])
    dng = real.files(a.dng)
    img = real.mean_frames(dng) - (real.mean_frames(real.files(a.dark)) if a.dark else 0.0)
    code, chip, t_exp = real.run_log(Path(dng[0]).with_suffix(".json"))
    r = np.arange(c["radius_px"][0], c["radius_px"][1] + 1.0)
    th = np.arange(*np.radians(c["arc_deg"]), 1 / r[-1])
    dt = chip / 16
    t = (np.arange(int(np.ceil(t_exp / dt))) + 0.5) * dt
    L = real.kernel_from_edge(img, c["hub"], r, th, a.rpm * 2 * np.pi / 60, t)
    (black, tau, t0), _ = real.fit_monitor(L, t, dt, code, chip)
    print(f"black level {black:.3f}, LCD lag {tau * 1e3:.2f} ms, code {t0 * 1e3:+.2f} ms late")
    for path in a.calib:
        c = load(path)
        c["monitor"] = dict(black=round(black, 4), tau=float(tau), t0=float(t0))
        save(path, c)


ap = argparse.ArgumentParser(description="write and fill one calibration json per camera. "
                             "Order: init, check it with --check on a capture, hub, monitor")
sub = ap.add_subparsers(required=True)
p = sub.add_parser("init", help="new calibration from a first guess read off a capture")
p.add_argument("calib")
p.add_argument("--hub", type=float, nargs=2, required=True, metavar=("ROW", "COL"))
p.add_argument("--radius", type=float, nargs=2, required=True, metavar=("R0", "R1"), help="px")
p.add_argument("--arc", type=float, nargs=2, required=True, metavar=("A0", "A1"),
               help="deg, counter-clockwise from +col, covering the blade and its smear")
p.add_argument("--readout", type=float, default=0.0, help="first to last line, s; 0 = global shutter")
p.add_argument("--axis", type=int, default=0, choices=(0, 1), help="0 = sensor lines are image rows (raw)")
p.set_defaults(run=init)
p = sub.add_parser("hub", help="refine the hub from two stills of the stopped blade, turned between them")
p.add_argument("calib")
p.add_argument("still0")
p.add_argument("still1")
p.add_argument("--deg", type=float, required=True,
               help="turn from still0 to still1, + = counter-clockwise as displayed (steps x 360 / steps per rev)")
p.set_defaults(run=hub)
p = sub.add_parser("monitor", help="fit the monitor from coded captures of a plain blade (global shutter)")
p.add_argument("dng", nargs="+", help="coded captures of a blade with nothing printed near its leading edge")
p.add_argument("--calib", nargs="+", required=True, help="calibration json(s) to write it into, first one is used")
p.add_argument("--rpm", type=float, required=True, help="+ = counter-clockwise as displayed")
p.add_argument("--dark", nargs="*", default=[])
p.set_defaults(run=monitor)
a = ap.parse_args()
a.run(a)
