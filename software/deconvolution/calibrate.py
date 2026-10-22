import argparse
import json
from pathlib import Path

import numpy as np

import real

# one calibration json per camera, everything in raw sensor orientation and in pixels of the image
# real.load_dng returns (half size for a colour sensor):
#   hub [row, col]           rotation centre
#   radius_px [r0, r1]       rings to deblur
#   arc_deg [a0, a1]         angles to deblur, counter-clockwise from +col as displayed: the blade and its smear.
#                            "auto" (what `auto` writes): found in each capture, the brightest blade and its smear
#   readout_s                first to last line, 0 for global shutter; measured by `readout`
#   readout_axis             0 = sensor lines are image rows
#   monitor {black, tau, t0} black level and LCD lag (s) from `monitor`; t0, when the code starts after the
#                            (first line's) exposure opens, from `readout` on a rolling shutter. On a global
#                            shutter t0 only turns the result, so the edge fit's value is kept


def load(path):
    return json.loads(Path(path).read_text())


def save(path, calib):
    Path(path).write_text(json.dumps(calib, indent=2) + "\n")
    print(f"{path}: {json.dumps(calib)}")


def turn_deg(a):
    """the turn between the two stills from --deg, or --steps and --steps-per-rev"""
    if (a.deg is None) == (a.steps is None) or (a.steps is not None and not a.steps_per_rev):
        raise SystemExit("give the turn as --deg, or as --steps with --steps-per-rev")
    return a.deg if a.deg is not None else a.steps * 360 / a.steps_per_rev


def arc_rad(c, r, img):
    """angles (rad) for the calibration's arc; "auto": found in img, or the whole circle without one"""
    if c["arc_deg"] != "auto":
        return np.arange(*np.radians(c["arc_deg"]), 1 / r[-1])
    if img is None:
        return np.arange(-np.pi, np.pi, 1 / r[-1])
    return np.arange(*np.radians(real.find_arc(img, c["hub"], r)), 1 / r[-1])


def auto(a):
    """the whole calibration from two stills of the stopped fan, no guess: hub, blade rings and which way
    the fan turned. The arc is left "auto", found in each capture. Draws what it found, to check"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f0, f1 = real.load_dng(a.still0), real.load_dng(a.still1)
    try:
        (hr, hc), sign, rms = real.hub_from_stills(f0, f1, np.radians(abs(turn_deg(a))))
    except ValueError as err:
        raise SystemExit(str(err))
    r0, r1 = real.radii_from_stills(f0, f1, (hr, hc))
    m = real.moved(f0, f1)
    unexplained = rms / np.sqrt(np.mean((f1 - f0)[m] ** 2))
    print(f"hub (row, col) ({hr:.2f}, {hc:.2f}), blade {r0:.0f} to {r1:.0f} px from it. From still0 to still1 the fan "
          f"turned {'counter-clockwise' if sign > 0 else 'clockwise'} as displayed: if that is the stepper's forward "
          f"direction, --rpm is {'+' if sign > 0 else '-'}. {100 * unexplained:.0f}% of the change between the stills "
          f"is left unexplained")
    if unexplained > 0.5:
        print("*** that is a poor fit: check the stills are of the same setup and the turn (--steps, --steps-per-rev)")
    save(a.calib, dict(hub=[round(hr, 2), round(hc, 2)], radius_px=[r0, r1], arc_deg="auto", readout_s=a.readout,
                       readout_axis=a.axis, monitor=dict(black=0.0, tau=0.0, t0=0.0)))
    Path(a.out).mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10 * f1.shape[0] / f1.shape[1]))
    ax.imshow(f1, cmap="gray", vmin=0, vmax=max(np.percentile(f1, 99.5), 1e-6))
    tt = np.linspace(0, 2 * np.pi, 400)
    for rr in (r0, r1):
        ax.plot(hc + rr * np.cos(tt), hr - rr * np.sin(tt), "r-", lw=1)
    ax.plot(hc, hr, "r+", ms=14)
    ax.set_xlim(0, f1.shape[1]), ax.set_ylim(f1.shape[0], 0)
    ax.set_title("hub (+) and the rings to deblur: the blade should sit between the circles, the + on its axle")
    out = Path(a.out) / f"{Path(a.still1).stem}_auto.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"check it: {out}")


def grid(a):
    """each capture with a pixel grid, in the coordinates every value here is in, to read the first guess off"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(a.out).mkdir(exist_ok=True)
    for path in map(Path, real.files(a.dng)):
        img = real.load_dng(path)
        step = 50 if max(img.shape) > 800 else 25
        fig, ax = plt.subplots(figsize=(14, 14 * img.shape[0] / img.shape[1]))
        ax.imshow(img, cmap="gray", vmin=0, vmax=max(np.percentile(img, 99.5), 1e-6))
        ax.set_xticks(np.arange(0, img.shape[1], step))
        ax.set_yticks(np.arange(0, img.shape[0], step))
        ax.tick_params(labelsize=7, labeltop=True, labelright=True)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.grid(color="c", alpha=0.4, lw=0.5)
        ax.set_xlabel("col"), ax.set_ylabel("row")
        out = Path(a.out) / f"{path.stem}_grid.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"{out}: read the hub (row, col) and blade radii (px) off this")


def init(a):
    save(a.calib, dict(hub=a.hub, radius_px=a.radius, arc_deg=a.arc, readout_s=a.readout, readout_axis=a.axis,
                       monitor=dict(black=0.0, tau=0.0, t0=0.0)))


def hub(a):
    c = load(a.calib)
    f0, f1 = real.load_dng(a.still0), real.load_dng(a.still1)
    r = np.arange(c["radius_px"][0], c["radius_px"][1] + 1.0)
    th = arc_rad(c, r, None)
    found = real.find_hub(f0, f1, np.radians(turn_deg(a)), c["hub"], r, th)
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
    th = arc_rad(c, r, img)
    dt = chip / 16
    t = (np.arange(int(np.ceil(t_exp / dt))) + 0.5) * dt
    L = real.kernel_from_edge(img, c["hub"], r, th, a.rpm * 2 * np.pi / 60, t)
    (black, tau, t0), _ = real.fit_monitor(L, t, dt, code, chip)
    print(f"black level {black:.3f}, LCD lag {tau * 1e3:.2f} ms")
    bad = real.check_bits(L, t, code, chip, black, t0)
    if bad:
        print(f"*** bits {bad} of {len(code)} did not show as coded (a refresh doubled or skipped one), or the arc "
              f"stops short of the end of the smear. Fix that before deblurring anything")
    else:
        print(f"all {len(code)} bits showed as coded")
    for path in a.calib:
        c = load(path)
        c["monitor"] = dict(black=round(black, 4), tau=float(tau), t0=float(t0))
        save(path, c)


def readout(a):
    """rolling shutter readout time and code start from a still scene under the code and under constant light"""
    c = load(a.calib)
    code, chip, t_exp = real.run_log(Path(a.coded).with_suffix(".json"))
    dark = real.mean_frames(real.files(a.dark)) if a.dark else 0.0
    m = c.get("monitor", {})
    ro, t0, rms, spread = real.fit_readout(real.load_dng(a.coded) - dark, real.load_dng(a.constant) - dark, code,
                                           chip, t_exp, c["readout_axis"], m.get("black", 0.0), m.get("tau", 0.0))
    print(f"readout {ro * 1e3:.2f} ms first to last line, code starts {t0 * 1e3:+.2f} ms after the first line opens, "
          f"fit rms {rms:.4f}, ratio spread over lines {spread:.3f}")
    if spread < 0.01:
        raise SystemExit("every line caught the same light: the exposure covers the whole code on every line, so "
                         "readout cannot be measured here, and per-row and per-frame will be the same anyway. "
                         "Expected with the data_capture GUI, which pads by the readout: skip this step, the deblur "
                         "takes the readout from the run log")
    if ro > 0.95 * real.READOUT_MAX or t0 > 0.95 * real.T0_MAX or rms > 0.2 * spread:
        raise SystemExit("the fit does not explain the stills (at the edge of its search, or a poor fit): check both "
                         "are of the same still scene with the same params, and that nothing moved. Not saved")
    c["readout_s"] = round(ro, 6)
    c["monitor"] = {**m, "t0": round(t0, 6)}
    save(a.calib, c)


ap = argparse.ArgumentParser(description="write and fill one calibration json per camera. Quick: auto, then "
                             "monitor. By hand: grid, init, check it with --check on a capture, hub, monitor. "
                             "readout only for rolling shutter captures not padded past the readout")
sub = ap.add_subparsers(required=True)
p = sub.add_parser("auto", help="whole calibration from two stills of the stopped fan, turned between them")
p.add_argument("calib")
p.add_argument("still0")
p.add_argument("still1")
p.add_argument("--steps", type=float, help="the turn in stepper steps (sign does not matter, it is found)")
p.add_argument("--steps-per-rev", type=float,
               help="with --steps: steps per revolution in the driver's step mode (28BYJ-48: about 2038 full steps)")
p.add_argument("--deg", type=float, help="or the turn in degrees")
p.add_argument("--readout", type=float, default=0.0, help="first to last line, s; 0 = global shutter or from the run log")
p.add_argument("--axis", type=int, default=0, choices=(0, 1), help="0 = sensor lines are image rows (raw)")
p.add_argument("--out", default="out", help="folder for the check image")
p.set_defaults(run=auto)
p = sub.add_parser("grid", help="save each capture with a pixel grid (<name>_grid.png) to read the first guess off")
p.add_argument("dng", nargs="+")
p.add_argument("--out", default="out", help="output folder")
p.set_defaults(run=grid)
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
p.add_argument("--deg", type=float, help="turn from still0 to still1, + = counter-clockwise as displayed")
p.add_argument("--steps", type=float, help="or the turn in stepper steps, + = counter-clockwise as displayed")
p.add_argument("--steps-per-rev", type=float,
               help="with --steps: steps per revolution in the driver's step mode (28BYJ-48: about 2038 full steps)")
p.set_defaults(run=hub)
p = sub.add_parser("monitor", help="fit the monitor from coded captures of a plain blade (global shutter)")
p.add_argument("dng", nargs="+", help="coded captures of a blade with nothing printed near its leading edge, "
               "each started from the same blade angle (they are averaged), or just one")
p.add_argument("--calib", nargs="+", required=True, help="calibration json(s) to write it into, first one is used")
p.add_argument("--rpm", type=float, required=True, help="+ = counter-clockwise as displayed")
p.add_argument("--dark", nargs="*", default=[])
p.set_defaults(run=monitor)
p = sub.add_parser("readout", help="rolling shutter: readout time and code start from a still scene")
p.add_argument("calib")
p.add_argument("coded", help="still scene (fan stopped) under the code; its run log json next to it")
p.add_argument("constant", help="the same scene under constant light, same exposure and gain")
p.add_argument("--dark", nargs="*", default=[])
p.set_defaults(run=readout)
a = ap.parse_args()
a.run(a)
