import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

import flutter as F
import real

# deblur real coded captures. Run through global-shutter.py, rolling-shutter-per-frame.py or
# rolling-shutter-per-row.py, which differ only in which exposure window each sample is given:
# global: every line the same; rs-frame: every line the middle line's; rs-row: every line its own
METHODS = ("global", "rs-frame", "rs-row")


def scene_for(calib, shape, t_exp, chip, readout, omega):
    """grid for one capture: the calibrated rings and arc, sub-frames short enough that the outer ring
    moves under half a pixel and a code bit is cut 16 ways, covering the exposure of the last line"""
    r0, r1 = calib["radius_px"]
    dt = min(chip / 16, 0.5 / (abs(omega) * r1))
    t = (np.arange(int(np.ceil((t_exp + readout) / dt))) + 0.5) * dt
    a0, a1 = np.radians(calib["arc_deg"])
    return F.grid(calib["hub"], np.arange(r0, r1 + 1.0), a0, a1, 1 / r1, shape, calib["readout_axis"], t, dt, omega)


def brightness(scene, code, chip, t_exp, monitor):
    """monitor brightness at the sub-frames: the code through the calibrated black level, timing and LCD
    lag, scaled so a still pixel deblurs to its own value in a still capture of the same exposure"""
    L = F.light(scene.t, code, chip, monitor.get("black", 0.0), monitor.get("t0", 0.0))
    if monitor.get("tau"):
        L = F.lcd(L, scene.dt, monitor["tau"])
    return L / F.weights(scene.t, scene.dt, L, t_exp, [0.0]).sum()


def deblur(img, calib, code, chip, t_exp, rpm, method, solver="dpir", lam=1e-2, find_speed=False):
    """(deblurred image, scene, rpm used). img: linear capture in raw sensor orientation"""
    readout = 0.0 if method == "global" else calib["readout_s"]
    if method != "global" and not readout:
        raise SystemExit("rolling shutter methods need readout_s in the calibration: run calibrate.py readout")
    scene = scene_for(calib, img.shape, t_exp, chip, readout, rpm * 2 * np.pi / 60)
    L = brightness(scene, code, chip, t_exp, calib.get("monitor", {}))
    kw = dict(t_line=readout / scene.n_lines, per_row=method == "rs-row", ref_line=scene.n_lines // 2)
    print(f"  {len(scene.r)} rings x {len(scene.th)} arc samples, {len(scene.t)} sub-frames of {scene.dt * 1e3:.2f} ms")
    P, lines = F.unwrap(scene, img)
    if find_speed:
        est = F.speed(scene, P, lines, L, t_exp, scene.omega * np.linspace(0.8, 1.2, 21), **kw)[0]
        print(f"  blind speed {est * 60 / (2 * np.pi):.3f} rpm (commanded {rpm:.3f})")
        scene.omega = est
    if solver == "dpir":
        x = F.dpir(*F.forward_model(scene, L, t_exp, lines, **kw), P)
    else:
        x = F.tikhonov(scene, P, lines, L, t_exp, lam, **kw)
    return F.wrap(scene, x), scene, scene.omega * 60 / (2 * np.pi)


def outline(ax, calib):
    """the calibrated hub and arc, drawn on an image axis"""
    (r0, r1), (a0, a1), (hr, hc) = calib["radius_px"], np.radians(calib["arc_deg"]), calib["hub"]
    a = np.linspace(a0, a1, 100)
    r = np.r_[np.full(100, r0), np.full(100, r1), r0]
    th = np.r_[a, a[::-1], a0]
    ax.plot(hc + r * np.cos(th), hr - r * np.sin(th), "r-", lw=1)
    ax.plot(hc, hr, "r+", ms=12)


def compare(x, scene, ref):
    """(reference turned and scaled onto x, PSNR, SSIM, PSNR by band of sensor lines) over the blade.
    The still is turned to where the deblurred blade is (real.align) and given one gain fitted to x, as
    the two are exposed differently; units are fractions of white, as in the simulation"""
    turned = real.align(ref, x, scene.hub, scene.r, scene.th)[0]
    window = F.wrap(scene, np.ones((len(scene.r), len(scene.th)))) > 0.5
    turned = turned * (x[window] @ turned[window]) / (turned[window] @ turned[window])
    scene.truth = turned
    scene.mask = window & binary_dilation(turned > 0.1 * np.percentile(turned[window], 99), iterations=5)
    return turned, *F.score(scene, x), F.bands(scene, x)


def show(img, ax, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=max(np.percentile(img, 99.5), 1e-6))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main(method):
    ap = argparse.ArgumentParser(description=f"deblur real coded captures ({method}). Each DNG needs the json "
                                 "run log coded_illumination.py saved next to it")
    ap.add_argument("dng", nargs="+", help="coded capture(s), deblurred one by one")
    ap.add_argument("--calib", required=True, help="this camera's calibration json (calibrate.py)")
    ap.add_argument("--rpm", type=float, required=True,
                    help="commanded fan speed; + = counter-clockwise as the raw image is displayed")
    ap.add_argument("--dark", nargs="*", default=[], help="dark frames at the same exposure and gain, subtracted")
    ap.add_argument("--solver", choices=("dpir", "tikhonov"), default="dpir")
    ap.add_argument("--lam", type=float, default=1e-2, help="tikhonov regularisation weight")
    ap.add_argument("--find-speed", action="store_true",
                    help="estimate the speed blind (within 20%% of --rpm) and deblur at the estimate")
    ap.add_argument("--reference", help="still of the stopped blade (constant light): reports PSNR, SSIM and PSNR "
                    "by band of sensor lines, and mean +- std over all the captures")
    ap.add_argument("--check", action="store_true", help="only draw the calibrated hub and arc on each capture")
    ap.add_argument("--out", default="out", help="output folder")
    args = ap.parse_args()
    if args.rpm == 0:
        ap.error("--rpm must not be 0")

    calib = json.loads(Path(args.calib).read_text())
    dark = real.mean_frames(real.files(args.dark)) if args.dark else 0.0
    ref = real.load_dng(args.reference) - dark if args.reference else None
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    scores, band_scores, speeds = [], [], []
    for path in map(Path, real.files(args.dng)):
        img = real.load_dng(path) - dark
        stem = out / f"{path.stem}_{method}"
        if args.check:
            fig, ax = plt.subplots(figsize=(8, 6))
            show(img, ax, f"{path.name}: hub and arc to deblur")
            outline(ax, calib)
            ax.set_xlim(0, img.shape[1]), ax.set_ylim(img.shape[0], 0)
            fig.savefig(f"{stem}_check.png", dpi=100, bbox_inches="tight")
            print(f"{path.name}: {stem}_check.png")
            continue
        log = path.with_suffix(".json")
        if not log.exists():
            raise SystemExit(f"{log} not found: the run log coded_illumination.py writes next to the DNG")
        code, chip, t_exp = real.run_log(log)
        print(f"{path.name}: {len(code)} bits of {chip * 1e3:.1f} ms, exposure {t_exp:.3f} s, {method}, {args.solver}")
        x, scene, rpm = deblur(img, calib, code, chip, t_exp, args.rpm, method, args.solver, args.lam, args.find_speed)
        np.save(f"{stem}.npy", x)
        speeds.append(rpm)
        fig, ax = plt.subplots(1, 3 if ref is not None else 2, figsize=(21 if ref is not None else 14, 5))
        show(img, ax[0], f"{path.name} (capture)")
        show(np.clip(x, 0, None), ax[1], f"deblurred, {method}, {args.solver}, {rpm:.3f} rpm")
        outline(ax[0], calib)
        if ref is not None:
            truth, p, q, b = compare(x, scene, ref)
            scores.append((p, q))
            band_scores.append(b)
            show(truth, ax[2], "reference, turned to the blade")
            print(f"  PSNR {p:.2f} dB, SSIM {q:.3f}, by band of sensor lines " + " ".join(f"{v:5.1f}" for v in b))
        fig.savefig(f"{stem}.png", dpi=100, bbox_inches="tight")
        plt.close("all")
        print(f"  {stem}.png, {stem}.npy")
    if len(speeds) > 1 and args.find_speed:
        print(f"blind speed over {len(speeds)} captures: {np.mean(speeds):.3f} +- {np.std(speeds):.3f} rpm "
              f"(commanded {args.rpm:.3f})")
    if len(scores) > 1:
        p, q = np.array(scores).T
        b = np.array(band_scores)
        print(f"over {len(scores)} captures: PSNR {p.mean():.2f} +- {p.std():.2f} dB, SSIM {q.mean():.3f} +- {q.std():.3f}")
        print("PSNR by band, mean   " + " ".join(f"{v:5.1f}" for v in np.nanmean(b, 0)))
        print("PSNR by band, std    " + " ".join(f"{v:5.1f}" for v in np.nanstd(b, 0)))
