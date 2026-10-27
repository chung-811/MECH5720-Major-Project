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
SIGN_MARGIN = 0.9   # switch direction only when the other sign's residual is under 0.9 x: in simulation the wrong
                    # sign's was 1.2 to 2.7 x the right one's (full and 0.1 x light, uneven light, 5 and 10 rpm)


def for_capture(calib, img):
    """the calibration with the arc for this capture: found in the capture when arc_deg is "auto" (the
    blade stops at a different angle every time), else as calibrated"""
    if calib["arc_deg"] != "auto":
        return calib
    r = np.arange(calib["radius_px"][0], calib["radius_px"][1] + 1.0)
    return {**calib, "arc_deg": list(real.find_arc(img, calib["hub"], r))}


def scene_for(calib, shape, t_exp, chip, readout, omega):
    """grid for one capture: the calibrated rings and arc, sub-frames short enough that the outer ring
    moves under half a pixel and a code bit is cut 16 ways, covering the exposure of the last line"""
    r0, r1 = calib["radius_px"]
    dt = min(chip / 16, 0.5 / (abs(omega) * r1))
    t = (np.arange(int(np.ceil((t_exp + readout) / dt))) + 0.5) * dt
    a0, a1 = np.radians(calib["arc_deg"])
    return F.grid(calib["hub"], np.arange(r0, r1 + 1.0), a0, a1, 1 / r1, shape, calib["readout_axis"], t, dt, omega)


def brightness(scene, code, chip, t_exp, monitor, t0=None):
    """monitor brightness at the sub-frames: the code through the calibrated black level, timing and LCD
    lag, scaled so a still pixel deblurs to its own value in a still capture of the same exposure.
    t0 overrides the calibrated code start"""
    L = F.light(scene.t, code, chip, monitor.get("black", 0.0), monitor.get("t0", 0.0) if t0 is None else t0)
    if monitor.get("tau"):
        L = F.lcd(L, scene.dt, monitor["tau"])
    return L / F.weights(scene.t, scene.dt, L, t_exp, [0.0]).sum()


def deblur(img, calib, code, chip, t_exp, rpm, method, solver="dpir", lam=1e-2, find_speed=False, logged=None):
    """(deblurred image, scene, rpm used). img: linear capture in raw sensor orientation. logged: the run
    log's timing (real.log_timing); rolling shutter methods use its readout and code start when the
    calibration has no measured readout (calibrate.py readout refuses captures padded past the readout)"""
    calib = for_capture(calib, img)
    readout, t0 = 0.0, None
    if method != "global":
        readout = calib["readout_s"]
        if not readout and logged:
            readout, t0 = logged["readout_s"], logged["t0"]
            print(f"  readout {readout * 1e3:.1f} ms and code start {t0 * 1e3:.1f} ms from the run log")
        if not readout:
            raise SystemExit("rolling shutter methods need the readout time: run calibrate.py readout, or capture "
                             "with the data_capture GUI, whose run log has it")
    scene = scene_for(calib, img.shape, t_exp, chip, readout, rpm * 2 * np.pi / 60)
    L = brightness(scene, code, chip, t_exp, calib.get("monitor", {}), t0)
    kw = dict(t_line=readout / scene.n_lines, per_row=method == "rs-row", ref_line=scene.n_lines // 2)
    print(f"  {len(scene.r)} rings x {len(scene.th)} arc samples, {len(scene.t)} sub-frames of {scene.dt * 1e3:.2f} ms")
    P, lines = F.unwrap(scene, img)
    Ps = F.unwrap(scene, img, F.SPEED_INTERP)[0]   # for the direction check and speed search (F.SPEED_INTERP)
    if np.isnan(lines).any():
        print(f"  {100 * np.isnan(lines).mean():.0f}% of the arc falls outside the image: left out of the fit")
    noise = F.blade_noise(img, F.wrap(scene, np.ones((len(scene.r), len(scene.th)))) > 0.5)
    sigma = None if noise is None else F.BLADE_NOISE_GAIN * noise
    if noise is None:
        print("  warning: too few bright blade pixels to measure the noise, so DPIR estimates it from the whole arc")
    right, wrong = F.residuals(scene, Ps, lines, L, t_exp, [scene.omega, -scene.omega], **kw)
    if wrong < SIGN_MARGIN * right:              # the other direction explains the capture clearly better
        print(f"  *** the fan turns the other way to --rpm: {-rpm:+g} rpm fits {right / wrong:.1f}x better, deblurring "
              f"at that. Check --rpm's sign (calibrate.py auto prints it)")
        scene.omega, rpm = -scene.omega, -rpm
    elif wrong < right:
        print(f"  warning: cannot tell which way the fan turns from this capture (the two fit within "
              f"{100 * (1 - SIGN_MARGIN):.0f}%), keeping --rpm's sign")
    if find_speed:
        grid = scene.omega * np.linspace(0.8, 1.2, 21)
        est, scores = F.speed(scene, Ps, lines, L, t_exp, grid, **kw)
        print(f"  blind speed {est * 60 / (2 * np.pi):.3f} rpm (commanded {rpm:.3f})")
        if np.argmin(scores) in (0, len(grid) - 1):
            print(f"  *** the blind speed is at the edge of its search ({0.8 * rpm:.3f} to {1.2 * rpm:.3f} rpm), so the "
                  f"true speed is outside it and this estimate is wrong: rerun with --rpm nearer the true speed")
        scene.omega = est
    if solver == "dpir":
        x = F.dpir(*F.forward_model(scene, L, t_exp, lines, **kw), P, sigma=sigma)
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
    if ref.shape != x.shape:
        raise SystemExit(f"the reference is {ref.shape[1]} x {ref.shape[0]} but the capture is {x.shape[1]} x "
                         f"{x.shape[0]}: take the reference still in the same camera mode as the coded captures")
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
                                 "run log the capture tool saved next to it")
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
    # what never moves (calibrate.py auto, 3+ stills): taken out of every capture and the reference, so the
    # arc finder sees only the blades and the deblur only what turns (real.remove_static)
    static = np.load(Path(args.calib).parent / calib["static"]) if calib.get("static") else None
    if ref is not None and static is not None:
        try:
            ref = real.remove_static(ref, static, calib["hub"], calib["radius_px"][1])[0]
        except ValueError as err:
            raise SystemExit(f"reference: {err}")
        print("scoring against the reference with the static scene taken out (the moving parts only)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores, band_scores, speeds = [], [], []
    for path in map(Path, real.files(args.dng)):
        img = real.load_dng(path) - dark
        stem = out / f"{path.stem}_{method}"
        try:
            if static is not None:
                img, a_static, _ = real.remove_static(img, static, calib["hub"], calib["radius_px"][1])
            calib_here = for_capture(calib, img)
        except ValueError as err:
            print(f"{path.name}: skipped, {err}")
            continue
        if args.check:
            fig, ax = plt.subplots(figsize=(8, 6))
            show(img, ax, f"{path.name}: hub and arc to deblur")
            outline(ax, calib_here)
            ax.set_xlim(0, img.shape[1]), ax.set_ylim(img.shape[0], 0)
            fig.savefig(f"{stem}_check.png", dpi=100, bbox_inches="tight")
            print(f"{path.name}: {stem}_check.png")
            continue
        log = path.with_suffix(".json")
        if not log.exists():
            raise SystemExit(f"{log} not found: the json run log the capture tool saves next to the DNG")
        code, chip, t_exp = real.run_log(log)
        timing = real.log_timing(log)
        if timing.get("aborted"):
            print(f"{path.name}: skipped, the run was stopped early (ESC)")
            continue
        print(f"{path.name}: {len(code)} bits of {chip * 1e3:.1f} ms, exposure {t_exp:.3f} s, {method}, {args.solver}")
        if static is not None:
            print(f"  static scene taken out (here {a_static:.2f} x as bright as in the stills)")
            if not 0.05 < a_static < 1.5:
                print("  *** that does not look like the stills' scene: did the camera, the rig or the room light "
                      "change? Retake the stills")
        if timing.get("late_bits"):
            print(f"  warning: {timing['late_bits']} bits were shown late (over 10 ms), expect a worse deblur")
        x, scene, rpm = deblur(img, calib_here, code, chip, t_exp, args.rpm, method, args.solver, args.lam, args.find_speed,
                               timing)
        np.save(f"{stem}.npy", x)
        speeds.append(rpm)
        fig, ax = plt.subplots(1, 3 if ref is not None else 2, figsize=(21 if ref is not None else 14, 5))
        show(img, ax[0], f"{path.name} (capture)")
        show(np.clip(x, 0, None), ax[1], f"deblurred, {method}, {args.solver}, {rpm:.3f} rpm")
        outline(ax[0], calib_here)
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
