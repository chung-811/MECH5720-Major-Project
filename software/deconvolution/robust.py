import argparse

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.metrics import peak_signal_noise_ratio as psnr

import flutter as F

# what the real rig does that the ideal simulation does not, and what each costs.
# global shutter, coded, lam fixed at demo.py's pick, same noise seed for every capture.
# "ideal kernel" deblurs with the designed code, "measured kernel" with the true waveform
NOISE = dict(full_e=10000, read_e=3.0)
LAM = 1e-2

ap = argparse.ArgumentParser(description="PSNR cost of black level, LCD lag, code timing, hub error and "
                             "stepper motion (global shutter, Tikhonov)")
ap.add_argument("render", nargs="?", default="renders/blade480.npz", help="smooth-motion render")
ap.add_argument("stepped", nargs="?", help="the same render with --steps_per_rev, for the stepper row")
ap.add_argument("--chip", type=float, default=F.CHIP, help="seconds per code bit")
args = ap.parse_args()

scene = F.load(args.render)
code, chip = F.RASKAR52, args.chip
Tc = len(code) * chip
flux = 0.7 * NOISE["full_e"] / (scene.ref.max() * Tc)
ideal = F.light(scene.t, code, chip)


def run(L_true, L_model, err=(0.0, 0.0), sim=scene):
    """capture under L_true, unwrap about a hub off by err (rows, cols) px, deblur assuming L_model.
    Scored against the truth unwrapped about the same hub, so only the deblur's loss counts"""
    img = F.capture(sim, L_true, Tc, flux, np.random.default_rng(1), **NOISE)
    off = F.grid((sim.hub[0] + err[0], sim.hub[1] + err[1]), sim.r, sim.th[0], sim.th[-1] + sim.dtheta / 2,
                 sim.dtheta, sim.shape, sim.axis, sim.t, sim.dt, sim.omega)
    P, lines = F.unwrap(off, img)
    x = F.wrap(off, F.tikhonov(off, P, lines, L_model, Tc, LAM))
    truth = F.wrap(off, F.unwrap(off, sim.ref)[0])
    window = F.wrap(off, np.ones_like(P)) > 0.5
    m = window & binary_dilation(truth > 0.05, iterations=5)
    return psnr(truth[m], x[m], data_range=1)


base = run(ideal, ideal)
rows = [("ideal rig", base, None)]
for b in (0.01, 0.05):
    L = F.light(scene.t, code, chip, black=b)
    rows.append((f"black level {b:.0%}", run(L, ideal), run(L, L)))
for tau in (0.002, 0.005):
    L = F.lcd(ideal, scene.dt, tau)
    rows.append((f"LCD lag {tau * 1e3:.0f} ms", run(L, ideal), run(L, L)))
L = F.lcd(F.light(scene.t, code, chip, black=0.02), scene.dt, 0.003)
rows.append(("black 2% + LCD 3 ms", run(L, ideal), run(L, L)))
for ph in (0.1, 0.25, 0.5):
    L = F.light(scene.t, code, chip, t0=ph * chip)
    rows.append((f"code late {ph} chip", run(L, ideal), run(L, L)))
for e in (1, 2, 5):
    rows.append((f"hub off {e} px radially", run(ideal, ideal, (e, 0)), None))
    rows.append((f"hub off {e} px sideways", run(ideal, ideal, (0, e)), None))
if args.stepped:
    rows.append(("28BYJ-48 full steps", run(ideal, ideal, sim=F.load(args.stepped)), None))

print(f"{'effect':24s} {'ideal kernel':>14s} {'measured kernel':>16s}   (PSNR dB, change vs ideal rig)")
for name, a, b in rows:
    fmt = lambda v: "" if v is None else f"{v:5.2f} ({v - base:+5.2f})"
    print(f"{name:24s} {fmt(a):>14s} {fmt(b):>16s}")
