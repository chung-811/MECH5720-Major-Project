import sys

import numpy as np

import flutter as F

NOISE = dict(full_e=10000, read_e=3.0)
LAM = 3e-3                                  # demo.py's pick for GS coded

s = F.load(sys.argv[1] if len(sys.argv) > 1 else "renders/fan480.npz")
code = F.RASKAR52
Tc = len(code) * F.CHIP
flux = 0.7 * NOISE["full_e"] / (s.ref.max() * Tc)
ideal = F.light(s.t, code)


def run(L_true, L_model, err_px=0.0, sim=s):
    """capture under L_true, unwrap about a centre err_px off in x, deblur assuming L_model.
    Scored against the truth unwrapped about the same centre, so only the deblur's loss counts"""
    img = F.capture(sim, L_true, Tc, flux, np.random.default_rng(1), **NOISE)
    c = (sim.c[0], sim.c[1] + err_px)
    P, rows = F.polar(img, c, sim.r, sim.n)
    x = F.deblur(P, rows, sim.t, sim.dt, L_model, Tc, sim.om, LAM)
    truth = F.cartesian(F.polar(sim.ref, c, sim.r, sim.n)[0], c, sim.r, sim.ref.shape)
    m = np.hypot(*(np.mgrid[:sim.H, :sim.H] - np.array(c)[:, None, None])) <= sim.r[-1]
    m &= np.hypot(*(np.mgrid[:sim.H, :sim.H] - np.array(c)[:, None, None])) >= sim.r[0]
    return F.psnr(truth[m], F.cartesian(x, c, sim.r, sim.ref.shape)[m], data_range=1)


base = run(ideal, ideal)
rows = [("ideal rig", base, None)]
for b in (0.01, 0.05):
    L = F.light(s.t, code, black=b)
    rows.append((f"black level {b:.0%}", run(L, ideal), run(L, L)))
for tau in (0.002, 0.005):
    L = F.lcd(ideal, s.dt, tau)
    rows.append((f"LCD lag {tau * 1e3:.0f} ms", run(L, ideal), run(L, L)))
L = F.lcd(F.light(s.t, code, black=0.02), s.dt, 0.003)
rows.append(("black 2% + LCD 3 ms", run(L, ideal), run(L, L)))
for ph in (0.1, 0.25, 0.5):
    L = F.light(s.t, code, t0=ph * F.CHIP)
    rows.append((f"code late {ph} chip", run(L, ideal), run(L, L)))
for e in (0.25, 0.5, 1, 2):
    rows.append((f"hub centre off {e} px", run(ideal, ideal, e), None))
if len(sys.argv) > 2:
    st = F.load(sys.argv[2])
    rows.append(("28BYJ-48 full steps", run(ideal, ideal, sim=st), None))

print(f"{'effect':24s} {'ideal kernel':>14s} {'measured kernel':>16s}   (PSNR dB, change vs ideal rig)")
for name, a, b in rows:
    fmt = lambda v: "" if v is None else f"{v:5.2f} ({v - base:+5.2f})"
    print(f"{name:24s} {fmt(a):>14s} {fmt(b):>16s}")
