import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

import flutter as F

ap = argparse.ArgumentParser(description="all three methods plus constant light and a short exposure on one "
                             "render: PSNR, PSNR by band of sensor lines, blind speed, figures in out/")
ap.add_argument("render", nargs="?", default="renders/blade.npz", help="npz from blender/render_fan.py")
ap.add_argument("--code", default=F.RASKAR52, help="monitor code, 1 = white")
ap.add_argument("--chip", type=float, default=F.CHIP, help="seconds per code bit")
ap.add_argument("--readout", type=float, default=0.0632,
                help="rolling shutter first to last line, s (OV5647 2592x1944 from the driver, not yet measured)")
args = ap.parse_args()

NOISE = dict(full_e=10000, read_e=3.0)   # sensor placeholders until the real cameras are characterised
LAMS = np.geomspace(1e-6, 1e-1, 11)

scene = F.load(args.render)
t = scene.t
t_line = args.readout / scene.n_lines
code, chip = args.code, args.chip
Tc = len(code) * chip
assert t[-1] + scene.dt / 2 >= Tc + args.readout, "render longer: t_end must cover code + readout"

rng = np.random.default_rng(0)
coded, box = F.light(t, code, chip), F.light(t, "1" * len(code), chip)
flux = 0.7 * NOISE["full_e"] / (scene.ref.max() * Tc)     # constant light on white fills 70% of the well
Pref = F.unwrap(scene, scene.ref)[0]


def best(P, lines, L, t_exp, **kw):
    """deblur at the lam that scores best on every 4th ring, then the full image at that lam"""
    sub = slice(None, None, 4)
    sc = [psnr(Pref[sub], F.tikhonov(scene, P[sub], lines[sub], L, t_exp, lam, **kw), data_range=1) for lam in LAMS]
    lam = LAMS[int(np.argmax(sc))]
    return F.wrap(scene, F.tikhonov(scene, P, lines, L, t_exp, lam, **kw)), lam


caps = {
    "GS coded": F.capture(scene, coded, Tc, flux, rng, **NOISE),
    "RS coded": F.capture(scene, coded, Tc, flux, rng, t_line, **NOISE),
    "GS constant": F.capture(scene, box, Tc, flux, rng, **NOISE),
    "GS short": F.capture(scene, box, chip, flux, rng, gain=16, **NOISE),
}
res, lams = {}, {}
print(f"{'method':24s} {'tikhonov (best lam)':>22s} {'DPIR (no tuning)':>18s}")
for name, L, img, kw in [
    ("GS coded, per-frame", coded, caps["GS coded"], {}),
    ("RS coded, per-frame", coded, caps["RS coded"], dict(t_line=t_line, per_row=False, ref_line=scene.n_lines // 2)),
    ("RS coded, per-row", coded, caps["RS coded"], dict(t_line=t_line)),
    ("GS constant, deblurred", box, caps["GS constant"], {}),
]:
    P, lines = F.unwrap(scene, img)
    tik, lams[name] = best(P, lines, L, Tc, **kw)
    res[name] = F.wrap(scene, F.dpir(*F.forward_model(scene, L, Tc, lines, **kw), P))
    p_dpir, ssim_dpir = F.score(scene, res[name])
    print(f"{name:24s} {F.score(scene, tik)[0]:8.2f} dB  lam {lams[name]:.0e} {p_dpir:10.2f} dB  SSIM {ssim_dpir:.3f}")
short = "GS short (1 chip, x16)"
res[short] = F.wrap(scene, F.unwrap(scene, caps["GS short"] / chip)[0])
print(f"{short:24s} {F.score(scene, res[short])[0]:8.2f} dB (no deblur)")

# criterion 1: PSNR per band of sensor lines, first to last read
print("\nDPIR PSNR by band of sensor lines (first read -> last read)")
for name in list(res)[:3]:
    print(f"{name:24s} " + " ".join(f"{v:5.1f}" for v in F.bands(scene, res[name])))

# criterion 3: blind speed from the GS coded capture, DPIR residual on a band of rings
om = scene.omega
P, lines = F.unwrap(scene, caps["GS coded"])
omegas = om * np.linspace(0.8, 1.2, 21)
est, sc = F.speed(scene, P, lines, coded, Tc, omegas)
print(f"\nblind speed: true {om * 60 / (2 * np.pi):.3f} rpm, estimate {est * 60 / (2 * np.pi):.3f} rpm "
      f"({100 * (est / om - 1):+.1f}%)")

out = Path("out")
out.mkdir(exist_ok=True)
panels = {"truth": scene.truth, **{k + " (raw)": v / (Tc if "short" not in k else chip) for k, v in caps.items()}, **res}
fig, ax = plt.subplots(2, 5, figsize=(20, 6))
for a, (k, v) in zip(ax.ravel(), panels.items()):
    a.imshow(v, cmap="gray", vmin=0, vmax=scene.ref.max()), a.set_title(k, fontsize=9), a.axis("off")
plt.tight_layout()
plt.savefig(out / "demo.png", dpi=90)
fig, a = plt.subplots(figsize=(5, 3))
a.plot(omegas / om, sc, "o-"), a.axvline(1, color="k", lw=0.5)
a.set_xlabel("candidate / true speed"), a.set_ylabel("DPIR data residual")
plt.tight_layout()
plt.savefig(out / "speed.png", dpi=90)
print(f"figures in {out}/")
