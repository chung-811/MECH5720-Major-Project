import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

import flutter as F

READOUT = 0.0632            # OV5647 2592x1944 readout, s, the likely mode at the GS resolution (unverified)
NOISE = dict(full_e=10000, read_e=3.0)   # sensor placeholders until the real cameras are characterised
LAMS = np.geomspace(1e-6, 1e-1, 11)

s = F.load(sys.argv[1] if len(sys.argv) > 1 else "renders/blade.npz")
t, dt, om = s.t, s.dt, s.om
t_line = READOUT / s.lines
code = F.RASKAR52
Tc = len(code) * F.CHIP
assert t[-1] + dt / 2 >= Tc + READOUT, "render longer: t_end must cover code + readout"

rng = np.random.default_rng(0)
coded, box = F.light(t, code), F.light(t, "1" * len(code))
flux = 0.7 * NOISE["full_e"] / (s.ref.max() * Tc)     # constant light on white fills 70% of the well
Pref = F.unwrap(s, s.ref)[0]


def best(P, lines, L, t_exp, **kw):
    """deblur at the lam that scores best on every 4th ring, then the full image at that lam"""
    sub = slice(None, None, 4)
    sc = [psnr(Pref[sub], F.deblur(P[sub], lines[sub], t, dt, L, t_exp, om, s.dth, lam, **kw), data_range=1)
          for lam in LAMS]
    lam = LAMS[int(np.argmax(sc))]
    return F.wrap(s, F.deblur(P, lines, t, dt, L, t_exp, om, s.dth, lam, **kw)), lam


caps = {
    "GS coded": F.capture(s, coded, Tc, flux, rng, **NOISE),
    "RS coded": F.capture(s, coded, Tc, flux, rng, t_line, **NOISE),
    "GS constant": F.capture(s, box, Tc, flux, rng, **NOISE),
    "GS short": F.capture(s, box, F.CHIP, flux, rng, gain=16, **NOISE),
}
res, lams = {}, {}
for name, L, img, kw in [
    ("GS coded, per-frame", coded, caps["GS coded"], {}),
    ("RS coded, per-frame", coded, caps["RS coded"], dict(t_line=t_line, per_row=False, ref_line=s.lines // 2)),
    ("RS coded, per-row", coded, caps["RS coded"], dict(t_line=t_line)),
    ("GS constant, deblurred", box, caps["GS constant"], {}),
]:
    res[name], lams[name] = best(*F.unwrap(s, img), L, Tc, **kw)
    print(f"{name:24s} lam {lams[name]:.0e}  PSNR {F.score(s, res[name])[0]:5.2f} dB  SSIM {F.score(s, res[name])[1]:.3f}")
short = "GS short (1 chip, x16)"
res[short] = F.wrap(s, F.unwrap(s, caps["GS short"] / F.CHIP)[0])
print(f"{short:24s}            PSNR {F.score(s, res[short])[0]:5.2f} dB")

# criterion 1: PSNR per band of sensor lines, first to last read
print("\nPSNR by band of sensor lines (first read -> last read)")
for name in list(res)[:3]:
    print(f"{name:24s} " + " ".join(f"{v:5.1f}" for v in F.bands(s, res[name])))

# criterion 3: blind speed from the GS coded capture, every 6th ring
P, lines = F.unwrap(s, caps["GS coded"])
omegas = om * np.linspace(0.8, 1.2, 21)
est, sc = F.speed(P[::6], lines[::6], t, dt, coded, Tc, omegas, s.dth, lams["GS coded, per-frame"])
print(f"\nblind speed: true {om * 60 / (2 * np.pi):.3f} rpm, estimate {est * 60 / (2 * np.pi):.3f} rpm "
      f"({100 * (est / om - 1):+.1f}%)")

out = Path("out")
out.mkdir(exist_ok=True)
panels = {"truth": s.truth, **{k + " (raw)": v / (Tc if "short" not in k else F.CHIP) for k, v in caps.items()}, **res}
fig, ax = plt.subplots(2, 5, figsize=(20, 6))
for a, (k, v) in zip(ax.ravel(), panels.items()):
    a.imshow(v, cmap="gray", vmin=0, vmax=s.ref.max()), a.set_title(k, fontsize=9), a.axis("off")
plt.tight_layout()
plt.savefig(out / "demo.png", dpi=90)
fig, a = plt.subplots(figsize=(5, 3))
a.plot(omegas / om, sc, "o-"), a.axvline(1, color="k", lw=0.5)
a.set_xlabel("candidate / true speed"), a.set_ylabel("negative light fraction")
plt.tight_layout()
plt.savefig(out / "speed.png", dpi=90)
print(f"figures in {out}/")
