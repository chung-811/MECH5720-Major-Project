"""Three-way comparison on Blender renders. Run after blender/render_fan.py:

    uv run python demo.py renders/fan.npz

lam is picked per method against the ground truth (an oracle), so this measures each method at its best.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

import flutter as F

READOUT = 0.0298            # OV5647 1080p active readout, s (kernel driver: 1080 lines x 27.6 us)
FULL_E, READ_E = 10000, 3.0  # sensor placeholders until the real cameras are characterised
BANDS = 8
LAMS = np.geomspace(1e-6, 1e-1, 11)

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "renders/fan.npz")
fr, t, dt, om, ref = d["frames"], d["t"], float(d["dt"]), float(d["omega"]), d["ref"]
H = ref.shape[0]
c = ((H - 1) / 2,) * 2
r = np.arange(30, float(d["rim_px"]) - 2)
n = int(2 * np.pi * r[-1])
t_line = READOUT / H
code = F.RASKAR52
Tc = len(code) * F.CHIP
assert t[-1] + dt / 2 >= Tc + READOUT, "render longer: t_end must cover code + readout"

rng = np.random.default_rng(0)
coded, box = F.light(t, code), F.light(t, "1" * len(code))
flux = 0.7 * FULL_E / (ref.max() * Tc)          # constant light on white fills 70% of the well


def shoot(L, t_exp, t_line=0.0, gain=1.0):
    starts = np.arange(H) * t_line if t_line else [0.0]
    return F.sensor(F.expose(fr, F.weights(t, dt, L, t_exp, starts)), flux, rng, FULL_E, READ_E, gain=gain)


mask = (lambda rr: (rr >= r[0]) & (rr <= r[-1]))(np.hypot(*(np.mgrid[:H, :H] - c[0])))
truth = F.cartesian(F.polar(ref, c, r, n)[0], c, r, ref.shape)   # same interpolation as the results


def score(x):
    return psnr(truth[mask], x[mask], data_range=1), ssim(truth, x * mask, data_range=1)


def best(P, rows, L, t_exp, **kw):
    """deblur at the lam that scores best on every 4th ring, then the full image at that lam"""
    sub = slice(None, None, 4)
    s = [psnr(F.polar(ref, c, r, n)[0][sub], F.deblur(P[sub], rows[sub], t, dt, L, t_exp, om, lam, **kw), data_range=1)
         for lam in LAMS]
    lam = LAMS[int(np.argmax(s))]
    return F.cartesian(F.deblur(P, rows, t, dt, L, t_exp, om, lam, **kw), c, r, ref.shape), lam


caps = {
    "GS coded": shoot(coded, Tc),
    "RS coded": shoot(coded, Tc, t_line),
    "GS constant": shoot(box, Tc),
    "GS short": shoot(box, F.CHIP, gain=16),
}
res = {}
for name, L, img, kw in [
    ("GS coded, per-frame", coded, caps["GS coded"], {}),
    ("RS coded, per-frame", coded, caps["RS coded"], dict(t_line=t_line, per_row=False, ref_row=H // 2)),
    ("RS coded, per-row", coded, caps["RS coded"], dict(t_line=t_line)),
    ("GS constant, deblurred", box, caps["GS constant"], {}),
]:
    P, rows = F.polar(img, c, r, n)
    res[name], lam = best(P, rows, L, Tc, **kw)
    print(f"{name:24s} lam {lam:.0e}  PSNR {score(res[name])[0]:5.2f} dB  SSIM {score(res[name])[1]:.3f}")
res["GS short (1 chip, x16)"] = F.cartesian(F.polar(caps["GS short"] / F.CHIP, c, r, n)[0], c, r, ref.shape)
print(f"{'GS short (1 chip, x16)':24s}            PSNR {score(res['GS short (1 chip, x16)'])[0]:5.2f} dB")

# criterion 1: PSNR per band of rows, top to bottom
edges = np.linspace(0, H, BANDS + 1).astype(int)
print("\nPSNR by row band (top -> bottom)")
for name in list(res)[:3]:
    vals = [psnr(truth[a:b][mask[a:b]], res[name][a:b][mask[a:b]], data_range=1) if mask[a:b].any() else np.nan
            for a, b in zip(edges[:-1], edges[1:])]
    print(f"{name:24s} " + " ".join(f"{v:5.1f}" for v in vals))

# criterion 3: blind speed from the GS coded capture, every 6th ring
P, rows = F.polar(caps["GS coded"], c, r, n)
omegas = om * np.linspace(0.8, 1.2, 21)
est, s = F.speed(P[::6], rows[::6], t, dt, coded, Tc, omegas, 1e-4)
print(f"\nblind speed: true {om * 60 / (2 * np.pi):.3f} rpm, estimate {est * 60 / (2 * np.pi):.3f} rpm "
      f"({100 * (est / om - 1):+.1f}%)")

out = Path("out")
out.mkdir(exist_ok=True)
panels = {"truth": truth, **{k + " (raw)": v / (Tc if "short" not in k else F.CHIP) for k, v in caps.items()}, **res}
fig, ax = plt.subplots(2, 5, figsize=(20, 8))
for a, (k, v) in zip(ax.ravel(), panels.items()):
    a.imshow(v, cmap="gray", vmin=0, vmax=ref.max()), a.set_title(k, fontsize=9), a.axis("off")
plt.tight_layout()
plt.savefig(out / "demo.png", dpi=90)
fig, a = plt.subplots(figsize=(5, 3))
a.plot(omegas / om, s, "o-"), a.axvline(1, color="k", lw=0.5)
a.set_xlabel("candidate / true speed"), a.set_ylabel("negative light fraction")
plt.tight_layout()
plt.savefig(out / "speed.png", dpi=90)
print(f"figures in {out}/")
