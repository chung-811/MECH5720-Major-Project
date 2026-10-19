import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import flutter as F

# criterion 1: when does per-row rolling shutter deconvolution beat per-frame?
# exposure = code length, per-frame kernel timed for the middle line, readout spans the frame
CODES = {
    "raskar52": F.RASKAR52,
    "agrawal31": "1011110110101000110111110011111",   # Agrawal & Xu 2009, best for invertibility
    "agrawal15": "111111100110001",                   # Agrawal & Xu 2009, outdoor shot
}
READOUTS = {"1296x972": 0.0211, "1920x1080": 0.0298, "2592x1944": 0.0632}   # OV5647 driver, s
NOISE = dict(full_e=10000, read_e=3.0)

ap = argparse.ArgumentParser(description="criterion 1: PSNR by band of sensor lines for global shutter and "
                             "rolling shutter per-frame and per-row, over code length and readout time")
ap.add_argument("render", nargs="?", default="renders/blade.npz", help="npz from blender/render_fan.py")
ap.add_argument("axis", nargs="?", type=int, default=1, choices=(0, 1),
                help="readout direction: 1 = across columns (camera on its side), 0 = down rows")
ap.add_argument("--chip", type=float, default=F.CHIP, help="seconds per code bit")
args = ap.parse_args()
chip = args.chip

scene = F.load(args.render, axis=args.axis)
t = scene.t
flux = 0.7 * NOISE["full_e"] / (scene.ref.max() * len(F.RASKAR52) * chip)   # one monitor brightness for all
assert t[-1] + scene.dt / 2 >= len(F.RASKAR52) * chip + max(READOUTS.values()), "render longer"


def deblur(img, L, Tc, **kw):
    P, lines = F.unwrap(scene, img)
    return F.wrap(scene, F.dpir(*F.forward_model(scene, L, Tc, lines, **kw), P))


fig, ax = plt.subplots(1, len(CODES), figsize=(5 * len(CODES), 3.6), sharey=True)
print(f"{'code':10s} {'readout':16s} {'method':10s} {'mean':>6s} {'worst':>6s}  band PSNR first -> last line read (dB)")
for a, (cname, code) in zip(ax, CODES.items()):
    Tc = len(code) * chip
    L = F.light(t, code, chip)
    gs = F.capture(scene, L, Tc, flux, np.random.default_rng(1), **NOISE)
    b = F.bands(scene, deblur(gs, L, Tc))
    print(f"{cname:10s} {'global':16s} {'':10s} {np.nanmean(b):6.2f} {np.nanmin(b):6.2f}  " + " ".join(f"{v:5.1f}" for v in b))
    a.plot(b, "k-", lw=2, label="global shutter")
    for (mode, ro), col in zip(READOUTS.items(), ("C0", "C1", "C3")):
        tl = ro / scene.n_lines
        rs = F.capture(scene, L, Tc, flux, np.random.default_rng(1), tl, **NOISE)
        for per_row, ls in ((False, "--"), (True, "-")):
            b = F.bands(scene, deblur(rs, L, Tc, t_line=tl, per_row=per_row, ref_line=scene.n_lines // 2))
            meth = "per-row" if per_row else "per-frame"
            print(f"{'':10s} {mode} {ro / chip:4.1f}ch {meth:10s} {np.nanmean(b):6.2f} {np.nanmin(b):6.2f}  "
                  + " ".join(f"{v:5.1f}" for v in b))
            a.plot(b, ls, color=col, label=f"{mode} {meth}")
    a.set_title(f"{cname} ({Tc * 1e3:.0f} ms exposure, DPIR)", fontsize=9)
    a.set_xlabel("band of sensor lines (first -> last read)")
ax[0].set_ylabel("PSNR over the blade (dB)")
ax[-1].legend(fontsize=7)
plt.tight_layout()
Path("out").mkdir(exist_ok=True)
plt.savefig(f"out/rows_axis{scene.axis}.png", dpi=90)
print(f"figure in out/rows_axis{scene.axis}.png")
