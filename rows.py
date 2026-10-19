import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

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
LAMS = np.geomspace(1e-5, 1e-1, 9)

s = F.load(sys.argv[1] if len(sys.argv) > 1 else "renders/blade.npz",
           axis=int(sys.argv[2]) if len(sys.argv) > 2 else 1)   # 1 = readout across columns, 0 = down rows
t, dt, om = s.t, s.dt, s.om
flux = 0.7 * NOISE["full_e"] / (s.ref.max() * len(F.RASKAR52) * F.CHIP)   # one monitor brightness for all
Pref = F.unwrap(s, s.ref)[0]
assert t[-1] + dt / 2 >= len(F.RASKAR52) * F.CHIP + max(READOUTS.values()), "render longer"


def deblur(img, L, Tc, lam, **kw):
    return F.wrap(s, F.deblur(*F.unwrap(s, img), t, dt, L, Tc, om, s.dth, lam, **kw))


fig, ax = plt.subplots(1, len(CODES), figsize=(5 * len(CODES), 3.6), sharey=True)
print(f"{'code':10s} {'readout':16s} {'method':10s} {'mean':>6s} {'worst':>6s}  band PSNR first -> last line read (dB)")
for a, (cname, code) in zip(ax, CODES.items()):
    Tc = len(code) * F.CHIP
    L = F.light(t, code)
    gs = F.capture(s, L, Tc, flux, np.random.default_rng(1), **NOISE)
    P, lines = F.unwrap(s, gs)
    lam = LAMS[int(np.argmax([psnr(Pref[::4], F.deblur(P[::4], lines[::4], t, dt, L, Tc, om, s.dth, l), data_range=1)
                              for l in LAMS]))]
    b = F.bands(s, deblur(gs, L, Tc, lam))
    print(f"{cname:10s} {'global':16s} {'':10s} {np.nanmean(b):6.2f} {np.nanmin(b):6.2f}  " + " ".join(f"{v:5.1f}" for v in b))
    a.plot(b, "k-", lw=2, label="global shutter")
    for (mode, ro), col in zip(READOUTS.items(), ("C0", "C1", "C3")):
        tl = ro / s.lines
        rs = F.capture(s, L, Tc, flux, np.random.default_rng(1), tl, **NOISE)
        for per_row, ls in ((False, "--"), (True, "-")):
            b = F.bands(s, deblur(rs, L, Tc, lam, t_line=tl, per_row=per_row, ref_line=s.lines // 2))
            meth = "per-row" if per_row else "per-frame"
            print(f"{'':10s} {mode} {ro / F.CHIP:4.1f}ch {meth:10s} {np.nanmean(b):6.2f} {np.nanmin(b):6.2f}  "
                  + " ".join(f"{v:5.1f}" for v in b))
            a.plot(b, ls, color=col, label=f"{mode} {meth}")
    a.set_title(f"{cname} ({Tc * 1e3:.0f} ms exposure, lam {lam:.0e})", fontsize=9)
    a.set_xlabel("band of sensor lines (first -> last read)")
ax[0].set_ylabel("PSNR over the blade (dB)")
ax[-1].legend(fontsize=7)
plt.tight_layout()
Path("out").mkdir(exist_ok=True)
plt.savefig(f"out/rows_axis{s.axis}.png", dpi=90)
print(f"figure in out/rows_axis{s.axis}.png")
