import sys

import numpy as np
import scipy.sparse as sp

import flutter as F

# does coding the light still pay with a strong prior? coded vs constant light over the same exposure,
# and a one-chip exposure at 16x gain, each reconstructed by DPIR (the short one is only denoised).
# args: renders at different fan speeds, e.g. renders/blade_tex*.npz
CODE = F.RASKAR52
NOISE = dict(full_e=10000, read_e=3.0)
Tc = len(CODE) * F.CHIP

print(f"{'rpm':>5s} {'blur px':>8s} {'light':>6s} {'coded':>8s} {'constant':>9s} {'short+denoise':>14s}  (PSNR dB)")
for path in sys.argv[1:]:
    s = F.load(path)
    for light in (1.0, 0.1):
        flux = light * 0.7 * NOISE["full_e"] / (s.ref.max() * Tc)
        out = []
        for L in (F.light(s.t, CODE), F.light(s.t, "1" * len(CODE))):
            P, lines = F.unwrap(s, F.capture(s, L, Tc, flux, np.random.default_rng(1), **NOISE))
            out.append(F.score(s, F.wrap(s, F.dpir(*F.operator(s, L, Tc, lines), P)))[0])
        short = F.capture(s, F.light(s.t, "1"), F.CHIP, flux, np.random.default_rng(1), gain=16, **NOISE) / F.CHIP
        P = F.unwrap(s, short)[0]
        out.append(F.score(s, F.wrap(s, F.dpir(sp.identity(P.size, format="csr"), 0, P)))[0])
        print(f"{s.om * 60 / (2 * np.pi):5.2f} {s.om * Tc * s.r[-1]:8.0f} {light:6.1f} "
              + " ".join(f"{v:8.2f}" for v in out), flush=True)
