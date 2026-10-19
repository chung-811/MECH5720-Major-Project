import sys

import numpy as np
import scipy.sparse as sp

import flutter as F
import ocr

# criterion 2: fraction of blade characters OCR reads correctly, against fan speed.
# args: renders made with --text, e.g. renders/ocr_*.npz. mean over SEEDS noise draws
CODE = F.RASKAR52
NOISE = dict(full_e=10000, read_e=3.0)
SEEDS = (1, 2, 3)
Tc = len(CODE) * F.CHIP

print(f"{'rpm':>5s} {'blur px':>8s} {'light':>6s} {'coded DPIR':>11s} {'coded Tikh':>11s} {'const DPIR':>11s} "
      f"{'short+den':>10s}   (fraction read correctly)")
for path in sys.argv[1:]:
    s = F.load(path)
    d = np.load(path)
    coded, box = F.light(s.t, CODE), F.light(s.t, "1" * len(CODE))
    for light in (1.0, 0.1):
        flux = light * 0.7 * NOISE["full_e"] / (s.ref.max() * Tc)
        acc = []
        for seed in SEEDS:
            P, lines = F.unwrap(s, F.capture(s, coded, Tc, flux, np.random.default_rng(seed), **NOISE))
            row = [F.wrap(s, F.dpir(*F.operator(s, coded, Tc, lines), P)),
                   F.wrap(s, F.deblur(P, lines, s.t, s.dt, coded, Tc, s.om, s.dth, 1e-2))]
            P, lines = F.unwrap(s, F.capture(s, box, Tc, flux, np.random.default_rng(seed), **NOISE))
            row.append(F.wrap(s, F.dpir(*F.operator(s, box, Tc, lines), P)))
            short = F.capture(s, F.light(s.t, "1"), F.CHIP, flux, np.random.default_rng(seed), gain=16, **NOISE) / F.CHIP
            P = F.unwrap(s, short)[0]
            row.append(F.wrap(s, F.dpir(sp.identity(P.size, format="csr"), 0, P)))
            acc.append([ocr.legibility(x, d)[0] for x in row])
        m = np.mean(acc, 0)
        print(f"{s.om * 60 / (2 * np.pi):5.2f} {s.om * Tc * s.r[-1]:8.0f} {light:6.1f} "
              + " ".join(f"{v:11.2f}" for v in m), flush=True)
