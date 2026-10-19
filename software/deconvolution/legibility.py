import argparse

import numpy as np
import scipy.sparse as sp

import flutter as F
import ocr

# criterion 2: fraction of blade characters OCR reads correctly, against fan speed.
ap = argparse.ArgumentParser(description="criterion 2: fraction of blade characters OCR reads, by fan speed "
                             "and light, mean over 3 noise draws (needs tesseract)")
ap.add_argument("renders", nargs="+", help="renders made with --text, e.g. renders/ocr_*.npz")
ap.add_argument("--chip", type=float, default=F.CHIP, help="seconds per code bit")
args = ap.parse_args()
CODE, chip = F.RASKAR52, args.chip
NOISE = dict(full_e=10000, read_e=3.0)
SEEDS = (1, 2, 3)
Tc = len(CODE) * chip

print(f"{'rpm':>5s} {'blur px':>8s} {'light':>6s} {'coded DPIR':>11s} {'coded Tikh':>11s} {'const DPIR':>11s} "
      f"{'short+den':>10s}   (fraction read correctly)")
for path in args.renders:
    scene = F.load(path)
    d = np.load(path)
    coded, box = F.light(scene.t, CODE, chip), F.light(scene.t, "1" * len(CODE), chip)
    for light in (1.0, 0.1):
        flux = light * 0.7 * NOISE["full_e"] / (scene.ref.max() * Tc)
        acc = []
        for seed in SEEDS:
            P, lines = F.unwrap(scene, F.capture(scene, coded, Tc, flux, np.random.default_rng(seed), **NOISE))
            row = [F.wrap(scene, F.dpir(*F.forward_model(scene, coded, Tc, lines), P)),
                   F.wrap(scene, F.tikhonov(scene, P, lines, coded, Tc, 1e-2))]
            P, lines = F.unwrap(scene, F.capture(scene, box, Tc, flux, np.random.default_rng(seed), **NOISE))
            row.append(F.wrap(scene, F.dpir(*F.forward_model(scene, box, Tc, lines), P)))
            short = F.capture(scene, F.light(scene.t, "1", chip), chip, flux, np.random.default_rng(seed), gain=16,
                              **NOISE) / chip
            P = F.unwrap(scene, short)[0]
            row.append(F.wrap(scene, F.dpir(sp.identity(P.size, format="csr"), 0, P)))
            acc.append([ocr.legibility(x, d)[0] for x in row])
        m = np.mean(acc, 0)
        print(f"{scene.omega * 60 / (2 * np.pi):5.2f} {scene.omega * Tc * scene.r[-1]:8.0f} {light:6.1f} "
              + " ".join(f"{v:11.2f}" for v in m), flush=True)
