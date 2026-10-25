import argparse

import numpy as np
import scipy.sparse as sp

import flutter as F

# does coding the light still pay with a strong prior? coded vs constant light over the same exposure,
# and a one-chip exposure at 16x gain, each reconstructed by DPIR (the short one is only denoised).
ap = argparse.ArgumentParser(description="coded vs constant light vs a denoised short exposure, by fan speed and light")
ap.add_argument("renders", nargs="+", help="renders at different fan speeds, e.g. renders/blade_tex*.npz")
ap.add_argument("--chip", type=float, default=F.CHIP, help="seconds per code bit")
args = ap.parse_args()
CODE, chip = F.RASKAR52, args.chip
NOISE = dict(full_e=10000, read_e=3.0)
Tc = len(CODE) * chip

print(f"{'rpm':>5s} {'blur px':>8s} {'light':>6s} {'coded':>8s} {'constant':>9s} {'short+denoise':>14s}  (PSNR dB)")
for path in args.renders:
    scene = F.load(path)
    for light in (1.0, 0.1):
        flux = light * 0.7 * NOISE["full_e"] / (scene.ref.max() * Tc)
        out = []
        for L in (F.light(scene.t, CODE, chip), F.light(scene.t, "1" * len(CODE), chip)):
            P, lines = F.unwrap(scene, F.capture(scene, L, Tc, flux, np.random.default_rng(1), **NOISE))
            out.append(F.score(scene, F.wrap(scene, F.dpir(*F.forward_model(scene, L, Tc, lines), P)))[0])
        short = F.capture(scene, F.light(scene.t, "1", chip), chip, flux, np.random.default_rng(1), gain=16,
                          **NOISE) / chip
        P = F.unwrap(scene, short)[0]
        out.append(F.score(scene, F.wrap(scene, F.dpir(sp.identity(P.size, format="csr"), 0, P)))[0])
        print(f"{scene.omega * 60 / (2 * np.pi):5.2f} {scene.omega * Tc * scene.r[-1]:8.0f} {light:6.1f} "
              + " ".join(f"{v:8.2f}" for v in out), flush=True)
