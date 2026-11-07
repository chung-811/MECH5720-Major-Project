import argparse
from pathlib import Path

import numpy as np

import flutter as F

# what a wrong speed or a slipped code bit costs: global shutter, coded light, DPIR. The capture is made at
# the true speed with the code as shown, the deblur assumes the commanded speed and the designed code.
# Same capture as coding_gain.py (full light, seed 1), so the no-fault row is its 25 px coded number.
ap = argparse.ArgumentParser(description="PSNR cost of a speed error and of one code bit shown twice")
ap.add_argument("render", nargs="?", default="renders/blade_tex.npz", help="npz from blender/render_fan.py")
ap.add_argument("--chip", type=float, default=F.CHIP, help="seconds per code bit")
args = ap.parse_args()
CODE, chip = F.RASKAR52, args.chip
NOISE = dict(full_e=10000, read_e=3.0)
Tc = len(CODE) * chip

scene = F.load(args.render)
flux = 0.7 * NOISE["full_e"] / (scene.ref.max() * Tc)
L = F.light(scene.t, CODE, chip)
# one bit shown twice and the next skipped: the first bit from the middle that differs from the next one
i = next(k for k in range(len(CODE) // 2, len(CODE) - 1) if CODE[k] != CODE[k + 1])
slipped = CODE[:i + 1] + CODE[i] + CODE[i + 2:]


def run(shown, omega):
    P, lines = F.unwrap(scene, F.capture(scene, F.light(scene.t, shown, chip), Tc, flux,
                                         np.random.default_rng(1), **NOISE))
    return F.score(scene, F.wrap(scene, F.dpir(*F.forward_model(scene, L, Tc, lines, omega=omega), P)))[0]


rows = []                                       # speed error (%), bit slipped (0/1), PSNR dB
for err in (0, 1, 2, 5):
    rows.append([err, 0, run(CODE, scene.omega * (1 + err / 100))])
rows.append([0, 1, run(slipped, scene.omega)])
for err, slip, p in rows:
    print(f"speed off {err:g}%{', bit ' + str(i + 1) + ' shown twice' if slip else '':22s} {p:6.2f} dB "
          f"({p - rows[0][2]:+.2f})", flush=True)
Path("out").mkdir(exist_ok=True)
np.save("out/faults.npy", np.array(rows))                      # for the report's limitations (make_figures.py)
