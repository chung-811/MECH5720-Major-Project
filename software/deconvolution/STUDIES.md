# Studies and simulation results

The background behind the pipeline: the Blender simulation, the solver benchmark, results against the success criteria, and what each fault costs. How to use the code is in README.md.

## Simulation (Blender)

Sharp sub-frames from Blender, coded light and rolling shutter applied in numpy, then deblurred.
Blender's own shutter curve and rolling shutter are not used: the curve leaks about 10% of the
light into off chips, and its rolling shutter squeezes the whole code into every row instead of
giving each row its own slice of one flashing monitor.

Scene: one 50 mm blade filling the frame height, hub above the frame, black backdrop, 0.35 rpm.
(The rig now frames the whole fan; to be re-rendered once the fan is measured.)

```
blender -b -P blender/render_fan.py -- --out renders/blade.npz
uv run python demo.py renders/blade.npz

# harder scene: a photo printed on the blade, and faster fans for the coding gain study
uv run python -c "from skimage import data, transform; from PIL import Image; import numpy as np; im = 0.1 + 0.7 * transform.resize(data.camera() / 255, (1024, 512), anti_aliasing=True); Image.fromarray(np.round(im * 65535).astype(np.uint16)).save('renders/texture.png')"
blender -b -P blender/render_fan.py -- --texture renders/texture.png --out renders/blade_tex.npz
blender -b -P blender/render_fan.py -- --texture renders/texture.png --rpm 0.7 --dt 0.0041667 --out renders/blade_tex_0.7.npz
blender -b -P blender/render_fan.py -- --texture renders/texture.png --rpm 1.4 --dt 0.0020833 --out renders/blade_tex_1.4.npz
uv run python coding_gain.py renders/blade_tex.npz renders/blade_tex_0.7.npz renders/blade_tex_1.4.npz

# criterion 1: per-row vs per-frame over code length and OV5647 readout. last arg: 1 = readout
# across columns (camera on its side), 0 = down rows
uv run python rows.py renders/blade.npz 1

# criterion 2: OCR legibility against fan speed (needs tesseract)
for p in 0.35:0.0083333 0.7:0.0041667 1.4:0.0020833 2.8:0.0010417; do blender -b -P blender/render_fan.py -- --text K7R,MX4,PAT,H9W,3EV,NJ6 --text_h 0.005 --rpm ${p%%:*} --dt ${p##*:} --out renders/ocr_${p%%:*}.npz; done
uv run python legibility.py renders/ocr_0.35.npz renders/ocr_0.7.npz renders/ocr_1.4.npz renders/ocr_2.8.npz

# robustness: finer sub-frames, smooth and 28BYJ-48 stepped
blender -b -P blender/render_fan.py -- --dt 0.0020833 --out renders/blade480.npz
blender -b -P blender/render_fan.py -- --dt 0.0020833 --steps_per_rev 2038 --out renders/blade480_steps.npz
uv run python robust.py renders/blade480.npz renders/blade480_steps.npz
```

Every study takes `--chip` (seconds per code bit, default 1/60).

## Deconvolution

`flutter.forward_model` builds the forward model as one sparse matrix; `flutter.dpir` inverts it with
DPIR (Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", TPAMI 2022): half-quadratic
splitting with a pretrained DRUNet denoiser as the prior, through the deepinv library. It needs no
tuning: the model is scaled to unit gain and the noise level is estimated from the capture.
`estimate_sigma` reads about a third of the true noise on the resampled polar data (0.31 to 0.35 over
both scenes, light 1x to 0.03x and both shutters) and DPIR did best at twice the true noise, so it
uses 6x the estimate (`flutter.NOISE_GAIN`). `flutter.tikhonov` (per-ring least squares with a
gradient penalty) stays as the fast baseline anyone can derive.

Benchmark, global shutter, Raskar 52-chip code, 0.35 rpm (PSNR over the blade):

| Solver | flat blade + letters | photo on the blade |
| --- | --- | --- |
| Tikhonov, best lam | 29.4 | 31.1 |
| TV + [0, 1] box, Chambolle-Pock (pyproximal), best lam | 41.1 | 34.6 |
| DPIR, no tuning | 44.3 | 38.4 |

TV smears thin lines (the tripod legs); DPIR keeps them. With DPIR, coding the light pays when the
blur is long (`coding_gain.py`, PSNR coded / constant / short exposure + denoise):

| blur | full light | a tenth of the light |
| --- | --- | --- |
| 25 px | 38.1 / 37.7 / 36.6 | 32.8 / 33.5 / 28.4 |
| 51 px | 36.4 / 33.2 / 34.0 | 29.2 / 29.0 / 28.1 |
| 101 px | 36.3 / 28.6 / 29.7 | 29.8 / 26.0 / 26.7 |

## Simulation results against the success criteria

Textured blade unless stated, DPIR, one noise draw per case (`rows.py`, `legibility.py`, `demo.py`).

1. Per-row within 1 dB of global shutter: met in every case, both readout directions, all three codes
   and readouts. Per-frame falls behind most when the readout runs down the blade's rows and the code is
   short: 2592x1944 readout (3.8 bits), 15-bit code, per-frame 26.6 dB mean (19.0 worst band), per-row
   41.4, global 42.0. With the Raskar code: 35.8 (31.2), 38.2, 38.3. Readout across the blade:
   per-frame within 0.9 dB of global with the Raskar code and 1.3 dB with the 31-bit code; with the
   15-bit code it drops 2.0 to 8.8 dB (33.6 at the longest readout, per-row 42.2, global 42.4).
2. Fraction of characters read (coded / constant / short + denoise), letters 49 px tall:
   at 101 px blur, a tenth of the light: 0.93 / 0.57 / 0.93; at 202 px: 1.00 / 0.28 / 1.00.
   Letters 25 px (half resolution), 51 px blur, a tenth of the light: 1.00 / 0.78 / 0.96. Letters 16 px:
   0.87 / 0.46 / 0.74, and the sharp still itself reads 0.89. Coded beats constant light as the
   criterion asks, but a denoised short exposure matches it on OCR at full resolution.
3. Blind speed: within 0.26% of the true speed in 9 of 9 runs (3 speeds x 3 noise draws), target 5%.

What each thing going wrong costs (one at a time, against 38.1 dB, 25 px of blur):

| | cost |
| --- | --- |
| speed off 1 / 2 / 5% (use `--find-speed` if unsure) | -0.8 / -2.7 / -8.5 dB |
| rotation direction wrong | -24 dB |
| speed ripple 2 Hz, +-2 / 5 / 10% | -0.1 / -0.7 / -2.1 dB |
| one bit shown twice, the next skipped (`calibrate.py monitor` flags it) | -6.1 dB |
| room light 5% / 20%, not modelled (dark room) | -11 / -23 dB |
| light 0.3 / 0.1 / 0.03 / 0.01x | -2.8 / -5.3 / -8.4 / -11 dB |
| 1.5 / 3x overexposed (coded light is half on, so headroom) | none |
| hub 5 / 10 / 20 px off radially (`calibrate.py hub`: 0.4 px) | -0.8 / -2.5 / -6.1 dB |
| rolling shutter readout assumed 50 / 80 ms (true 63) | -0.7 / -2.2 dB |
| rolling shutter treated as global | -7.2 dB |

Cost per deblur at 2.8 rpm with 40-50 mm blades: 7 s and 0.8 GB at 10 px/mm, 11 s and 1.5 GB at 20,
44 s and 5.2 GB at 40; memory grows about as the cube of px/mm, so 100 px/mm is out of reach.

One solver serves all three methods; they differ only in which exposure window each polar sample
gets. Blade geometry and sensor noise numbers are placeholders until the fan and cameras are measured.

## Rig-scale stress test (3-blade fan, the real commands)

A simulated 3-blade fan at rig scale (3 mm rods, a 15 x 23 mm text paddle per blade, 3 px/mm, global
shutter, `code_16bit_4refresh`, 5 rpm), put through `calibrate.py auto` and `global-shutter.py --reference`,
3 captures per case. PSNR against the reference still:

| case | PSNR (dB) |
| --- | --- |
| baseline | 32.9 to 33.4 |
| 3 / 10 / 15 rpm | 32.6 to 33.0 / 29.6 to 30.2 / 22.4 |
| fan turning in 0.18 deg steps | 32.9 to 33.3 |
| told 2048 steps per rev (really 2038) | 32.7 to 33.5 |
| uneven monitor light (half as bright on one side) | 33.5 to 34.4 |
| 5% room light | 30.9 to 31.2 |
| white stand behind the blades | 8.3 to 11.0 |
| one bit held twice as long | 19.6 |
| blind speed (`--find-speed`), true 5 rpm | 5.000, 5.001, 5.002 rpm |
| `--rpm` with the wrong sign | 15.8 before the direction check, 32.95 with it |
