# Deconvolution and velocity estimation

Deblurs coded-illumination captures of the spinning blade, by the three methods: global shutter,
rolling shutter per-frame and rolling shutter per-row. It also estimates the fan speed blind, and has
a Blender simulation to test all of it before the rig exists.

## Setup

Needs [uv](https://docs.astral.sh/uv/), which installs Python and every package itself:

```
curl -LsSf https://astral.sh/uv/install.sh | sh                          # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"                 # Windows
```

Then, in this folder:

```
uv sync          # about 1 GB of packages (PyTorch, CPU only), about 2 minutes the first time
uv run pytest    # 23 checks, about a minute; the first run downloads the 131 MB DRUNet weights
```

If the tests pass, the pipeline works on your machine. Optional extras: Blender 5.2 for new renders,
and [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) for the OCR legibility study.

## Real captures

Every capture is a DNG plus the json run log `software/illumination/coded_illumination.py` saves next to
it. The log gives the code, the bit length and the exposure, so nothing about the code is typed in.
Work in raw sensor orientation (do not rotate the images): the rolling shutter timing follows sensor
lines. On a colour sensor, coordinates are in the half-size green image `real.load_dng` returns.

One calibration json per camera, made once per setup:

```
# 1. first guess, read off a capture: hub (row col), radii and the arc (deg, counter-clockwise from
#    +col) covering the blade and its smear
uv run python calibrate.py init gs.json --hub 480 640 --radius 150 450 --arc 200 300
uv run python calibrate.py init rs.json --hub 480 640 --radius 150 450 --arc 200 300

# 2. check it: draws the hub and arc on a capture, out/<name>_global_check.png
uv run python global-shutter.py cap.dng --calib gs.json --rpm 2.8 --check

# 3. hub: two stills of the stopped blade, turned by a known angle between them (100 steps or more)
uv run python calibrate.py hub gs.json still0.dng still1.dng --deg 17.7

# 4. monitor: 10 coded captures of a plain blade (global shutter camera), into both calibrations.
#    Also checks every bit showed as coded: one refresh per bit can double or skip one
uv run python calibrate.py monitor plain*.dng --calib gs.json rs.json --rpm 2.8 --dark dark*.dng

# 5. rolling shutter only: readout time and when the code starts, from the fan stopped, once under
#    the code and once under constant light (params_constant52.json), same params as the real captures
uv run python calibrate.py readout rs.json still_coded.dng still_constant.dng --dark dark*.dng
```

Then deblur. Each script writes `out/<name>_<method>.png` (capture beside result) and `.npy` (the
deblurred image, same units as a still capture):

```
uv run python global-shutter.py            cap*.dng --calib gs.json --rpm 2.8 --dark dark*.dng
uv run python rolling-shutter-per-frame.py cap*.dng --calib rs.json --rpm 2.8 --dark dark*.dng
uv run python rolling-shutter-per-row.py   cap*.dng --calib rs.json --rpm 2.8 --dark dark*.dng
```

`--rpm` is the commanded speed, + for counter-clockwise as the raw image is displayed. `--find-speed`
also estimates the speed blind (criterion 3) and deblurs at the estimate. `--reference still.dng` (the
stopped blade) turns the still to where the deblurred blade is and reports PSNR, SSIM and PSNR by band
of sensor lines (criterion 1), with mean and std over all the captures given. `--solver tikhonov` is
the fast baseline. `-h` on any script lists its options.

Exposure padding (`exposure_padding_s` in the illumination params): coded_illumination.py starts the
code about 20 ms after the exposure opens, so with no padding the end of the code falls outside it.
Global shutter: pad 0.05 s so the whole code is inside. Rolling shutter: padding decides the
experiment. With none, every line catches a different part of the code, which is what per-row
deconvolution is for (step 5 measures the timing it needs); padded past code + readout, every line
catches all of it and per-row and per-frame become the same.

Set the monitor's `total_duration_s` to a whole number of refreshes per bit (16 bits at 4 refreshes
= 1.0667 s): the screen only changes on a refresh, so other lengths come out uneven.

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

## Files

| File | What |
| --- | --- |
| `global-shutter.py`, `rolling-shutter-per-frame.py`, `rolling-shutter-per-row.py` | deblur real captures by each method |
| `capture.py` | what those three run: grid from the calibration, monitor model, deblur, figures |
| `calibrate.py` | per-camera calibration json: first guess, hub, monitor (with a bit check), rolling shutter readout |
| `flutter.py` | the model: monitor code, line exposure weights, sensor noise, unwrap about the hub, forward model, Tikhonov, DPIR, blind speed |
| `real.py` | DNG loading, run log, hub from two stills, monitor waveform from a blade edge |
| `blender/render_fan.py` | blade as sharp linear frames at t = (k + 0.5) dt, plus the t = 0 reference |
| `demo.py` | all three methods plus constant light and a short exposure, speed estimate |
| `rows.py` | criterion 1: GS vs RS per-frame vs RS per-row by band of sensor lines, three codes, three readouts |
| `legibility.py`, `ocr.py` | criterion 2: fraction of blade characters tesseract reads, against fan speed and light |
| `coding_gain.py` | coded vs constant light vs denoised short exposure, over blur length and light level |
| `robust.py` | PSNR cost of black level, LCD lag, code timing, hub error, stepper motion (Tikhonov) |
| `tests/` | synthetic blade, no Blender needed: forward model, all three methods, DPIR, monitor fit, DNG loading |

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

## Capturing on the rig

What the simulation says the real captures need, and the Lab 4 lessons:

1. Dark room, black backdrop, manual exposure and gain (no auto anything), DNG only. Log every file
   with its condition as it is taken, and check clipping (under 0.1% of photosites at white) before
   leaving the lab.
2. Dark frames: 10 with the lens capped, at the capture exposure and gain.
3. Hub: fan stopped, constant light, one still; step the motor 100 full steps (about 18 deg; at least
   30), second still, then `calibrate.py hub`. Count the real steps per revolution once (28BYJ-48:
   about 2038, not 2048). In simulation a 4 deg turn found the hub to 0.4 px, and the deblur tolerates 5 px.
4. Reference: a sharp still of the stopped blade, the ground truth for PSNR and OCR.
5. Monitor: 10 coded captures of a plain blade (nothing printed within the smear of its leading edge,
   blade wider than the smear) at about 2.8 rpm, the code and exposure used for the real captures, then
   `calibrate.py monitor`. In simulation this recovered 80% of the 2.1 dB lost to a 3% black level and a
   3 ms LCD lag; one frame recovered none of it.
6. Captures: at least 10 per speed per condition (coded, constant, one-chip short), one monitor
   brightness throughout, commanded rpm recorded.
7. Rolling shutter: record the sensor mode (`picam2.camera_configuration()["raw"]`), and take the two
   stills for `calibrate.py readout` (step 5 above) with the same params as the real captures.
