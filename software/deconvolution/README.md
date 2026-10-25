# Deconvolution and velocity estimation

Deblurs coded-illumination captures of the spinning blade, by the three methods: global shutter,
rolling shutter per-frame and rolling shutter per-row. It also estimates the fan speed blind, and has
a Blender simulation to test all of it before the rig exists.

## Setup

The quick way: run `start-mac.command` or `start-windows.bat` from the repo root. It installs uv,
sets this folder up, downloads the model weights and runs `check.py`, which prints one line per check
and READY (or what to do). By hand:

Needs [uv](https://docs.astral.sh/uv/), which installs Python and every package itself:

```
curl -LsSf https://astral.sh/uv/install.sh | sh                          # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"                 # Windows
```

Then, in this folder:

```
uv sync          # about 1 GB of packages (PyTorch, CPU only), about 2 minutes the first time
uv run pytest    # 35 checks, about a minute; the first run downloads the 131 MB DRUNet weights
uv run python check.py   # plain-English: Python, packages, weights, reads a DNG, deblurs a test blade
```

If the tests pass, the pipeline works on your machine. Optional extras: Blender 5.2 for new renders,
and [Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) for the OCR legibility study.

## Real captures

Capture with the camview GUI's Coded Capture (`software/data_capture`). Each capture is a DNG plus a
json run log next to it, which gives the code, bit length, exposure, readout and timing, so nothing
about the code is typed in. Logs from `software/illumination/coded_illumination.py` also work. Keep
the images in raw sensor orientation (do not rotate them). On a colour sensor, coordinates are in the
half-size green image `real.load_dng` returns: `calibrate.py grid` shows them.

### The rig

Nothing bright may sit behind the blades: the deblur assumes everything in the blade's arc turns with
it, and a white stand behind the blades took a simulated deblur from 33 dB to 9 dB. Cover the front of
the stand in matte black (card or gaffer tape) as well as the backdrop. Put the text on a flat paddle
on each blade, different letters on each, at least 5 to 8 mm tall.

### What to capture, per camera (same gain throughout, nothing moved between shots)

| Shot | Used for |
| --- | --- |
| coded captures of the spinning fan, commanded rpm noted | the result |
| three stills of the stopped fan, each `s 400` on from the last with Ben's sketch (not 1359 steps: 120 deg looks the same on a 3-blade fan) | `calibrate.py auto`: hub, blade radii, turn direction, and a check nothing moved |
| about 10 coded captures of a blade with nothing printed near its leading edge, each started from the same blade angle (global shutter) | monitor (optional) |
| a still of the stopped fan in constant light, in the same setup as the captures | `--reference` (criterion 1 scoring) |

Use `code_16bit_4refresh.json` (and `flat_16bit_4refresh.json` for steady light) in the camera GUI: 16
bits of exactly 4 refreshes of a 60 Hz monitor (1.0667 s). The screen only changes on a refresh, so
the 1 s files give bits of 3.75 refreshes, shown unevenly.

### Steps

The only values you type are the stepper turn between the two hub stills and the fan speed.

```
# 1. calibration from three stills of the stopped fan, each `s 400` on (Ben's sketch half-steps: about
#    4076 steps per rev; take up the gearbox slack with `s 200` first). Finds the hub, the blade radii and which way the fan turns, no guess needed, and
#    warns if the pairs of stills disagree (something moved: retake them). Two stills also work, unchecked.
#    Check out/<still1>_auto.png: the + on the axle, the blade between the two circles
uv run python calibrate.py auto gs.json gs_still0.dng gs_still1.dng gs_still2.dng --steps 400 --steps-per-rev 4076
uv run python calibrate.py auto rs.json rs_still0.dng rs_still1.dng rs_still2.dng --steps 400 --steps-per-rev 4076

# 2. monitor (optional): black level, LCD lag and a check every bit showed as coded, from coded
#    captures of a plain blade wider than its smear, all started from the same blade angle (or just
#    one). Thin rods smear wider than themselves, so on them it refuses to save: skip it, the deblur
#    does without (in simulation, an unmodelled 2% black level and 4 ms lag left 33 dB)
uv run python calibrate.py monitor plain*.dng --calib gs.json rs.json --rpm 5

# 3. deblur. --rpm is the commanded speed, signed as step 1 printed. Writes out/<name>_<method>.png
#    (capture beside result, with the arc it deblurred) and .npy (the deblurred image)
uv run python global-shutter.py            cap*.dng --calib gs.json --rpm 5 --reference still_gs.dng
uv run python rolling-shutter-per-frame.py cap*.dng --calib rs.json --rpm 5 --reference still_rs.dng
uv run python rolling-shutter-per-row.py   cap*.dng --calib rs.json --rpm 5 --reference still_rs.dng
```

The arc to deblur is found in each capture (the brightest blade and its smear, between the dark gaps),
as the blade stops at a different angle every time. `--check` draws it on each capture without
deblurring. If `auto` fits badly it says so; then set things by hand: `calibrate.py grid cap.dng`
saves the capture with a pixel grid, read the hub (row, col), radii and arc off it, and
`calibrate.py init gs.json --hub ROW COL --radius R0 R1 --arc A0 A1` (degrees counter-clockwise from
+col; `--arc` fixes the arc for every capture). `calibrate.py hub` then refines the hub.

`--reference` turns the still to where the deblurred blade is and reports PSNR, SSIM and PSNR by band
of sensor lines (criterion 1), with mean and std over all captures given. `--find-speed` estimates the
speed blind (criterion 3) and deblurs at the estimate. It refines a speed already known to within 20%
(the commanded rpm): it searches 0.8 to 1.2 x `--rpm`, and says so if the best is at the edge. Further off
it can settle on a false dip (slow speeds explain a capture almost as well), so check `--rpm` first.
Every deblur also checks which way the fan turns: if the other sign of `--rpm` fits clearly better, it
deblurs at that and says so. `--dark dark*.dng`
subtracts dark frames (same exposure and gain, monitor black). `--solver tikhonov` is the fast
baseline. `-h` on any script lists its options.

Rolling shutter timing: the GUI pads the exposure by the readout, so every sensor line sees the whole
code, the deblur takes the readout from the run log, and per-row and per-frame give near enough the same result.
To make per-row matter, capture with less padding and measure the timing first, from the stopped fan
under the code and under constant light:
`uv run python calibrate.py readout rs.json still_coded.dng still_constant.dng`.

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
| `calibrate.py` | per-camera calibration json: auto from two stills, or by hand (grid, init, hub); monitor (with a bit check); rolling shutter readout |
| `flutter.py` | the model: monitor code, line exposure weights, sensor noise, unwrap about the hub, forward model, Tikhonov, DPIR, blind speed |
| `real.py` | DNG loading, run log (both capture tools), hub and radii from two stills, arc of each capture, monitor waveform from a blade edge |
| `blender/render_fan.py` | blade as sharp linear frames at t = (k + 0.5) dt, plus the t = 0 reference |
| `demo.py` | all three methods plus constant light and a short exposure, speed estimate |
| `rows.py` | criterion 1: GS vs RS per-frame vs RS per-row by band of sensor lines, three codes, three readouts |
| `legibility.py`, `ocr.py` | criterion 2: fraction of blade characters tesseract reads, against fan speed and light |
| `coding_gain.py` | coded vs constant light vs denoised short exposure, over blur length and light level |
| `robust.py` | PSNR cost of black level, LCD lag, code timing, hub error, stepper motion (Tikhonov) |
| `check.py` | is this computer ready: one line per check, then READY or what to do |
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

## Lab day checklist

With Ben's fan (`software/motor_control/test.ino`, Arduino Uno + ULN2003 + 28BYJ-48). His sketch takes
commands in the Arduino Serial Monitor at 9600 baud: `v 5` sets 5 rpm, `s 400` turns 400 steps, `r 5`
turns 5 revolutions. It half-steps, so one revolution is about **4076** steps (not 2038, the full-step
count). Opening the serial port resets the Uno and replays its startup demo: wait for the help text.

1. Rig: dark room or black backdrop, **matte black over the front of the white stand** (a bright still
   surface behind the blades took a simulated deblur from 33 dB to 9 dB), a white card paddle with
   letters on each blade (different letters, at least 5 to 8 mm tall), camera about 33 cm away with the
   fan filling the frame. Manual exposure and gain, nothing automatic.
2. Software: run `start-mac.command` / `start-windows.bat` from the repo root: READY, then pick the camera.
3. Gearbox slack: `s 200` once before the first still, then step every still the same way.
4. Calibration stills, per camera, fan stopped, constant light, same gain, nothing touched in between:
   still, `s 400`, still, `s 400`, still. Then
   `calibrate.py auto gs.json s0.dng s1.dng s2.dng --steps 400 --steps-per-rev 4076`. Not 1359 steps:
   that is 120 deg, where a 3-blade fan looks the same.
5. Reference: one still of the stopped fan in constant light, same setup (for `--reference`).
6. Check before collecting everything: two or three coded captures (`code_16bit_4refresh.json` in the
   camera GUI), then `global-shutter.py ... --rpm 5 --reference ref.dng`. Look for readable letters and no
   "pairs disagree" or "turns the other way" message. Fix the rig now if not.
7. Dataset: for each speed (`v 3`, `v 5`, `v 8`, `v 10`, then `r 5` to spin, capture while it turns) at
   least 10 coded captures, 10 under steady light (`flat_16bit_4refresh.json`) and 10 short plain
   exposures, for criterion 2. Write down the commanded rpm for every file. Ben's sketch counts 4096
   steps per rev where the motor has about 4076, which alone makes `v 5` about 0.5% fast; the Uno's time
   per step slows it a little, by an amount the code cannot tell. Time a few revolutions if criterion 3
   needs the true speed to better than 1%.
8. Rolling shutter camera: steps 4 to 7 again.
9. Before leaving: copy `output/` to the shared drive; note gain, speeds and anything that went wrong.
