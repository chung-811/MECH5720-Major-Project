# MECH5720 Major Project: Flutter Shutter/Coded Illumination

## Team and Roles
- Jack Thompson (490386475): Coded sequence design and deblurring pipeline
- Oliver Turner (520419913): Camera control and calibration in setup
- Krish Singh (530258892): Deconvolution & velocity estimation pipeline
- Daniel Chung (530585097): Monitor animation, velocity estimation algorithm
- Ben Ritchie (520479733): Test rig design and implementation

## Goal
Our system will compare three methods for flutter shutter photography with coded illumination: rolling shutter
deconvolved on a per-frame basis, rolling shutter deconvolved per-row and global shutter deconvolved per-frame.
Flutter shutter cameras break the light/motion blur trade-off that conventional cameras face when imaging moving
objects. Global shutter cameras overcome the artefacts rolling shutters exhibit from their row-by-row capture. The
rolling shutter artefacts can also be accounted for by applying the deconvolution on a row-by-row basis.
The recovered sharp images from flutter shutter cameras are useful for high speed applications. This project will
emulate the flutter shutter method with coded illumination to image fast-moving subjects and compare the performance
of the three different methods mentioned above.

## Approach
A fan with characters on the blades will be spinning at a controllable rate and imaged by a coded illumination rig
including a monitor and a camera. Within one exposure on the camera, the monitor will flash in a coded binary
sequence, producing a known blurring effect in the recorded image. Both a global shutter and a rolling shutter camera
will be used to image the subject and the images will be put through a processing pipeline which de-convolves the
image using the known shutter sequence to undo the motion blur and estimate a velocity for each of the 3 methods
discussed above. The results can then be compared.

## Success Criteria
All metrics are reported as mean ± std over ≥10 repeated captures per speed; ground-truth speed from the motor
step rate. Three-way comparison:
1. For each proposed method, we plot image quality against a photo of the stationary fan (PSNR/SSIM) for each
band of rows from top to bottom of the image. Target: per-row rolling shutter within 1 dB of global shutter across
all rows, while per-frame rolling shutter degrades away from the rows its code was timed for.
2. Fan legibility vs. speed: fraction of fan blade characters read correctly by OCR for coded light then deblurred,
constant light then deblurred, and a very short exposure with brightness boosted. Target: ≥90% read correctly
with coded light at speeds where constant light gives less than 50%.
3. Blind speed estimate: rotation speed estimated from the length of the blur arc, plotted against the true speed.
Target: within 5%.
Does the system behave as designed? We photograph a bright dot on a moving blade to see the blur pattern directly,
and check that it matches our flashing sequence and keeps the fine detail that constant light loses.

## Fallback
If a global shutter camera cannot be sourced and mounted by Week 10, we run the pipeline on the kit rolling shutter
camera alone and compare coded against constant monitor illumination. If per-row deconvolution fails, we deconvolve
each rolling shutter frame with a single kernel, keeping the full code inside the window where all rows are exposing. If
the stepper rig cannot hold a constant speed, we switch to a slow fan target with its speed measured independently
by high-frame-rate video. If blind speed estimation fails, we deblur using the commanded stepper speed and report
PSNR/SSIM against blur length. At minimum, we show that the measured kernel of a point target on the rig matches
the designed code and has no spectral nulls, unlike the box kernel.
6. Hardware
We will build: A coded illumination rig using a monitor displaying black/white screens in a designed binary sequence
while the global shutter camera and rolling shutter camera hold one continuous exposure, so the target’s motion is
convolved with a broadband code instead of a simple box blur. This will be tested primarily with a spinning fan
controlled via a stepper motor, so that the speed can be controlled and recorded.

## Simulation (Blender)

Sharp sub-frames from Blender, coded light and rolling shutter applied in numpy, then deblurred.
Blender's own shutter curve and rolling shutter are not used: the curve leaks about 10% of the
light into off chips, and its rolling shutter squeezes the whole code into every row instead of
giving each row its own slice of one flashing monitor.

Scene: one 50 mm blade filling the frame height, hub above the frame, black backdrop, 0.35 rpm.

```
uv sync
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

# criterion 2: OCR legibility against fan speed (tesseract on the PATH)
for p in 0.35:0.0083333 0.7:0.0041667 1.4:0.0020833 2.8:0.0010417; do blender -b -P blender/render_fan.py -- --text K7R,MX4,PAT,H9W,3EV,NJ6 --text_h 0.005 --rpm ${p%%:*} --dt ${p##*:} --out renders/ocr_${p%%:*}.npz; done
uv run python legibility.py renders/ocr_0.35.npz renders/ocr_0.7.npz renders/ocr_1.4.npz renders/ocr_2.8.npz

# robustness: finer sub-frames, smooth and 28BYJ-48 stepped
blender -b -P blender/render_fan.py -- --dt 0.0020833 --out renders/blade480.npz
blender -b -P blender/render_fan.py -- --dt 0.0020833 --steps_per_rev 2038 --out renders/blade480_steps.npz
uv run python robust.py renders/blade480.npz renders/blade480_steps.npz
```

Process captures in raw sensor orientation: the per-row timing follows sensor lines, so rotate for
display only after deblurring.

| File | What |
| --- | --- |
| `blender/render_fan.py` | blade as sharp linear frames at t = (k + 0.5) dt, plus the t = 0 reference |
| `flutter.py` | monitor code, line exposure weights, sensor noise, arc unwrap about the hub, deblur, blind speed |
| `demo.py` | GS per-frame vs RS per-frame vs RS per-row, constant light, short exposure, speed estimate |
| `coding_gain.py` | coded vs constant light vs denoised short exposure, over blur length and light level |
| `ocr.py` | tesseract per printed line, fraction of characters read correctly (edit distance) |
| `legibility.py` | criterion 2: legibility of coded, constant and short captures against fan speed and light |
| `real.py` | real captures: DNG loading, hub from two stepped stills, monitor waveform from a blade edge |
| `rows.py` | GS vs RS per-frame vs RS per-row by band of sensor lines, 52/31/15-chip codes, three readouts |
| `robust.py` | PSNR cost of black level, LCD lag, code timing, hub error, stepper motion |

### Deconvolution

`flutter.operator` builds the forward model as one sparse matrix; `flutter.dpir` inverts it with DPIR
(Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", TPAMI 2022): half-quadratic
splitting with a pretrained DRUNet denoiser as the prior, through the deepinv library. It needs no
tuning: the noise level is estimated from the capture. `flutter.deblur` (per-ring Tikhonov with a
gradient penalty) stays as the fast baseline and drives the blind speed search.

Benchmark, global shutter, Raskar 52-chip code, 0.35 rpm (PSNR over the blade):

| Solver | flat blade + letters | photo on the blade |
| --- | --- | --- |
| Tikhonov, best lam | 29.4 | 31.1 |
| TV + [0, 1] box, Chambolle-Pock (pyproximal), best lam | 41.1 | 34.6 |
| DPIR, no tuning | 44.3 | 36.3 |

TV smears thin lines (the tripod legs); DPIR keeps them. With DPIR, coding the light only pays when
the blur is long or the light is low: at 25 px of blur in full light constant light ties it, at
50 to 100 px or a tenth of the light coded wins by 3 to 7 dB (`coding_gain.py`).

One solver serves all three methods; they differ only in which exposure window each polar sample
gets. Blade geometry and sensor noise numbers are placeholders until the fan and cameras are measured.

## Capturing on the rig

What the simulation says the real captures need, and the Lab 4 lessons:

1. Dark room, black backdrop, manual exposure and gain (no auto anything), DNG only. Log every file
   with its condition as it is taken, and check clipping (under 0.1% of photosites at white) before
   leaving the lab.
2. Dark frames: 10 with the lens capped, at the capture exposure and gain.
3. Hub: fan stopped, constant light, one still; step the motor 100 full steps (about 18 deg; at least
   30), second still. `real.find_hub(a, b, steps * 2 * pi / steps_per_rev, guess, r, th)`. Count the
   real steps per revolution once (28BYJ-48: about 2038, not 2048). In simulation a 4 deg turn found the
   hub to 0.4 px, and the deblur tolerates 5 px.
4. Reference: a sharp still of the stopped blade, the ground truth for PSNR and OCR.
5. Monitor: 10 coded captures of a plain blade (nothing printed within the smear of its leading edge)
   at about 2.8 rpm, the code and exposure used for the real captures. `real.kernel_from_edge` on their
   mean, then `real.fit_monitor`. In simulation this recovered 80% of the 2.1 dB lost to a 3% black
   level and a 3 ms LCD lag; one frame recovered none of it.
6. Captures: at least 10 per speed per condition (coded, constant, one-chip short), one monitor
   brightness throughout, commanded rpm recorded.
7. Rolling shutter: record the sensor mode (`picam2.camera_configuration()["raw"]`), readout direction
   and line time. Deblur in raw sensor orientation; on the green sublattice a line is two sensor lines.
