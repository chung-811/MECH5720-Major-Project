# Results data

`rig.csv` has one row per capture on the rig. It holds every rig number in the main README and the
report. It was written by
[`software/deconvolution/rig_eval.py`](../software/deconvolution/rig_eval.py) from the captures in the
[dataset release](https://github.com/chung-811/MECH5720-Major-Project/releases/tag/dataset-v1).

| Column | Meaning |
| --- | --- |
| `date` | capture session |
| `method` | `global`, `rs-row` (rolling shutter, per row) or `rs-frame` (rolling shutter, per frame) |
| `light` | `coded` or `steady` |
| `rpm` | commanded fan speed (negative: clockwise in the image) |
| `flipped` | 1 if the speed's sign had to be flipped |
| `capture` | capture file name |
| `psnr_db`, `ssim` | deblurred blade against the stopped fan |
| `bands_db` | PSNR in 8 bands of sensor rows, first to last read (`nan`: no blade in that band) |
| `blade_margin` | how much better the chosen blade matched than the runner-up (near 1 is a guess; coded captures 1.7 to 3.0) |
| `static_scale` | brightness of the static scene (mount, wires) in this capture relative to the stills, before it is subtracted |
| `late_bits` | code bits the capture log records as shown more than 10 ms late |
| `card` | letter on the reference card (read by OCR from the still) |
| `read` | letter OCR reads after deblurring (empty: none) |
| `read_blurred` | letter OCR reads from the raw capture |
| `speed_rpm` | speed estimated from the blur |
| `speed_at_edge` | 1 if the speed search hit the edge of its range |

With the padded exposure, the rolling shutter per-row and per-frame rows agree to 0.01 dB; the figures
use the per-row rows.
