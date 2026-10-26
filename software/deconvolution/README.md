# Deblurring

Takes a coded photo of the spinning fan and makes it sharp, three ways: global shutter, rolling
shutter per-frame, rolling shutter per-row. It can also work out the fan speed from the photo.

## Setup

Run `start-mac.command` (Mac) or `start-windows.bat` (Windows) from the project folder, or in VS Code:
Run Task > **1. Lab: set up everything and connect a camera**. It ends with **READY**.

By hand, in this folder: `uv sync`, then `uv run python check.py`.

## Lab day

**0. Latest code.** In VS Code, click the branch name (bottom left) and pick `krish/sim-deconv`, then
pull (or `git checkout krish/sim-deconv && git pull`).

**1. The rig**
- Cover the front of the white stand in matte black. Anything bright behind the blades ruins the deblur.
- Tape a white card with letters (5 to 8 mm tall, different on each blade) to each blade.
- Dark room. Camera about 33 cm away, fan as large as possible but **all of it inside the picture**.

**2. The fan** (Arduino IDE > Serial Monitor, 9600 baud, Newline; wait for the help text)
- `s 200` once first, to take up slack in the gears
- `s 400` turns 400 steps (between stills)
- `v 3` sets 3 rpm, then `r 5` spins 5 turns

**3. The shots.** Everything is a **Coded Capture** (camera GUI > Stream Settings > Coded Capture). Type the
**Test folder** name and pick the **Code file** before each group. Same gain throughout, touch nothing else.

| Test folder | Code file | Fan | Do |
| --- | --- | --- | --- |
| `gs_stills` | `flat_16bit_4refresh.json` | stopped | Run, `s 400`, Run, `s 400`, Run, then one more Run (the reference) |
| `gs_3rpm` | `code_16bit_best_4refresh.json` | `v 3`, `r 5` | Run while it spins, up to 10 times |
| `gs_5rpm`, `gs_8rpm`, ... | same | `v 5`, ... | same; the speed must be in the folder name |
| `gs_3rpm_flat` | `flat_16bit_4refresh.json` | `v 3`, `r 5` | steady light, for comparison |

Rolling shutter camera: the same with `rs_stills`, `rs_3rpm`, ... and **Padding 0.25 s** in Coded Capture.

**4. Deblur.** VS Code: Run Task > **4. Deblur a capture folder**, type `gs_3rpm`. Or in a terminal (task 5):

```
uv run python deblur.py gs_3rpm
```

It calibrates from `gs_stills` (the first time), deblurs every capture in `gs_3rpm` and scores each one
against the reference. Results: `out/gs_3rpm/`, one picture per capture (capture, deblurred, reference).
Good: readable letters and no `***` lines. Add `--find-speed` to also estimate the speed.

Optional, once: a small white dot (about 3 mm) near the tip of one blade, for one coded capture. Its streak
shows the real light pattern (monitor lag and timing). Send it to Krish. Copy `output/` somewhere safe
before leaving.

## Commands

| Command | What it does |
| --- | --- |
| `deblur.py FOLDER` | everything below in one go, for a Test folder from the camera GUI |
| `calibrate.py auto CAL.json S1 S2 S3 --steps 400 --steps-per-rev 4076` | finds the fan centre and blade size from the 3 stills |
| `global-shutter.py CAPTURES --calib gs.json --rpm 5` | deblurs, global shutter |
| `rolling-shutter-per-row.py CAPTURES --calib rs.json --rpm 5` | deblurs, rolling shutter, each row its own timing |
| `rolling-shutter-per-frame.py CAPTURES --calib rs.json --rpm 5` | deblurs, rolling shutter, one timing for all rows |

Add to a deblur command:
- `--reference REF.dng` scores it against the still photo (PSNR, SSIM)
- `--find-speed` works out the speed from the photo (needs `--rpm` within 20%)
- `--check` only draws the area it would deblur
- `--solver tikhonov` faster, lower quality

All commands start with `uv run python`. `-h` on any of them lists every option.

## If something goes wrong

| You see | Do this |
| --- | --- |
| `NOT READY` | do what the line says |
| `no stills folder` | take the stills first (`gs_stills` or `rs_stills`, step 3) |
| `give the speed` | put it in the folder name (`gs_3rpm`) or add `--rpm 3` |
| `could not reach the Pi` | check the ethernet cable or hotspot; or use the Pi's IP: `PI=pi@IP ./start-mac.command` |
| `the pairs disagree` | something moved between the stills: take them again |
| `poor fit` | wrong steps: with Ben's sketch it is `--steps 400 --steps-per-rev 4076` |
| `turns the other way` | the sign of `--rpm` was wrong; it fixed it. Flip the sign next time |
| `edge of its search` | `--rpm` is more than 20% off the real speed |
| `over 180` or `no dark gap` | the blades blur into each other: slow the fan, or check the stand is black |
| blurry result, no warning | stand not black, fan over 10 rpm, or room light on |
| `not saved` from `calibrate.py monitor` | normal for thin blades: skip that step |

## Files

| File | What |
| --- | --- |
| `check.py` | is this computer ready |
| `deblur.py` | the one command for lab day |
| `calibrate.py` | makes the calibration file for each camera |
| `global-shutter.py`, `rolling-shutter-per-*.py` | the three deblur methods |
| `capture.py`, `real.py`, `flutter.py` | the code behind them |
| `tests/` | automatic checks: `uv run pytest` |
| `blender/`, `demo.py`, `rows.py`, `legibility.py`, `coding_gain.py`, `robust.py` | simulation studies |
| `STUDIES.md` | how it works, simulation results, what each fault costs |

## Extras

- Calibrate by hand if `auto` fails: `calibrate.py grid CAP.dng` (picture with a pixel grid), read the
  centre, radii and angles off it, then `calibrate.py init CAL.json --hub ROW COL --radius R0 R1 --arc A0 A1`.
- `calibrate.py monitor` measures the monitor's black level and lag. Only works on blades wider than
  their blur; the deblur is fine without it.
- `calibrate.py readout` measures rolling shutter timing. Not needed with the camera GUI, which already
  makes every row see the whole code (so per-row and per-frame come out the same).
