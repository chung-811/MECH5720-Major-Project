"""The report's rig numbers, one CSV row per capture: each capture through the deblur exactly as
global-shutter.py or rolling-shutter-per-*.py run it (--method; static scene out, arc found, measured monitor,
run log timing, DPIR, scored on the reference, also by band of sensor lines), then
  - the letter on the deblurred card, read by OCR (ocr.read_letter), against the same card on the reference
    (the ground truth) and on the capture itself before deblurring
  - the speed from the blur (capture.speed_from_blur), against the commanded one: global shutter, or a
    rolling shutter whose every line saw the whole code (the capture GUI pads for that); else left blank

    uv run python rig_eval.py FOLDER [FOLDER ...] --calib out/gs_stills/calib.json \\
        --reference REF.dng --date 2026-10-28 --out rig.csv

The speed comes from each folder's name (gs_-3rpm) unless --rpm is given; give it signed, the way the fan
turns, for steady light, whose blur cannot show which way that is. Rows are added to --out, and captures
already in it are skipped, so a run can be resumed. A capture the deblur cannot use is skipped, as
capture.py does, and its warnings are printed.
"""
import argparse
import contextlib
import csv
import io
import json
import re
from pathlib import Path

import numpy as np

import capture
import ocr
import real

FIELDS = ["date", "method", "light", "rpm", "flipped", "capture", "psnr_db", "ssim", "bands_db", "blade_margin",
          "static_scale", "late_bits", "card", "read", "read_blurred", "speed_rpm", "speed_at_edge"]

ap = argparse.ArgumentParser(description="rig numbers for the report: PSNR, OCR and speed from the blur, per capture")
ap.add_argument("folders", nargs="+", help="capture folders, each named with its speed")
ap.add_argument("--calib", required=True, help="calibration json (calibrate.py auto, with a static scene)")
ap.add_argument("--reference", required=True, help="still of the stopped fan, the ground truth")
ap.add_argument("--rpm", type=float, help="the speed, signed (default: from each folder's name)")
ap.add_argument("--method", choices=("global", "rs-row", "rs-frame"), default="global")
ap.add_argument("--date", required=True, help="the session, written in every row")
ap.add_argument("--out", required=True, help="CSV to add the rows to; captures already in it are skipped")
ap.add_argument("--cards", help="also save each capture's card (as captured, deblurred, on the still) to this folder, "
                "one npz per capture: the report's rig figure")
args = ap.parse_args()

calib = json.loads(Path(args.calib).read_text())
if not calib.get("static"):
    raise SystemExit(f"{args.calib} has no static scene: run calibrate.py auto with 3 stills")
static = np.load(Path(args.calib).parent / calib["static"])
hub, r1 = calib["hub"], calib["radius_px"][1]
ref = real.remove_static(real.load_dng(args.reference), static, hub, r1)[0]
out = Path(args.out)
done = set()
if out.exists():
    with out.open() as f:
        done = {(r["date"], r["method"], r["capture"]) for r in csv.DictReader(f)}
with out.open("a", newline="") as f:
    rows = csv.DictWriter(f, FIELDS)
    if not done and out.stat().st_size == 0:
        rows.writeheader()
    for folder in map(Path, args.folders):
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*rpm", folder.name)
        rpm = args.rpm if args.rpm is not None else float(m.group(1)) if m else None
        if not rpm:
            raise SystemExit(f"{folder}: give the speed (--rpm), the folder name has none")
        for path in sorted(folder.glob("*.dng")):
            if (args.date, args.method, path.stem) in done:
                continue
            log = path.with_suffix(".json")
            if not log.exists():
                raise SystemExit(f"{log} not found: the json run log the capture tool saves next to the DNG")
            code, chip, t_exp = real.run_log(log)
            timing = real.log_timing(log)
            if timing.get("aborted"):
                print(f"{path.name}: skipped, the run was stopped early")
                continue
            if args.method != "global" and "readout_s" not in timing:
                raise SystemExit(f"{log}: no readout in it (not from the capture GUI): rolling shutter needs it")
            said = io.StringIO()
            try:
                with contextlib.redirect_stdout(said):                 # the deblur's progress lines
                    img, a_static, _ = real.remove_static(real.load_dng(path), static, hub, r1)
                    cal = capture.for_capture(calib, img)
                    x, scene, used = capture.deblur(img, cal, code, chip, t_exp, rpm, args.method, "dpir",
                                                    logged=timing)
                    truth, p, q, bands = capture.compare(x, scene, ref, img)
                rc, thc = ocr.card(truth, scene.hub, scene.r, scene.th)
            except ValueError as err:                                  # as capture.main: this capture only
                print(f"{path.name}: skipped, {err}")
                continue
            for line in said.getvalue().splitlines():                  # but never hide a warning
                if "***" in line or "warning" in line:
                    print(f"{path.name}: {line.strip()}")
            half = ocr.card_half(scene.r[-1])
            read = {k: ocr.read_letter(ocr.card_crop(v, scene.hub, rc, thc, half)) for k, v in
                    (("card", truth), ("read", x), ("read_blurred", img))}
            if args.cards:                     # a wider square than OCR's, for context, blade pointing down
                Path(args.cards).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(args.cards) / f"{path.stem}.npz", **{
                    k: ocr.card_crop(np.clip(v, 0, None), scene.hub, rc, thc, round(2.2 * half)).astype(np.float16)
                    for k, v in (("capture", img), ("deblurred", x), ("still", truth))})
            # one line's timing is every line's when each saw the whole code: it starts after the last line
            # opens and ends before the first one closes
            whole = args.method == "global" or timing["readout_s"] <= timing["t0"] and \
                timing["t0"] + len(code) * chip <= t_exp
            speed, edge = capture.speed_from_blur(img, cal, ref, code, chip, t_exp, used) if whole else (None, None)
            row = dict(date=args.date, method=args.method, light="steady" if set(code) == {"1"} else "coded",
                       rpm=f"{used:g}", flipped=int(np.sign(used) != np.sign(rpm)), capture=path.stem,
                       psnr_db=f"{p:.2f}", ssim=f"{q:.3f}", bands_db=";".join(f"{v:.2f}" for v in bands),
                       blade_margin=f"{scene.blade_margin:.2f}", static_scale=f"{a_static:.3f}",
                       late_bits=timing.get("late_bits", 0), **read,
                       speed_rpm="" if speed is None else f"{speed:.3f}", speed_at_edge="" if edge is None else int(edge))
            rows.writerow(row)
            f.flush()
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
