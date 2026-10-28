"""Simple text-only DNG analysis for the flutter-shutter success criteria.

- No manifest file
- No results written to disk
- Only console output
- Put the variable values in this file itself

Edit the variables at the top of this file, then run:
    uv run python analyse.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import capture
import flutter as F
import ocr
import real

ROOT = Path(__file__).resolve().parent.parent

# Set this to your calibration JSON. Keep it in this file so it is easy to edit.
CALIB_PATH = None  # e.g. ROOT / "some_folder" / "calib.json"

# Folder(s) to scan for .dng files. Edit these as needed.
DNG_FOLDERS = {
    "coded": ROOT / "illumination" / "data" / "coded",
    "constant": ROOT / "illumination" / "data" / "constant",
    "short": ROOT / "illumination" / "data" / "short",
}

# If the filename doesn't contain rpm, add it here.
FORCED_SPEEDS: dict[str, float] = {}
METHODS = ("global", "rs-frame", "rs-row")


def infer_speed(path: Path) -> float:
    if path.name in FORCED_SPEEDS:
        return FORCED_SPEEDS[path.name]
    m = re.findall(r"(\d+(?:\.\d+)?)\s*(?:rpm|RPM)", path.stem)
    if m:
        return float(m[0])
    m = re.findall(r"(\d+(?:\.\d+)?)", path.stem)
    if m:
        return float(m[-1])
    raise ValueError(f"Cannot infer speed from {path.name}. Add it to FORCED_SPEEDS.")


def mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def find_truth(dng_path: Path) -> Path | None:
    for cand in [
        dng_path.with_suffix(".npz"),
        dng_path.with_name(dng_path.stem + ".npz"),
        dng_path.with_name(dng_path.stem + "_truth.npz"),
        dng_path.with_name(dng_path.stem + "_render.npz"),
    ]:
        if cand.exists():
            return cand
    return None


def find_reference(dng_path: Path) -> Path | None:
    for cand in [
        dng_path.with_name(dng_path.stem + "_ref.dng"),
        dng_path.with_name("ref.dng"),
        dng_path.with_name("reference.dng"),
    ]:
        if cand.exists():
            return cand
    return None


def evaluate_dng(dng: Path, light: str, speed_rpm: float) -> dict:
    print(f"\n{dng.name} | light={light} | speed={speed_rpm:.2f} rpm")
    if CALIB_PATH is None or not CALIB_PATH.exists():
        print("  Skip: set CALIB_PATH in analyse.py to a calibration JSON before deblurring.")
        return {"light": light, "speed_rpm": speed_rpm}

    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    img = real.load_dng(dng)
    truth = find_truth(dng)
    ref = find_reference(dng)

    if light == "short":
        ocr_frac = np.nan if truth is None else float(ocr.legibility(img * 16.0, np.load(truth))[0])
        print(f"  short exposure OCR = {ocr_frac:.3f}")
        return {"light": light, "speed_rpm": speed_rpm, "ocr_fraction": ocr_frac}

    log = dng.with_suffix(".json")
    if not log.exists():
        print(f"  Skip: no run log next to the DNG ({log.name}).")
        return {"light": light, "speed_rpm": speed_rpm}

    code = F.RASKAR52 if light == "coded" else "1" * len(F.RASKAR52)
    _, chip, exposure = real.run_log(log)
    timing = real.log_timing(log)

    row: dict[str, object] = {"light": light, "speed_rpm": speed_rpm}
    for method in METHODS:
        x, scene, est_rpm = capture.deblur(
            img,
            calib,
            code,
            chip,
            exposure,
            speed_rpm,
            method,
            solver="tikhonov",
            find_speed=True,
            logged=timing,
        )

        psnr_val = np.nan
        ssim_val = np.nan
        if ref is not None:
            ref_img = real.load_dng(ref)
            turned = real.align(ref_img, x, scene.hub, scene.r, scene.th)[0]
            window = F.wrap(scene, np.ones((len(scene.r), len(scene.th)))) > 0.5
            scale = float(x[window] @ turned[window]) / max(float(turned[window] @ turned[window]), 1e-12)
            truth_img = turned * scale
            mask = window & (truth_img > 0.1 * np.percentile(truth_img[window], 99))
            psnr_val = float(__import__("skimage.metrics").metrics.peak_signal_noise_ratio(truth_img[mask], x[mask], data_range=1))
            ssim_val = float(__import__("skimage.metrics").metrics.structural_similarity(truth_img, x * window, data_range=1))

        ocr_frac = np.nan if truth is None else float(ocr.legibility(x, np.load(truth))[0])
        print(f"  {method:9s} PSNR={psnr_val:7.3f} dB SSIM={ssim_val:6.4f} OCR={ocr_frac:5.3f} blind={est_rpm:7.3f} rpm")
        row[method] = {"psnr": psnr_val, "ssim": ssim_val, "ocr_fraction": ocr_frac, "blind_rpm": est_rpm}

    return row


def aggregate(rows: list[dict]) -> None:
    print("\n=== Aggregated mean ± std ===")
    for light in ("coded", "constant", "short"):
        for method in METHODS + ("short",):
            subset = [r for r in rows if r.get("light") == light]
            if not subset:
                continue
            if method == "short":
                vals = [float(r["ocr_fraction"]) for r in subset if "ocr_fraction" in r and np.isfinite(r["ocr_fraction"])]
                mean, std = mean_std(vals)
                print(f"{light:8s} {method:9s} OCR = {mean:6.3f} ± {std:6.3f} (n={len(vals)})")
            else:
                for metric in ("psnr", "ssim", "ocr_fraction", "blind_rpm"):
                    vals = []
                    for r in subset:
                        item = r.get(method)
                        if isinstance(item, dict) and metric in item and np.isfinite(item[metric]):
                            vals.append(float(item[metric]))
                    mean, std = mean_std(vals)
                    unit = "dB" if metric == "psnr" else "" if metric in ("ssim", "ocr_fraction") else "rpm"
                    print(f"{light:8s} {method:9s} {metric:11s} = {mean:6.3f} ± {std:6.3f} {unit} (n={len(vals)})")


def main() -> None:
    print("Flutter shutter DNG analysis")
    print("Set CALIB_PATH and the folders at the top of analyse.py before running.")

    rows: list[dict] = []
    for light, folder in DNG_FOLDERS.items():
        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue
        dngs = sorted(folder.glob("*.dng"))
        if not dngs:
            print(f"No .dng files found in {folder}")
            continue
        for dng in dngs:
            try:
                speed = infer_speed(dng)
            except ValueError as err:
                print(f"  {err}")
                continue
            rows.append(evaluate_dng(dng, light, speed))

    if rows:
        aggregate(rows)
    else:
        print("No usable captures found.")


if __name__ == "__main__":
    main()
