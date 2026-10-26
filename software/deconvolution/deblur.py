"""One command for lab day: deblur a folder of coded captures.

    uv run python deblur.py gs_3rpm                  # output/gs_3rpm, 3 rpm read from the name
    uv run python deblur.py rs_5rpm --find-speed     # rolling shutter, also estimate the speed

The folders are the "Test folder" names typed into the camera GUI's Coded Capture section:
  gs_stills   fan stopped, flat_16bit_4refresh.json: capture, `s 400`, capture, `s 400`, capture, then one
              more capture (the reference). rs_stills: the same on the rolling shutter camera.
  gs_3rpm     fan spinning, code_16bit_best_4refresh.json. Any name with the speed in it ("5rpm").
Stills are coded captures too (with steady light) so they are the same size and light as the real ones.

It calibrates from <camera>_stills once (calib.json, saved in that folder), then runs the deblur for the
camera (global shutter, or both rolling shutter methods) on every capture in the folder, scored against the
reference. Results in out/<folder>/. It only calls calibrate.py and the deblur scripts, so everything they
print (warnings included) shows here.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import real

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[1] / "output"           # where the camera GUI saves coded captures
SCRIPTS = {"gs": ["global-shutter.py"], "rs": ["rolling-shutter-per-row.py", "rolling-shutter-per-frame.py"]}


def captures(folder):
    """(the coded captures in a folder, oldest first; their camera, "gs" or "rs"). .npy for simulated ones"""
    found = sorted([*folder.glob("coded_*.dng"), *folder.glob("coded_*.npy")], key=lambda p: p.name)
    if not found:
        raise SystemExit(f"no coded captures (coded_*.dng) in {folder}")
    cams = {p.name.split("_")[1] for p in found}
    if len(cams) > 1 or not cams <= set(SCRIPTS):
        raise SystemExit(f"{folder} mixes cameras ({', '.join(sorted(cams))}): use one folder per camera")
    return found, cams.pop()


def run(*cmd):
    print("\n> " + " ".join(map(str, cmd)), flush=True)
    if subprocess.run([sys.executable, *map(str, cmd)], cwd=HERE).returncode:
        raise SystemExit(f"stopped: {cmd[0]} failed (see above)")


def main():
    ap = argparse.ArgumentParser(description="deblur a folder of coded captures in one go: see the top of this file")
    ap.add_argument("folder", help="the capture folder: a name in output/ (e.g. gs_3rpm) or a path")
    ap.add_argument("--rpm", type=float, help="fan speed (default: the number before 'rpm' in the folder name)")
    ap.add_argument("--stills", help="folder with the stills and reference (default: output/<camera>_stills)")
    ap.add_argument("--steps", type=float, default=400, help="motor steps between stills (default 400)")
    ap.add_argument("--steps-per-rev", type=float, default=4076, help="half-steps per turn (Ben's sketch: 4076)")
    ap.add_argument("--recalibrate", action="store_true", help="calibrate again even if calib.json exists")
    ap.add_argument("--find-speed", action="store_true", help="also estimate the speed from each capture")
    ap.add_argument("--solver", choices=("dpir", "tikhonov"), default="dpir")
    ap.add_argument("--out", help="results folder (default: out/<folder>)")
    args = ap.parse_args()

    folder = Path(args.folder) if Path(args.folder).is_dir() else OUTPUT / args.folder
    if not folder.is_dir():
        have = sorted(p.name for p in OUTPUT.iterdir() if p.is_dir()) if OUTPUT.is_dir() else []
        raise SystemExit(f"no folder {args.folder} in {OUTPUT}. Folders there: {', '.join(have) or 'none'}")
    caps, cam = captures(folder)

    rpm = args.rpm
    if rpm is None:
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*rpm", folder.name, re.IGNORECASE)
        if not m:
            raise SystemExit(f"give the speed with --rpm (or put it in the folder name, e.g. {cam}_3rpm)")
        rpm = float(m.group(1))

    still_dir = Path(args.stills) if args.stills else OUTPUT / f"{cam}_stills"
    if not still_dir.is_dir():
        raise SystemExit(f"no stills folder {still_dir}: take the stills first (see the README, Lab day)")
    stills, still_cam = captures(still_dir)
    if still_cam != cam:
        raise SystemExit(f"the stills in {still_dir} are from the other camera ({still_cam}, captures are {cam})")
    if len(stills) < 3:
        raise SystemExit(f"{still_dir} has {len(stills)} capture(s): it needs 3 stills, then 1 reference")
    stills = stills[-4:]                          # retakes: the newest four
    ref = stills[3] if len(stills) == 4 else None
    if real.load_dng(stills[0]).shape != real.load_dng(caps[0]).shape:
        raise SystemExit(f"the stills and the captures are different sizes: take both with Coded Capture on the "
                         f"same camera")

    out = Path(args.out) if args.out else HERE / "out" / folder.name
    print(f"{len(caps)} {cam} capture(s) in {folder.name} at {rpm:g} rpm")
    print(f"stills: {', '.join(p.name for p in stills[:3])}; reference: {ref.name if ref else 'none (not scored)'}")
    calib = still_dir / "calib.json"
    if args.recalibrate or not calib.exists():
        run("calibrate.py", "auto", calib, *stills[:3], "--steps", args.steps, "--steps-per-rev", args.steps_per_rev,
            "--out", out)
    else:
        print(f"calibration: {calib} (from before; --recalibrate to redo it)")
    for script in SCRIPTS[cam]:
        cmd = [script, *caps, "--calib", calib, "--rpm", rpm, "--solver", args.solver, "--out", out]
        run(*cmd, *(["--reference", ref] if ref else []), *(["--find-speed"] if args.find_speed else []))
    print(f"\ndone: results in {out}")


if __name__ == "__main__":
    main()
