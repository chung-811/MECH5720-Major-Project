"""is this computer ready? Checks Python, the packages, the DRUNet weights (downloaded here, once, so
game day needs no internet for them), reading a real DNG, and a deblur of the synthetic test blade.
Prints one line per check and READY, or what to do. Exit code 0 only when every check passes.

    uv run python check.py
"""
import contextlib
import io
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DNG = HERE.parent / "illumination/data/coded/coded_20261019_220955.dng"


def line(name, ok, note=""):
    print(f"{name + ' ':.<26} {'ok' if ok else 'FAILED'}{f' ({note})' if note else ''}", flush=True)
    return ok


def main():
    if not line("Python", sys.version_info >= (3, 11), sys.version.split()[0]):
        return "Python 3.11 or newer is needed: run the setup script again, uv installs it"
    print("Loading packages (the first time after a restart or a pull, macOS scans them: up to a minute,"
          " please wait)...", flush=True)
    try:
        import numpy, scipy, skimage, rawpy, matplotlib, torch, deepinv  # noqa: F401
    except ImportError as err:
        line("Packages", False, str(err))
        return "a package is missing: run `uv sync` in software/deconvolution"
    line("Packages", True, f"torch {torch.__version__}, deepinv {deepinv.__version__}")

    import flutter as F
    try:
        t = time.time()
        F._drunet()
        line("Deblur model weights", True, f"{time.time() - t:.0f} s, kept for next time")
    except Exception as err:                       # download or load failure, reported as is
        line("Deblur model weights", False, str(err)[:120])
        return "could not get the DRUNet weights (131 MB, downloaded once): connect to the internet and run again"

    import real
    if DNG.exists():
        img = real.load_dng(DNG)
        if not line("Reads a real DNG", img.ndim == 2 and img.size > 0, f"{img.shape[1]} x {img.shape[0]}"):
            return f"{DNG.name} did not load as an image: the rawpy install is broken, run `uv sync` again"
    else:
        line("Reads a real DNG", True, "skipped, no sample DNG in this checkout")

    sys.path.insert(0, str(HERE / "tests"))
    import numpy as np
    import capture
    import test_pipeline as T                       # the synthetic blade the tests use
    img, truth = T.simulate()
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):         # the deblur's own progress lines
        x = capture.deblur(img, T.calib(), T.CODE, T.CHIP, T.T_EXP, T.OMEGA * 60 / (2 * np.pi), "global", "dpir")[0]
    gain = T.blade_psnr(x, truth) - T.blade_psnr(img, truth)
    if not line("Deblurs a test blade", gain > 10, f"+{gain:.1f} dB over the blurred capture, {time.time() - t:.0f} s"):
        return "the deblur ran but did not sharpen the test blade: run `uv run pytest` to see which step fails"
    return None


if __name__ == "__main__":
    try:
        problem = main()
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C) before the checks finished: nothing is wrong, run it again and let it finish")
        sys.exit(1)
    print("\nREADY: see the README's \"Lab day\" section" if problem is None else f"\nNOT READY: {problem}")
    sys.exit(0 if problem is None else 1)
