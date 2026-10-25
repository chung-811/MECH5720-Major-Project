import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

CHARS = "ACDEFGHJKLMNPRTUVWXY34679"     # no 0/O, 1/I, 5/S, 8/B, 2/Z: those confuse OCR even when sharp


def read_line(img):
    """tesseract on one line of dark text on a light background"""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "l.png"
        im = np.pad(np.uint8(255 * np.clip(img / 0.8, 0, 1)), 20, constant_values=255)   # white margin
        Image.fromarray(im).save(f)
        out = subprocess.run(["tesseract", str(f), "-", "--psm", "7", "-c", f"tessedit_char_whitelist={CHARS}"],
                             capture_output=True, text=True).stdout
    return "".join(out.split())


def edits(a, b):
    """levenshtein distance"""
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def legibility(img, d):
    """fraction of the blade's characters read correctly, and what was read. d: the render's npz"""
    hub, h = d["hub"], float(d["text_h_px"])
    truth = str(d["text"]).split(",")
    read = []
    for line, r in zip(truth, d["text_r_px"]):
        row, half = int(round(hub[0] + r)), int(round(0.38 * h * len(line) + 0.15 * h))
        read.append(read_line(img[row - int(0.9 * h):row + int(0.9 * h), int(hub[1]) - half:int(hub[1]) + half]))
    n = sum(map(len, truth))
    return max(0.0, 1 - sum(edits(a, b) for a, b in zip(read, truth)) / n), read
