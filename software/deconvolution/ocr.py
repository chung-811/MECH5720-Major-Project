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


# ---------------------------------------------------------------- the rig: one letter card per blade
# Each blade of the rig carries a card with one capital letter near its tip. The card is found on the
# still of the stopped fan turned onto the deblurred blade (capture.compare's truth), the same square is cut
# from both, and each is read as one word. On the stills the letter alone (the biggest dark blob not touching
# the square's edge, which background and card edges always do) read right on all 22 sharp crops tried.

def card(truth, hub, r, th):
    """(radius, angle) of the card's centre: the bright print on the outer 45% of the rings"""
    from flutter import polar
    P = polar(truth, hub, r, th)[0]
    outer = r > r[0] + 0.55 * (r[-1] - r[0])
    w = np.clip(P[outer] - 0.5 * np.percentile(P[outer], 99), 0, None)
    if w.sum() <= 0:
        raise ValueError("no bright card on the outer part of the blade")
    return float((w.sum(1) * r[outer]).sum() / w.sum()), float((w.sum(0) * th).sum() / w.sum())


def card_half(r_outer):
    """the crop's half size (px) for a blade r_outer px long: the card is a fixed share of the blade, and
    42 px held every letter on the global shutter camera's 257 px blade without touching the edge"""
    return int(round(42 * r_outer / 257))


def card_crop(img, hub, rc, thc, half=42):
    """the (2 half + 1) px square centred on the card, turned so the blade points down: the rig's letters
    then read upright"""
    from scipy.ndimage import map_coordinates
    rows, cols = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    a = thc + np.pi / 2                                         # outward along the blade ends up pointing down
    x, y = cols, -rows
    X, Y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    return map_coordinates(img, [hub[0] - rc * np.sin(thc) - Y, hub[1] + rc * np.cos(thc) + X], order=1)


def read_letter(crop):
    """what tesseract reads (one word, capitals and digits) in the letter alone, "" when there is no letter:
    ink is darker than Otsu's threshold, and the letter is the biggest ink blob not touching the edge"""
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    c = crop / max(np.percentile(crop, 99), 1e-6)
    if np.ptp(c) == 0:
        return ""
    lab, n = ndimage.label(c < threshold_otsu(c))
    inner = sorted(set(range(1, n + 1)) - set(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]].tolist()))
    if not inner:
        return ""
    sizes = ndimage.sum(np.ones_like(c), lab, inner)
    k = inner[int(np.argmax(sizes))]
    if sizes.max() < 30:
        return ""
    rr, cc = np.nonzero(lab == k)
    glyph = np.pad(np.uint8(255 * (lab != k)[rr.min():rr.max() + 1, cc.min():cc.max() + 1]), 20, constant_values=255)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "g.png"
        Image.fromarray(glyph).resize((4 * glyph.shape[1], 4 * glyph.shape[0]), Image.NEAREST).save(f)
        try:
            out = subprocess.run(["tesseract", str(f), "-", "--psm", "8", "-c",
                                  "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"],
                                 capture_output=True, text=True, check=True).stdout
        except FileNotFoundError:
            raise SystemExit("tesseract is not installed (brew install tesseract)") from None
    return "".join(out.split())
