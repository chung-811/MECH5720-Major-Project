import glob
import json

import numpy as np
import rawpy
from scipy.optimize import minimize
from scipy.signal import savgol_filter

import flutter as F

# real captures: linear raw loading, hub calibration from two stepped frames, light waveform from an edge


def files(patterns):
    """paths from names or patterns like cap*.dng, sorted (Windows shells leave patterns for us to expand)"""
    out = []
    for p in map(str, patterns):
        hits = sorted(glob.glob(p))
        if not hits:
            raise SystemExit(f"no file matches {p}")
        out += hits
    return out


def load_dng(path):
    """linear image in [0, 1] of white: per-channel black level off, first green sublattice for a colour
    sensor (half size, so a rolling shutter line here is two sensor lines), whole frame for a mono one.
    Keep it in raw sensor orientation: the rolling shutter timing follows sensor lines"""
    with rawpy.imread(str(path)) as raw:
        bayer = raw.raw_image_visible.astype(np.float64)
        colours = np.asarray(raw.raw_colors_visible)
        black = np.asarray(raw.black_level_per_channel, dtype=np.float64)
        white = float(raw.white_level)
        desc = raw.color_desc.decode()
    img = (bayer - black[colours]) / (white - black.mean())
    if len(set(desc)) == 1:
        return img
    r, c = next((r, c) for r in range(2) for c in range(2) if desc[colours[r, c]] == "G")   # G and G2 differ
    return img[r::2, c::2]


def run_log(path):
    """(code, chip, t_exp) from the JSON that software/illumination/coded_illumination.py writes next to
    each DNG. chip is total_duration_s / bits there, not one refresh: pass it to F.light(..., chip=chip)"""
    with open(path) as f:
        d = json.load(f)
    code = "".join(map(str, d["sequence_played"]))
    return code, d["params"]["bit_duration_s"], d["actual_exposure_us"] / 1e6


def mean_frames(paths):
    return np.mean([load_dng(p) for p in paths], axis=0)


def find_hub(f0, f1, angle, guess, r, th):
    """rotation centre (row, col) that best maps frame f0 onto f1 turned counter-clockwise by `angle`
    radians (stepper steps x step angle). r, th: radii and angles over the blade, about the guess.
    Needs a turn of a few degrees: in simulation 4 deg gave 0.4 px, 0.02 deg diverged"""
    def cost(c):
        a = F.polar(f0, c, r, th)[0]
        b = F.polar(f1, c, r, th + angle)[0]
        return np.mean((a - b) ** 2)
    res = minimize(cost, np.asarray(guess, float), method="Nelder-Mead",
                   options=dict(xatol=0.05, fatol=1e-12, initial_simplex=np.asarray(guess) + [[0, 0], [20, 0], [0, 20]]))
    return tuple(res.x)


def kernel_from_edge(img, hub, r, th, omega, t, window=9):
    """monitor brightness at times t, from the leading (counter-clockwise) edge of a PLAIN blade (no
    print within the blur of the edge) in a coded capture. Along a ring the edge smears forward as
    b = sum_k w_k [theta - omega t_k < edge], so -db/dtheta past the edge is the light at
    t = (theta - edge) / omega. Single rings are too noisy to differentiate, so each ring is aligned on
    where its smear falls to half its peak (the moment half the light is in, the same t on every ring),
    the rings are averaged, and the average is differentiated once. th: from inside the blade to past
    the end of the smear. omega: the known (commanded) speed of this capture. window: samples in the
    Savitzky-Golay derivative, keep it well under a chip (omega x chip / dth samples).
    Clockwise (omega < 0) is mirrored left-right first, so it reads as counter-clockwise.
    Returns L(t) with its brightest level at 1, 0 outside the measured smear"""
    if omega < 0:              # angle a becomes pi - a, so th reverses to stay increasing
        img, hub, th, omega = img[:, ::-1], (hub[0], img.shape[1] - 1 - hub[1]), np.pi - th[::-1], -omega
    P = F.polar(img, hub, r, th)[0]
    idx = np.arange(P.shape[1], dtype=float)
    half = []
    for p in P:
        top = int(np.argmax(p))
        k = top + int(np.argmax(p[top:] < 0.5 * p[top]))                      # first sample below half
        half.append(k - 1 + (p[k - 1] - 0.5 * p[top]) / (p[k - 1] - p[k]))     # interpolated crossing
    P = np.array([np.interp(idx + c - np.mean(half), idx, p) for p, c in zip(P, half)])
    h = -savgol_filter(P.mean(0), window, 2, deriv=1)
    k = int(np.argmax(h > 0.5 * h.max()))                                     # first rise of light: t = 0
    lag = (np.arange(len(h)) - k) * (th[1] - th[0])
    L = np.interp(omega * np.asarray(t), lag, h, left=0, right=0)
    return L / np.percentile(L[L > 0], 99)


def fit_monitor(L, t, dt, code, chip=F.CHIP):
    """(black, tau, t0), L_model: the designed code through a black level, a first-order LCD lag tau and a
    timing offset t0, fitted to a measured waveform L on the uniform grid t. Three numbers are far better
    determined by a noisy measurement than the waveform itself. lcd is linear and keeps constants, so
    the model is black + (1 - black) lcd(code): linear in black, grid search over tau and t0"""
    inside = t < len(code) * chip
    best = None
    for t0 in np.linspace(-0.5, 0.5, 21) * chip:
        for tau in np.geomspace(2e-4, 1e-2, 25):
            c = F.lcd(F.light(t, code, chip, t0=t0), dt, tau)
            a = (1 - c)[inside]
            black = float(np.clip(a @ (L - c)[inside] / (a @ a), 0, 0.5))
            err = np.sum((black * (1 - c) + c - L)[inside] ** 2)
            if best is None or err < best[0]:
                best = (err, black, tau, t0)
    _, black, tau, t0 = best
    return (black, tau, t0), F.lcd(F.light(t, code, chip, black=black, t0=t0), dt, tau)
