import glob
import json

import numpy as np
import rawpy
from scipy.ndimage import binary_dilation, map_coordinates, uniform_filter1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from skimage.transform import downscale_local_mean

import flutter as F

# real captures: raw loading, hub and blade radii from two stepped stills, the arc of each capture, monitor
# light from a blade edge and a bit check, rolling shutter readout from a still scene, and turning the
# reference still to the deblurred blade


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
    Keep it in raw sensor orientation: the rolling shutter timing follows sensor lines. A .npy is taken as
    already linear (simulated captures, tests)"""
    if str(path).endswith(".npy"):
        return np.load(path).astype(np.float64)
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
    """(code, chip, t_exp) from the json run log saved next to each DNG, by either capture tool:
    software/data_capture (the camview GUI's Coded Capture) or software/illumination/coded_illumination.py.
    chip is total_duration_s / bits, not one refresh: pass it to F.light(..., chip=chip)"""
    with open(path) as f:
        d = json.load(f)
    try:
        if "code" in d:                                          # data_capture GUI
            bits, chip = d["code"]["bits"], d["code"]["bit_duration_s"]
        else:                                                    # coded_illumination.py
            bits, chip = d["sequence_played"], d["params"]["bit_duration_s"]
        return "".join(map(str, bits)), chip, d["actual_exposure_us"] / 1e6
    except (KeyError, TypeError) as err:
        raise SystemExit(f"{path}: no {err} in it, so not a run log from the capture GUI or coded_illumination.py")


def log_timing(path):
    """what the data_capture GUI also logs, {} for coded_illumination.py logs: readout_s (first to last
    line, 1 / max fps of the sensor mode), t0 (when the code starts after the first line opens: the camera
    has been exposing one readout when the frame boundary arrives, then the display is told GO),
    late_bits and aborted"""
    with open(path) as f:
        d = json.load(f)
    if "code" not in d:
        return {}
    try:
        t0 = d["readout_s"] + (d["t0_monotonic_ns"] - d["frame_boundary_monotonic_ns"]) / 1e9
    except KeyError as err:
        raise SystemExit(f"{path}: no {err} in it, so not a complete run log from the capture GUI")
    return dict(readout_s=d["readout_s"], t0=t0, late_bits=d.get("late_bits", 0), aborted=d.get("aborted", False))


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


def check_bits(L, t, code, chip, black, t0):
    """indices of bits the measured light L shows wrongly: the mean over the second half of each bit
    (past the LCD's rise) against half way between black and white. A bit the screen showed twice and
    the one it then skipped come out as a wrong bit"""
    out = []
    for i, c in enumerate(code):
        sel = (t >= t0 + (i + 0.5) * chip) & (t < t0 + (i + 1) * chip)
        if sel.any() and (L[sel].mean() > (1 + black) / 2) != (c == "1"):
            out.append(i)
    return out


READOUT_MAX, T0_MAX = 0.15, 0.1          # s, fit_readout's search: past either, the fit is not trusted


def fit_readout(coded, constant, code, chip, t_exp, axis=0, black=0.0, tau=0.0):
    """(readout_s, t0, rms, spread) for a rolling shutter: first to last line, and when the code starts after the
    first line opens. From a still scene shot once under the code and once under constant light (same
    exposure, gain, code length). Their ratio cancels the scene, leaving per line the share of light its
    window caught; line l opens at l readout / lines. Only measurable when lines catch the code differently
    (exposure shorter than code + readout + t0); otherwise per-row and per-frame are the same anyway.
    spread: std / mean of the measured ratio over lines, about 0 when it is not measurable"""
    n = coded.shape[axis]
    keep = constant > 0.1 * np.percentile(constant, 99)
    num = np.where(keep, coded, 0).sum(1 - axis)
    den = np.where(keep, constant, 0).sum(1 - axis)
    ok = den > 0.2 * den.max()
    lines, ratio = np.arange(n)[ok], num[ok] / den[ok]
    dt = chip / 32
    t = np.arange(-0.3, t_exp + 0.6, dt)
    cum = lambda L: np.r_[0, np.cumsum(F.lcd(L, dt, tau) if tau else L)] * dt
    edges = np.r_[t - dt / 2, t[-1] + dt / 2]
    Cc, Cb = cum(F.light(t, code, chip, black)), cum(F.light(t, "1" * len(code), chip, black))
    share = lambda C, a, t0: np.interp(a + t_exp - t0, edges, C) - np.interp(a - t0, edges, C)

    def err(p, all_=False):
        a = lines * p[0] / n
        pred = share(Cc, a, p[1]) / np.maximum(share(Cb, a, p[1]), 1e-12)
        k = pred @ ratio / (pred @ pred)                   # one free gain
        return np.sqrt(np.mean((k * pred - ratio) ** 2))

    grid = [(ro, t0) for ro in np.arange(0, READOUT_MAX, 0.001) for t0 in np.arange(-chip, T0_MAX, chip / 10)]
    best = min(grid, key=err)
    res = minimize(err, best, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-12))
    return float(res.x[0]), float(res.x[1]), float(res.fun), float(ratio.std() / ratio.mean())


def align(ref, x, hub, r, th):
    """(reference turned to where the blade is in x, angle turned): a still of the stopped blade, rotated
    about the hub by the angle that best matches x over the rings r and angles th (circular
    cross-correlation of every ring, summed). Puts the ground truth where the deblurred blade is.
    Matched on the change along each ring, scaled to unit size per ring: edges and print, so uneven
    light cannot pull it onto a brighter look-alike blade"""
    dth = th[1] - th[0]
    full = th[0] + np.arange(int(np.ceil(2 * np.pi / dth))) * dth
    Px, Pr = F.polar(x, hub, r, th)[0], F.polar(ref, hub, r, full)[0]
    Px = np.diff(Px, axis=1, append=Px[:, -1:])
    Pr = np.diff(Pr, axis=1, append=Pr[:, :1])                   # the reference's ring is a full turn
    Px = Px / (np.linalg.norm(Px, axis=1, keepdims=True) + 1e-12)
    Pr = Pr / (np.linalg.norm(Pr, axis=1, keepdims=True) + 1e-12)
    pad = np.zeros_like(Pr)
    pad[:, :Px.shape[1]] = Px
    corr = np.fft.irfft(np.conj(np.fft.rfft(pad)) * np.fft.rfft(Pr), Pr.shape[1]).sum(0)
    s = int(np.argmax(corr))
    a, b, c = corr[s - 1], corr[s], corr[(s + 1) % len(corr)]
    d = a - 2 * b + c
    shift = (s + (0.5 * (a - c) / d if d < 0 else 0.0)) * dth
    turned = F.cartesian(F.polar(ref, hub, r, th + shift)[0], hub, r, th, x.shape)
    return turned, (shift + np.pi) % (2 * np.pi) - np.pi


def turned_at(img, centre, angle, rows, cols):
    """img turned counter-clockwise (as displayed) by angle radians about centre (row, col), read at the
    pixels rows, cols: the value each pixel gets from where it came from"""
    y, x = centre[0] - rows, cols - centre[1]                      # up, right
    c, s = np.cos(angle), np.sin(angle)
    return map_coordinates(img, [centre[0] - (c * y - s * x), centre[1] + (c * x + s * y)], order=1, mode="nearest")


def moved(f0, f1):
    """pixels that changed between two stills (the blade at either angle), 2 px margin. The static
    background is left out, so it cannot pull a fit of the turn"""
    d = np.abs(f1 - f0)
    return binary_dilation(d > 0.25 * np.percentile(d, 99.9), iterations=2)


def hub_from_stills(f0, f1, angle):
    """(hub (row, col), turn sign, rms): the centre about which still f0, turned by |angle| radians,
    best matches still f1, with no guess. sign +1 = the fan turned counter-clockwise as displayed.
    A grid over the frame and half a frame around it (the hub may be out of view) on images shrunk to
    about 200 px, at 400 of the pixels that changed, then Nelder-Mead at full size on up to 50,000.
    Only pixels that changed count"""
    m = moved(f0, f1)
    if m.sum() < 50:
        raise ValueError("the two stills are the same: turn the fan between them")
    k = max(1, round(max(f0.shape) / 200))
    g0, g1 = downscale_local_mean(f0, (k, k)), downscale_local_mean(f1, (k, k))
    gm = downscale_local_mean(m.astype(float), (k, k)) > 0
    rows, cols = np.nonzero(gm)
    pick = np.random.default_rng(0).permutation(len(rows))[:400]       # plenty to find the hub roughly
    rows, cols = rows[pick], cols[pick]
    cr, cc = np.meshgrid(np.arange(-0.5 * g0.shape[0], 1.5 * g0.shape[0], 2.0),
                         np.arange(-0.5 * g0.shape[1], 1.5 * g0.shape[1], 2.0), indexing="ij")
    cr, cc = cr.ravel()[:, None], cc.ravel()[:, None]
    n = max(1, 4_000_000 // len(rows))                            # centres per batch
    best = (np.inf,)
    for sign in (1, -1):
        for i in range(0, len(cr), n):
            e = np.mean((g1[rows, cols] - turned_at(g0, (cr[i:i + n], cc[i:i + n]), sign * angle, rows, cols)) ** 2, 1)
            j = int(np.argmin(e))
            if e[j] < best[0]:
                best = (e[j], sign, (cr[i + j, 0] + 0.5) * k - 0.5, (cc[i + j, 0] + 0.5) * k - 0.5)   # full-size px
    _, sign, cr, cc = best
    rows, cols = np.nonzero(m)
    pick = np.random.default_rng(0).permutation(len(rows))[:50_000]
    rows, cols = rows[pick], cols[pick]
    cost = lambda c: np.mean((f1[rows, cols] - turned_at(f0, c, sign * angle, rows, cols)) ** 2)
    res = minimize(cost, [cr, cc], method="Nelder-Mead",
                   options=dict(xatol=0.02, fatol=1e-14, initial_simplex=[[cr, cc], [cr + 2 * k, cc], [cr, cc + 2 * k]]))
    return tuple(res.x), sign, float(np.sqrt(res.fun))


def radii_from_stills(f0, f1, hub):
    """(r0, r1) px: the rings the blade covers, from the pixels that changed between the stills, 3 px margin"""
    rows, cols = np.nonzero(moved(f0, f1))
    rr = np.hypot(rows - hub[0], cols - hub[1])
    return max(2.0, float(np.floor(np.percentile(rr, 1))) - 3), float(np.ceil(np.percentile(rr, 99.5))) + 3


def find_arc(img, hub, r):
    """(a0, a1) deg, counter-clockwise from +col: the arc to deblur in one capture, the brightest blade
    and its smear. The capture's brightness around the hub over the outer half of rings r (nearer the
    hub the blades run together, more so if the hub is a little off): blades and their smears are
    bright, the gaps between them dark. Widened by a tenth (at least 3 deg) each side for the faint ends
    of the smear, where only one bit of the code lit the blade. Refuses arcs over 180 deg: the deblur
    needs the arc plus its smear to stay short of a full turn"""
    th = np.arange(-np.pi, np.pi, 1 / r[-1])
    P, rows, cols = F.polar(img, hub, r[r >= (r[0] + r[-1]) / 2], th)
    inside = (rows >= 0) & (rows <= img.shape[0] - 1) & (cols >= 0) & (cols <= img.shape[1] - 1)
    prof = (P * inside).sum(0) / np.maximum(inside.sum(0), 1)
    prof = uniform_filter1d(prof, max(1, int(np.radians(2) * r[-1])), mode="wrap")
    bg, top = np.percentile(prof, 10), np.percentile(prof, 99)
    on = prof > bg + 0.05 * (top - bg)
    if on.all() or not on.any():
        raise ValueError("no dark gap between blades: set the arc by hand (calibrate.py init --arc)")
    shift = int(np.argmin(on))                                     # start the circle in a gap
    on, prof = np.roll(on, -shift), np.roll(prof, -shift) - bg
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]
    i = int(np.argmax([prof[a:b].sum() for a, b in zip(starts, ends)]))
    dth = th[1] - th[0]
    a0, a1 = th[0] + (starts[i] + shift) * dth, th[0] + (ends[i] + shift) * dth
    pad = max(np.radians(3), 0.1 * (a1 - a0))
    a0, a1 = a0 - pad, a1 + pad
    if a1 - a0 > np.pi:
        raise ValueError(f"the brightest blade and its smear span {np.degrees(a1 - a0):.0f} deg, over 180: the "
                         "blades run together (hub off, or too fast for the gaps), set the arc by hand")
    return float(np.degrees(a0)), float(np.degrees(a1))
