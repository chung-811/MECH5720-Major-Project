"""Coded illumination: exposure synthesis, sensor noise, polar unwrap, deblur, speed estimate.

Time model. Sub-frame k is the fan at t_k = (k + 0.5) dt and stands for [t_k - dt/2, t_k + dt/2).
A sensor line that opens at s and exposes for t_exp collects
    w_k = L(t_k) * overlap(sub-frame k, [s, s + t_exp))       (light x seconds)
Global shutter: every line opens at 0. Rolling shutter: line l opens at l t_line.
Lines are image rows (axis 0) or columns (axis 1, camera on its side).

Deblur model. Unwrapped about the hub, the blade turns by omega t along theta, so polar sample i sees
    b_i = sum_k w_k(line_i) x(theta_i - omega t_k)
one matrix A per radius over the arc in view. The arc does not wrap, so x carries extra samples
behind the window for the light that moved into it. Global shutter: the same w for every sample.
Rolling shutter per-row: w from the sample's own sensor line. Per-frame: one line's w for all.
Polar samples that fall outside the image carry no data: unwrap marks them with a NaN sensor line, and
every solver here gives them a zero row in A, so they constrain nothing (they used to be fitted as black).
"""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_dilation, map_coordinates
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

CHIP = 1 / 60                                               # one monitor refresh
INTERP = 3          # spline order of the polar unwrap and wrap back. Bilinear (1) twice capped a still blade's
                    # round trip at 38-40 dB in simulation; cubic lifts it to 49-52 dB and the honest deblur PSNR
                    # (against the untouched render) by 1.1 dB (photo on the blade) to 4.6 dB (letters).
                    # The speed search and direction check stay on bilinear data (SPEED_INTERP): on cubic data the
                    # search fell to the edge of its range with the true speed 18% above --rpm, inside it
SPEED_INTERP = 1
RASKAR52 = "1010000111000001010000110011110111010111001001100111"   # Raskar 2006, sec 3.2


def light(t, code, chip=CHIP, black=0.0, t0=0.0):
    """monitor brightness at times t: 1 for an on chip, `black` for off chips and outside the code"""
    j = np.floor((np.asarray(t) - t0) / chip).astype(int)
    c = np.array([int(ch) for ch in code])
    on = np.zeros(j.shape)
    inside = (j >= 0) & (j < len(c))
    on[inside] = c[j[inside]]
    return black + (1 - black) * on


def lcd(L, dt, rise, fall=None, start=None):
    """first-order lag on uniformly sampled brightness, a crude LCD response: time constant `rise` while it
    brightens and `fall` (default: the same) while it dims, 0 = instant. start: what the screen showed
    before the first sample (default L[0], no lag on it). Our monitor: real.MONITOR"""
    fall = rise if fall is None else fall
    up = 1 - np.exp(-dt / rise) if rise > 0 else 1.0
    down = 1 - np.exp(-dt / fall) if fall > 0 else 1.0
    out = np.empty(len(L))
    prev = L[0] if start is None else start
    for k, target in enumerate(L):
        prev += (up if target > prev else down) * (target - prev)
        out[k] = prev
    return out


def weights(t, dt, L, t_exp, starts):
    """(lines, frames) light x seconds each line collects from each sub-frame. starts: line open times"""
    s = np.asarray(starts, float)[:, None]
    ov = np.minimum(t + dt / 2, s + t_exp) - np.maximum(t - dt / 2, s)
    return L * np.clip(ov, 0, None)


def expose(frames, w, axis=0):
    """image from sub-frames. w is (1, K) for global shutter or (lines, K) for one line each"""
    if len(w) == 1:
        return np.tensordot(w[0], frames, 1)
    return np.einsum("yk,kyx->yx" if axis == 0 else "xk,kyx->yx", w, frames)


def sensor(img, flux, rng, full_e=10000, read_e=3.0, bits=10, gain=1.0):
    """shot + read noise, full well clip, ADC. img in light x seconds, flux electrons per unit of it.
    gain is analogue: it shrinks the ADC step but not the read noise. Returns img units again"""
    e = np.minimum(rng.poisson(np.clip(img * flux, 0, None)), full_e) + rng.normal(0, read_e, img.shape)
    step = full_e / (2 ** bits - 1) / gain
    dn = np.clip(np.round(e / step), 0, 2 ** bits - 1)
    return dn * step / flux


def polar(img, centre, r, th, order=1):
    """samples at radii r and angles th (counter-clockwise from +x on screen), with their rows and cols.
    Outside the image a sample reads 0. order: spline order (1 bilinear, 3 cubic)"""
    rows = centre[0] - r[:, None] * np.sin(th)
    cols = centre[1] + r[:, None] * np.cos(th)
    return map_coordinates(img, [rows, cols], order=order), rows, cols


def cartesian(p, centre, r, th, shape, order=1):
    """inverse of polar, zero outside the sampled radii and angles. th[0] in (-pi, pi], see grid"""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    rr = np.hypot(yy - centre[0], xx - centre[1])
    ang = np.arctan2(centre[0] - yy, xx - centre[1])
    ang = np.where(ang < th[0], ang + 2 * np.pi, ang)          # measured on from th[0], for arcs past +-pi
    out = map_coordinates(p, [(rr - r[0]) / (r[1] - r[0]), (ang - th[0]) / (th[1] - th[0])], order=order, mode="nearest")
    out[(rr < r[0]) | (rr > r[-1]) | (ang < th[0]) | (ang > th[-1])] = 0
    return out


def ring_matrix(w, t, omega, dth, n):
    """(A, m): A[i, j] is the weight of x_j in b_i, x starting m samples before the window and
    running past it, to hold the light that moved in from either side. w is (n, K) or (1, K)"""
    w = np.broadcast_to(w, (n, len(t)))
    shift = omega * t / dth                                  # samples moved by each sub-frame
    m = int(np.ceil(max(shift.max(), 0))) + 1
    N = n + m + int(np.ceil(max(-shift.min(), 0))) + 1
    pos = np.arange(n)[:, None] - shift + m
    j = np.floor(pos).astype(int)
    f = pos - j
    i = np.repeat(np.arange(n)[:, None], len(t), 1)
    A = np.bincount((i * N + j).ravel(), (w * (1 - f)).ravel(), n * N)
    A += np.bincount((i * N + j + 1).ravel(), (w * f).ravel(), n * N)
    return A.reshape(n, N), m


def tikhonov_ring(A, m, b, lam):
    """argmin |Ax - b|^2 + lam |Dx|^2 with D the first difference, returned over the window"""
    N = A.shape[1]
    D = np.diff(np.eye(N), axis=0)
    return np.linalg.solve(A.T @ A + lam * D.T @ D, A.T @ b)[m:m + len(b)]


def _starts(lines_k, t_line, per_row, ref_line):
    """open time of the sensor line of each sample in one ring (one value when they all share it)"""
    return (lines_k if t_line and per_row else np.array([ref_line])) * t_line


def _ring(scene, L, t_exp, lines_k, omega, n, t_line, per_row, ref_line):
    """(A, m) of one ring of n samples, with a zero row for each sample outside the image (NaN line)"""
    inside = np.isfinite(lines_k)
    w = weights(scene.t, scene.dt, L, t_exp, _starts(np.where(inside, lines_k, 0.0), t_line, per_row, ref_line))
    A, m = ring_matrix(w, scene.t, omega, scene.dtheta, n)
    A[~inside] = 0.0
    return A, m


def tikhonov(scene, P, lines, L, t_exp, lam, t_line=0.0, per_row=True, ref_line=0, omega=None):
    """fast baseline deblur, ring by ring. P: polar image, lines: sensor line of each sample (NaN outside
    the image), L: monitor brightness at scene.t. per_row=False uses ref_line's timing for every sample.
    A ring entirely outside the image has no data and comes back 0"""
    om = scene.omega if omega is None else omega
    out = np.zeros_like(P)
    for k in range(len(P)):
        if np.isfinite(lines[k]).any():
            A, m = _ring(scene, L, t_exp, lines[k], om, P.shape[1], t_line, per_row, ref_line)
            out[k] = tikhonov_ring(A, m, np.where(np.isfinite(lines[k]), P[k], 0.0), lam)
    return out


def forward_model(scene, L, t_exp, lines, t_line=0.0, per_row=True, ref_line=0, omega=None):
    """(A, m): sparse forward model of every ring in `lines`, polar image (R, N) -> polar data (R, n).
    A sample outside the image (NaN line) gets a zero row. omega defaults to scene.omega"""
    om = scene.omega if omega is None else omega
    blocks = []
    for k in range(len(lines)):
        Ak, m = _ring(scene, L, t_exp, lines[k], om, len(scene.th), t_line, per_row, ref_line)
        blocks.append(sp.csr_matrix(Ak))
    return sp.block_diag(blocks, format="csr"), m


@lru_cache(1)
def _drunet():
    import deepinv as dinv
    return dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained="download", device="cpu")


NOISE_GAIN = 6.0     # DPIR noise level / estimate_sigma on the polar data, see dpir
BLADE_NOISE_GAIN = 1.5   # DPIR noise level / blade_noise on the raw capture. Over five simulated scenes (letters,
                         # photo, photo at a tenth of the light, the test blade, rolling shutter) it is never more
                         # than 1.0 dB below each scene's best gain; 2.0 loses up to 2.2 dB, and 1.0 collapses


def blade_noise(img, window):
    """noise std of the blade in a raw capture: median absolute finest diagonal detail of 2x2 pixel blocks
    (Donoho's estimator, as skimage's estimate_sigma) over blocks inside `window` (bool, image shape) that are
    bright. Shot noise grows with brightness, so a whole-image estimate is pulled down by the dark background
    (one sixth of the blade's noise in simulation); this read 0.91-1.03 of the blade's true noise.
    None when fewer than 100 bright blocks"""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    a, b, c, d = img[0:h:2, 0:w:2], img[0:h:2, 1:w:2], img[1:h:2, 0:w:2], img[1:h:2, 1:w:2]
    detail, mean = (a - b - c + d) / 2, (a + b + c + d) / 4
    inside = window[0:h:2, 0:w:2] & window[1:h:2, 1:w:2]
    if not inside.any():
        return None
    bright = inside & (mean > 0.5 * np.percentile(mean[inside], 99))
    return float(np.median(np.abs(detail[bright])) / 0.6745) if bright.sum() >= 100 else None


def dpir(A, m, P, sigma=None, full=False):
    """plug-and-play deblur: DPIR (HQS with a pretrained DRUNet denoiser as the prior, Zhang et al.,
    TPAMI 2022) through deepinv, on our forward model. The model is first scaled to unit gain (a still
    pixel keeps its value), which DPIR's noise-level schedule assumes; our simulations have gain 0.43
    (26 on-bits x 1/60 s), real captures 1. sigma: the noise level DPIR uses, in units of P. When None:
    estimate_sigma on the polar data reads 0.31-0.35 of the true noise (resampling smooths it; two scenes,
    light 1x to 0.03x, both shutters), and DPIR did best at 2x the true noise in all of those and in OCR,
    so NOISE_GAIN x the estimate; real captures pass sigma from blade_noise instead. Rows of A that are all
    zero (samples outside the image) carry no data and are left out of the gain and the brightness scale.
    The data is then scaled so its bright end (99th percentile) is 1, the range DRUNet was trained on: end to
    end through capture.deblur, the same capture at 0.15x brightness lost 2.4 dB without this, and none with it.
    full: return the whole polar estimate (R, N) instead of the window"""
    import torch
    import deepinv as dinv
    R, n = P.shape
    N = A.shape[1] // R
    rowsum = np.asarray(A.sum(1)).ravel()
    data = rowsum > 0                                        # samples inside the image
    if not data.any():
        raise ValueError("no polar sample falls inside the image: check the hub and radii")
    g = float(rowsum[data].mean())                           # gain: mean row sum of the forward model
    A, P = A / g, P / g
    sigma = NOISE_GAIN * float(estimate_sigma(P)) if sigma is None else sigma / g
    scale = float(np.percentile(P.ravel()[data], 99))
    scale = scale if scale > 0 else 1.0
    P, sigma = P / scale, sigma / scale
    fwd = lambda X: torch.from_numpy(A @ X.numpy().ravel()).float().reshape(1, 1, R, n)
    adj = lambda Y: torch.from_numpy(A.T @ Y.numpy().ravel()).float().reshape(1, 1, R, N)
    with torch.no_grad():
        X = dinv.optim.DPIR(sigma=sigma, denoiser=_drunet())(torch.from_numpy(P).float()[None, None],
                                                             dinv.physics.LinearPhysics(A=fwd, A_adjoint=adj))
    X = X.numpy()[0, 0] * scale
    return X if full else X[:, m:m + n]


def residuals(scene, P, lines, L, t_exp, omegas, rings=64, sigma=None, **kw):
    """|A X - P| of the DPIR deblur at each speed, on a band of rings mid-blade: how well each speed
    explains the data. A wrong speed (or direction) predicts the blur wrongly and the prior will not
    let DPIR fit it anyway. Samples outside the image are left out. sigma: as in dpir"""
    band = slice(max(0, len(P) // 2 - rings // 2), len(P) // 2 + rings // 2)
    Pb = np.where(np.isfinite(lines[band]), P[band], 0.0)
    out = []
    for om in omegas:
        A, m = forward_model(scene, L, t_exp, lines[band], omega=om, **kw)
        X = dpir(A, m, Pb, sigma=sigma, full=True)
        out.append(np.linalg.norm(A @ X.ravel() - Pb.ravel()))
    return np.array(out)


def speed(scene, P, lines, L, t_exp, omegas, rings=64, **kw):
    """blind speed: the omega whose DPIR deblur best explains the data (residuals), refined by a
    parabola through the best grid point and its neighbours. In simulation (3 speeds x 3 noise draws)
    this was within 0.6%, where negative light after Tikhonov read 2.5% low. returns estimate, scores.
    A best point at either end of omegas means the speed is outside them: the estimate is then wrong"""
    scores = residuals(scene, P, lines, L, t_exp, omegas, rings, **kw)
    i = int(np.clip(np.argmin(scores), 1, len(scores) - 2))
    a, b, c = scores[i - 1:i + 2]
    d = a - 2 * b + c
    return omegas[i] + (0.5 * (a - c) / d if d > 0 else 0.0) * (omegas[1] - omegas[0]), scores


def grid(hub, r, th0, th1, dtheta, shape, axis, t, dt, omega):
    """the unwrap grid and time grid every function here works on: rings r (px) about hub (row, col),
    angles th0..th1 (rad, counter-clockwise from +col) dtheta apart, sub-frames t of length dt, blade
    speed omega (rad/s, + = counter-clockwise in the image). axis: sensor lines are rows (0) or cols (1)"""
    while th0 > np.pi:                     # any arc start, e.g. 200 deg, as the same arc from (-pi, pi]
        th0, th1 = th0 - 2 * np.pi, th1 - 2 * np.pi
    while th0 <= -np.pi:
        th0, th1 = th0 + 2 * np.pi, th1 + 2 * np.pi
    return SimpleNamespace(hub=tuple(hub), r=np.asarray(r, float), th=np.arange(th0, th1, dtheta), dtheta=dtheta,
                           shape=tuple(shape), axis=axis, n_lines=shape[axis], t=np.asarray(t), dt=float(dt),
                           omega=float(omega))


def load(path, axis=1, pad=0.02):
    """a Blender render (blender/render_fan.py) as a scene: the grid (radii 1 px apart, arc 1 px apart at
    the outer edge, from the blade at t = 0 to past where it ends up), the frames, the ground truth through
    the same interpolation, and the mask. axis: readout direction, 1 = columns (camera on its side)"""
    d = np.load(path)
    ref, t, dt, om = d["ref"], d["t"], float(d["dt"]), float(d["omega"])
    r = np.arange(float(d["r0_px"]) - 5, float(d["r1_px"]) + 5)
    ha = float(d["half_angle"])
    scene = grid(d["hub"], r, -np.pi / 2 - ha - pad, -np.pi / 2 + ha + om * (t[-1] + dt) + pad, 1 / r[-1],
                 ref.shape, axis, t, dt, om)
    scene.frames, scene.ref = d["frames"], ref
    scene.truth = wrap(scene, unwrap(scene, ref)[0])
    window = wrap(scene, np.ones((len(scene.r), len(scene.th)))) > 0.5
    scene.mask = window & binary_dilation(scene.truth > 0.05, iterations=5)      # the blade, 5 px margin
    return scene


def unwrap(scene, img, order=None):
    """polar image and the sensor line of each sample; the line is NaN where the sample falls outside the
    image, so the solvers ignore it (see the module docstring). order: spline order, default INTERP"""
    P, rows, cols = polar(img, scene.hub, scene.r, scene.th, INTERP if order is None else order)
    inside = (rows >= 0) & (rows <= scene.shape[0] - 1) & (cols >= 0) & (cols <= scene.shape[1] - 1)
    return P, np.where(inside, cols if scene.axis else rows, np.nan)


def wrap(scene, P):
    return cartesian(P, scene.hub, scene.r, scene.th, scene.shape, INTERP)


def capture(scene, L, t_exp, flux, rng, t_line=0.0, gain=1.0, **noise):
    """noisy image of the render under brightness L. t_line 0 = global shutter"""
    starts = np.arange(scene.n_lines) * t_line if t_line else [0.0]
    return sensor(expose(scene.frames, weights(scene.t, scene.dt, L, t_exp, starts), scene.axis), flux, rng,
                  gain=gain, **noise)


def bands(scene, x, n=8):
    """PSNR over the mask in n bands of sensor lines, first line to last (nan where a band misses it)"""
    e = np.linspace(0, scene.n_lines, n + 1).astype(int)
    T, X, M = (scene.truth, x, scene.mask) if scene.axis == 0 else (scene.truth.T, x.T, scene.mask.T)
    return np.array([psnr(T[a:b][M[a:b]], X[a:b][M[a:b]], data_range=1) if M[a:b].any() else np.nan
                     for a, b in zip(e[:-1], e[1:])])


def score(scene, x):
    """(PSNR, SSIM) against the ground truth over the blade"""
    return psnr(scene.truth[scene.mask], x[scene.mask], data_range=1), ssim(scene.truth, x * scene.mask, data_range=1)
