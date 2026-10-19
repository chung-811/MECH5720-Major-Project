"""Coded illumination: exposure synthesis, sensor noise, polar unwrap, deblur, speed estimate.

Time model. Sub-frame k is the fan at t_k = (k + 0.5) dt and stands for [t_k - dt/2, t_k + dt/2).
A sensor line that opens at s and exposes for t_exp collects
    w_k = L(t_k) * overlap(sub-frame k, [s, s + t_exp))       (light x seconds)
Global shutter: every line opens at t_start. Rolling shutter: line l opens at t_start + l t_line.
Lines are image rows (axis 0) or columns (axis 1, camera on its side).

Deblur model. Unwrapped about the hub, the blade turns by omega t along theta, so polar sample i sees
    b_i = sum_k w_k(line_i) x(theta_i - omega t_k)
one matrix A per radius over the arc in view. The arc does not wrap, so x carries extra samples
behind the window for the light that moved into it. Global shutter: the same w for every sample.
Rolling shutter per-row: w from the sample's own sensor line. Per-frame: one line's w for all.
"""
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

CHIP = 1 / 60                                               # one monitor refresh
RASKAR52 = "1010000111000001010000110011110111010111001001100111"   # Raskar 2006, sec 3.2


def light(t, code, chip=CHIP, black=0.0, t0=0.0):
    """monitor brightness at times t: 1 for an on chip, `black` for off chips and outside the code"""
    j = np.floor((np.asarray(t) - t0) / chip).astype(int)
    c = np.array([int(ch) for ch in code])
    on = np.zeros(j.shape)
    inside = (j >= 0) & (j < len(c))
    on[inside] = c[j[inside]]
    return black + (1 - black) * on


def lcd(L, dt, tau):
    """first-order lag of time constant tau on uniformly sampled brightness, a crude LCD response"""
    out, a = np.empty_like(L), 1 - np.exp(-dt / tau)
    out[0] = L[0]
    for k in range(1, len(L)):
        out[k] = out[k - 1] + a * (L[k] - out[k - 1])
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


def polar(img, centre, r, th):
    """samples at radii r and angles th (counter-clockwise from +x on screen), with their rows and cols"""
    rows = centre[0] - r[:, None] * np.sin(th)
    cols = centre[1] + r[:, None] * np.cos(th)
    return map_coordinates(img, [rows, cols], order=1), rows, cols


def cartesian(p, centre, r, th, shape):
    """inverse of polar, zero outside the sampled radii and angles"""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    rr = np.hypot(yy - centre[0], xx - centre[1])
    ang = np.arctan2(centre[0] - yy, xx - centre[1])
    out = map_coordinates(p, [(rr - r[0]) / (r[1] - r[0]), (ang - th[0]) / (th[1] - th[0])], order=1, mode="nearest")
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


def deblur_ring(A, m, b, lam):
    """argmin |Ax - b|^2 + lam |Dx|^2 with D the first difference, returned over the window"""
    N = A.shape[1]
    D = np.diff(np.eye(N), axis=0)
    return np.linalg.solve(A.T @ A + lam * D.T @ D, A.T @ b)[m:m + len(b)]


def deblur(P, lines, t, dt, L, t_exp, omega, dth, lam, t_start=0.0, t_line=0.0, per_row=True, ref_line=0):
    """deblur every ring of polar image P. lines: sensor line of each sample.
    per_row=False uses ref_line's timing for every sample"""
    out = np.empty_like(P)
    for k in range(len(P)):
        starts = t_start + (lines[k] if t_line and per_row else np.array([ref_line])) * t_line
        out[k] = deblur_ring(*ring_matrix(weights(t, dt, L, t_exp, starts), t, omega, dth, P.shape[1]), P[k], lam)
    return out


def speed(P, lines, t, dt, L, t_exp, omegas, dth, lam, **kw):
    """blind speed: the omega whose deblur leaves the least negative light. returns best, scores"""
    scores = []
    for om in omegas:
        x = deblur(P, lines, t, dt, L, t_exp, om, dth, lam, **kw)
        scores.append(np.clip(-x, 0, None).sum() / np.abs(x).sum())
    return omegas[int(np.argmin(scores))], np.array(scores)


def load(path, axis=1, pad=0.02):
    """renders plus the unwrap grid (radii 1 px apart, arc 1 px apart at the outer edge, from the blade
    at t = 0 to past where it ends up), the ground truth through the same interpolation, and the mask.
    axis: readout direction of the rolling shutter, 1 = columns (camera on its side)"""
    d = np.load(path)
    s = SimpleNamespace(fr=d["frames"], t=d["t"], dt=float(d["dt"]), om=float(d["omega"]), ref=d["ref"], axis=axis)
    s.H, s.W = s.ref.shape
    s.c = tuple(d["hub"])
    s.r = np.arange(float(d["r0_px"]) - 5, float(d["r1_px"]) + 5)
    s.dth = 1 / s.r[-1]
    ha = float(d["half_angle"])
    s.th = np.arange(-np.pi / 2 - ha - pad, -np.pi / 2 + ha + s.om * (s.t[-1] + s.dt) + pad, s.dth)
    s.lines = s.W if axis else s.H
    s.truth = cartesian(polar(s.ref, s.c, s.r, s.th)[0], s.c, s.r, s.th, s.ref.shape)
    window = cartesian(np.ones((len(s.r), len(s.th))), s.c, s.r, s.th, s.ref.shape) > 0.5
    s.mask = window & binary_dilation(s.truth > 0.05, iterations=5)      # the blade, 5 px margin
    return s


def unwrap(s, img):
    """polar image and the sensor line of each sample"""
    P, rows, cols = polar(img, s.c, s.r, s.th)
    return P, (cols if s.axis else rows)


def wrap(s, P):
    return cartesian(P, s.c, s.r, s.th, s.ref.shape)


def capture(s, L, t_exp, flux, rng, t_line=0.0, gain=1.0, **noise):
    """noisy image of the renders under brightness L. t_line 0 = global shutter"""
    starts = np.arange(s.lines) * t_line if t_line else [0.0]
    return sensor(expose(s.fr, weights(s.t, s.dt, L, t_exp, starts), s.axis), flux, rng, gain=gain, **noise)


def bands(s, x, n=8):
    """PSNR over the mask in n bands of sensor lines, first line to last (nan where a band misses it)"""
    e = np.linspace(0, s.lines, n + 1).astype(int)
    T, X, M = (s.truth, x, s.mask) if s.axis == 0 else (s.truth.T, x.T, s.mask.T)
    return np.array([psnr(T[a:b][M[a:b]], X[a:b][M[a:b]], data_range=1) if M[a:b].any() else np.nan
                     for a, b in zip(e[:-1], e[1:])])


def score(s, x):
    """(PSNR, SSIM) against the ground truth over the blade"""
    return psnr(s.truth[s.mask], x[s.mask], data_range=1), ssim(s.truth, x * s.mask, data_range=1)
