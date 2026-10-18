"""Coded illumination: exposure synthesis, sensor noise, polar unwrap, deblur, speed estimate.

Time model. Sub-frame k is the fan at t_k = (k + 0.5) dt and stands for [t_k - dt/2, t_k + dt/2).
A sensor row that opens at s and exposes for t_exp collects
    w_k = L(t_k) * overlap(sub-frame k, [s, s + t_exp))       (light x seconds)
and its pixel value is sum_k w_k frame_k. Global shutter: every row opens at t_start.
Rolling shutter: row y opens at t_start + y t_line.

Deblur model. Unwrapped about the hub, the fan turns by omega t along theta, so polar sample i sees
    b_i = sum_k w_k(row_i) x(theta_i - omega t_k)
which is one ring matrix A per radius. Global shutter: the same w for every sample (circulant).
Rolling shutter per-row: w from the sample's own sensor row. Per-frame: one row's w for all samples.
"""
import numpy as np
from scipy.ndimage import map_coordinates

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
    """(rows, frames) light x seconds each row collects from each sub-frame. starts: row open times"""
    s = np.asarray(starts, float)[:, None]
    ov = np.minimum(t + dt / 2, s + t_exp) - np.maximum(t - dt / 2, s)
    return L * np.clip(ov, 0, None)


def expose(frames, w):
    """image from sub-frames. w is (1, K) for global shutter or (H, K) for one row each"""
    if len(w) == 1:
        return np.tensordot(w[0], frames, 1)
    return np.einsum("yk,kyx->yx", w, frames)


def sensor(img, flux, rng, full_e=10000, read_e=3.0, bits=10, gain=1.0):
    """shot + read noise, full well clip, ADC. img in light x seconds, flux electrons per unit of it.
    gain is analogue: it shrinks the ADC step but not the read noise. Returns img units again"""
    e = np.minimum(rng.poisson(np.clip(img * flux, 0, None)), full_e) + rng.normal(0, read_e, img.shape)
    step = full_e / (2 ** bits - 1) / gain
    dn = np.clip(np.round(e / step), 0, 2 ** bits - 1)
    return dn * step / flux


def polar(img, centre, r, n):
    """samples at radii r and n angles (counter-clockwise from +x on screen), and the row of each"""
    th = 2 * np.pi * np.arange(n) / n
    rows = centre[0] - r[:, None] * np.sin(th)
    cols = centre[1] + r[:, None] * np.cos(th)
    return map_coordinates(img, [rows, cols], order=1), rows


def cartesian(p, centre, r, shape):
    """inverse of polar, zero outside the sampled radii"""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    rr = np.hypot(yy - centre[0], xx - centre[1])
    th = np.arctan2(centre[0] - yy, xx - centre[1]) % (2 * np.pi)
    ri = (rr - r[0]) / (r[1] - r[0])
    ti = th / (2 * np.pi) * p.shape[1]
    out = map_coordinates(p, [ri, ti], order=1, mode="grid-wrap")
    out[(rr < r[0]) | (rr > r[-1])] = 0
    return out


def ring_matrix(w, t, omega, n):
    """A[i, j]: weight of x_j in b_i. w is (n, K), one row of weights per sample, or (1, K)"""
    w = np.broadcast_to(w, (n, len(t)))
    pos = np.arange(n)[:, None] - omega * t * n / (2 * np.pi)   # where x was at each sub-frame
    j = np.floor(pos).astype(int)
    f = pos - j
    i = np.repeat(np.arange(n)[:, None], len(t), 1)
    A = np.bincount((i * n + j % n).ravel(), (w * (1 - f)).ravel(), n * n)
    A += np.bincount((i * n + (j + 1) % n).ravel(), (w * f).ravel(), n * n)
    return A.reshape(n, n)


def deblur_ring(A, b, lam):
    """argmin |Ax - b|^2 + lam |Dx|^2, D the circular first difference along theta"""
    n = len(b)
    D = np.eye(n) - np.roll(np.eye(n), 1, 1)
    return np.linalg.solve(A.T @ A + lam * D.T @ D, A.T @ b)


def deblur(P, rows, t, dt, L, t_exp, omega, lam, t_start=0.0, t_line=0.0, per_row=True, ref_row=None):
    """deblur every ring of polar image P. per_row=False uses ref_row's timing for every sample"""
    out = np.empty_like(P)
    n = P.shape[1]
    for k in range(len(P)):
        if t_line and per_row:
            w = weights(t, dt, L, t_exp, t_start + rows[k] * t_line)
        else:
            w = weights(t, dt, L, t_exp, [t_start + (ref_row or 0) * t_line])
        out[k] = deblur_ring(ring_matrix(w, t, omega, n), P[k], lam)
    return out


def speed(P, rows, t, dt, L, t_exp, omegas, lam, **kw):
    """blind speed: the omega whose deblur leaves the least negative light. returns best, scores"""
    scores = []
    for om in omegas:
        x = deblur(P, rows, t, dt, L, t_exp, om, lam, **kw)
        scores.append(np.clip(-x, 0, None).sum() / np.abs(x).sum())
    return omegas[int(np.argmin(scores))], np.array(scores)
