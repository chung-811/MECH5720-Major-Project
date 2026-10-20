import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import capture
import flutter as F
import real

# a synthetic rotating blade, written down analytically, so no Blender or renders are needed.
# image 128 x 128, hub at the centre, blade 20..60 px, the 15-bit Agrawal code at 60 Hz
CODE, CHIP = "111111100110001", 1 / 60
T_EXP = len(CODE) * CHIP
HUB, SHAPE, R0, R1, HALF, START = (63.5, 63.5), (128, 128), 20, 60, 0.25, -np.pi / 2
OMEGA = 20 / (T_EXP * R1)                     # 20 px of blur at the outer edge


def blade(rows, cols, angle, texture=True, half=HALF):
    """the blade turned to `angle`: a sector with 1 px soft edges, stripes and rings printed on it"""
    rr = np.hypot(rows - HUB[0], cols - HUB[1])
    a = (np.arctan2(HUB[0] - rows, cols - HUB[1]) - angle + np.pi) % (2 * np.pi) - np.pi
    edge = lambda d: np.clip(d + 0.5, 0, 1)
    inside = edge(rr - R0) * edge(R1 - rr) * edge((half - np.abs(a)) * rr)
    ink = 0.6 + 0.4 * np.sin(rr / 3) * np.cos(8 * a) if texture else 1.0
    return 0.05 + 0.75 * inside * ink


def at_polar(r, th, angle, **kw):
    return blade(HUB[0] - r[:, None] * np.sin(th), HUB[1] + r[:, None] * np.cos(th), angle, **kw)


def grid(omega, t, dt, half=HALF):
    r = np.arange(R0 - 3.0, R1 + 4.0)
    end = START + half + max(omega, 0) * T_EXP + 0.1
    return F.grid(HUB, r, START - half + min(omega, 0) * T_EXP - 0.1, end, 1 / r[-1], SHAPE, 0, t, dt, omega)


def calib(readout=0.0):
    scene = grid(OMEGA, [0.0], 1.0)
    return dict(hub=list(HUB), radius_px=[scene.r[0], scene.r[-1]], arc_deg=list(np.degrees(scene.th[[0, -1]])),
                readout_s=readout, readout_axis=0, monitor=dict(black=0.0, tau=0.0, t0=0.0))


def simulate(readout=0.0, texture=True, omega=OMEGA, dt=CHIP / 40, half=HALF):
    """capture of the blade under the code, finer in time than capture.py's own grid. Global shutter
    when readout is 0, else line (row) l opens at l * readout / rows. Scaled like capture.brightness"""
    t = (np.arange(int(np.ceil((T_EXP + readout) / dt))) + 0.5) * dt
    L = F.light(t, CODE, CHIP)
    L = L / F.weights(t, dt, L, T_EXP, [0.0]).sum()
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    frames = np.array([blade(rows, cols, START + omega * tk, texture, half) for tk in t])
    starts = np.arange(SHAPE[0]) * readout / SHAPE[0] if readout else [0.0]
    return F.expose(frames, F.weights(t, dt, L, T_EXP, starts), 0), blade(rows, cols, START, texture, half)


def blade_psnr(x, truth):
    m = truth > 0.06
    return 10 * np.log10(1 / np.mean((x[m] - truth[m]) ** 2))


@pytest.mark.parametrize("omega", [OMEGA, -OMEGA])
@pytest.mark.parametrize("readout", [0.0, 0.05])
def test_forward_model_matches_direct_sum(omega, readout):
    """A x equals sum_k w_k x(theta - omega t_k), with each sample's own line timing for a rolling shutter"""
    dt = CHIP / 8
    t = (np.arange(int(np.ceil((T_EXP + readout) / dt))) + 0.5) * dt
    scene = grid(omega, t, dt)
    L = F.light(t, CODE, CHIP)
    lines = HUB[0] - scene.r[:, None] * np.sin(scene.th)                    # sensor line (row) of each sample
    t_line = readout / SHAPE[0]
    A, m = F.forward_model(scene, L, T_EXP, lines, t_line=t_line)
    N = A.shape[1] // len(scene.r)
    x = at_polar(scene.r, scene.th[0] + (np.arange(N) - m) * scene.dtheta, START)
    w = F.weights(t, dt, L, T_EXP, (lines * t_line).ravel()).reshape(*lines.shape, len(t))
    b = sum(w[..., k] * at_polar(scene.r, scene.th, START + omega * tk) for k, tk in enumerate(t))
    err = np.linalg.norm(A @ x.ravel() - b.ravel()) / np.linalg.norm(b)
    assert err < 0.01, err


def test_global_shutter_deblur():
    img, truth = simulate()
    x, _, _ = capture.deblur(img, calib(), CODE, CHIP, T_EXP, OMEGA * 60 / (2 * np.pi), "global", "tikhonov", 1e-3)
    assert blade_psnr(x, truth) > blade_psnr(img, truth) + 10


def test_rolling_shutter_per_row_beats_per_frame():
    """criterion 1 in miniature: per-row keeps up with global shutter, per-frame falls behind"""
    readout = 4 * CHIP
    img, truth = simulate(readout)
    rpm = OMEGA * 60 / (2 * np.pi)
    row = blade_psnr(capture.deblur(img, calib(readout), CODE, CHIP, T_EXP, rpm, "rs-row", "tikhonov", 1e-3)[0], truth)
    frame = blade_psnr(capture.deblur(img, calib(readout), CODE, CHIP, T_EXP, rpm, "rs-frame", "tikhonov", 1e-3)[0],
                       truth)
    gs_img, _ = simulate()
    gs = blade_psnr(capture.deblur(gs_img, calib(), CODE, CHIP, T_EXP, rpm, "global", "tikhonov", 1e-3)[0], truth)
    assert row > frame + 3 and row > gs - 1, (row, frame, gs)


def test_dpir_runs_and_deblurs():
    """the real deepinv import and DRUNet weights (downloaded once), on a noisy capture"""
    img, truth = simulate()
    img = img + np.random.default_rng(0).normal(0, 0.01, img.shape)
    x, _, _ = capture.deblur(img, calib(), CODE, CHIP, T_EXP, OMEGA * 60 / (2 * np.pi), "global", "dpir")
    assert blade_psnr(x, truth) > blade_psnr(img, truth) + 10


@pytest.mark.parametrize("arc_deg", [(-100, -40), (200, 260), (150, 210)])
def test_unwrap_and_wrap_any_arc(arc_deg):
    """an arc anywhere round the hub, including across +-180 deg, unwraps and wraps back"""
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    img = 0.5 + 0.3 * np.sin(rows / 9) * np.cos(cols / 7)
    r = np.arange(R0, R1 + 1.0)
    scene = F.grid(HUB, r, *np.radians(arc_deg), 1 / r[-1], SHAPE, 0, [0.0], 1.0, 0.0)
    back = F.wrap(scene, F.unwrap(scene, img)[0])
    inside = back > 0
    area = np.radians(arc_deg[1] - arc_deg[0]) / 2 * (R1 ** 2 - R0 ** 2)
    assert abs(inside.sum() / area - 1) < 0.05 and np.abs(back - img)[inside].max() < 0.05


@pytest.mark.parametrize("chip", [1 / 60, 0.0625])
def test_fit_monitor_recovers_black_lag_and_timing(chip):
    dt = chip / 16
    t = (np.arange(int(1.1 * len(CODE) * chip / dt)) + 0.5) * dt
    L = F.lcd(F.light(t, CODE, chip, black=0.04, t0=0.2 * chip), dt, 3e-3)
    black, tau, t0 = real.fit_monitor(L, t, dt, CODE, chip)[0]
    assert abs(black - 0.04) < 0.01 and abs(tau / 3e-3 - 1) < 0.1 and abs(t0 - 0.2 * chip) <= 0.05 * chip


@pytest.mark.parametrize("sign", [1, -1])
def test_kernel_from_edge_reads_the_code(sign):
    """the light from the leading edge of a plain blade, either way round: clockwise is the mirror image.
    The blade is wider than the smear, as on the rig, so its middle sees the whole code"""
    img, _ = simulate(texture=False, omega=OMEGA * 4, half=1.2)
    scene = grid(OMEGA * 4, [0.0], 1.0, half=1.2)
    hub, th = HUB, scene.th
    if sign < 0:
        img, hub, th = img[:, ::-1], (HUB[0], SHAPE[1] - 1 - HUB[1]), np.pi - scene.th[::-1]
    dt = CHIP / 16
    t = (np.arange(int(np.ceil(T_EXP / dt))) + 0.5) * dt
    L = real.kernel_from_edge(img, hub, scene.r[5:-5], th, sign * OMEGA * 4, t, window=5)
    assert np.corrcoef(L, F.light(t, CODE, CHIP))[0, 1] > 0.9


def test_run_log(tmp_path):
    """the json coded_illumination.py writes next to each DNG"""
    log = tmp_path / "coded.json"
    log.write_text(json.dumps(dict(sequence_played=[1, 0, 1, 1], actual_exposure_us=999852,
                                   params=dict(bit_duration_s=0.0625))))
    assert real.run_log(log) == ("1011", 0.0625, 0.999852)


DNG = Path(__file__).parents[2] / "illumination/data/coded/coded_20261019_220955.dng"


@pytest.mark.skipif(not DNG.exists(), reason="no rig capture in the repo")
def test_load_dng_on_a_rig_capture():
    img = real.load_dng(DNG)                  # OV5647 2592 x 1944 RGBG, first green sublattice
    assert img.shape == (972, 1296) and np.isfinite(img).all() and -0.05 < img.min() and img.max() <= 1.0


@pytest.mark.skipif(not DNG.exists(), reason="no rig capture in the repo")
def test_scripts_run_on_a_rig_capture(tmp_path):
    """calibrate.py and a deblur script end to end on a real DNG and its run log (a still scene, so the
    numbers mean nothing: this checks the plumbing). The arc is past 180 deg on purpose"""
    here = Path(__file__).parents[1]
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], cwd=here, check=True, capture_output=True)
    calib = tmp_path / "rs.json"
    run("calibrate.py", "init", calib, "--hub", 480, 640, "--radius", 150, 200, "--arc", 200, 230, "--readout", 0.0632)
    run("rolling-shutter-per-row.py", DNG, "--calib", calib, "--rpm", 2.8, "--solver", "tikhonov", "--out", tmp_path)
    x = np.load(tmp_path / f"{DNG.stem}_rs-row.npy")
    assert x.shape == (972, 1296) and np.isfinite(x).all() and (x != 0).sum() > 1000


def test_check_bits_flags_a_doubled_bit():
    dt = CHIP / 16
    t = (np.arange(int(np.ceil(T_EXP / dt))) + 0.5) * dt
    clean = F.lcd(F.light(t, CODE, CHIP, black=0.04), dt, 3e-3)
    assert real.check_bits(clean, t, CODE, CHIP, 0.04, 0.0) == []
    i = next(k for k in range(len(CODE) - 1) if CODE[k] != CODE[k + 1])       # bit 6: 1 then 0
    bad = CODE[:i + 1] + CODE[i] + CODE[i + 2:]                               # 6 shown twice, 7 skipped
    L = F.lcd(F.light(t, bad, CHIP, black=0.04), dt, 3e-3)
    assert real.check_bits(L, t, CODE, CHIP, 0.04, 0.0) == [i + 1]


@pytest.mark.parametrize("black,tau", [(0.0, 0.0), (0.03, 3e-3)])
def test_fit_readout_recovers_line_time_and_start(black, tau):
    """still scene under the code and under constant light on a rolling shutter, exposure = code length,
    the code starting 21 ms after the first line opens (as coded_illumination.py does)"""
    readout, t0, rows = 0.0632, 0.021, 240
    rng = np.random.default_rng(0)
    scene = 0.2 + 0.6 * rng.random((rows, 1)) * np.ones((1, 160))           # a different shade on every row
    dt = CHIP / 64
    t = np.arange(-0.1, T_EXP + 0.3, dt)
    starts = np.arange(rows) * readout / rows
    shot = lambda c: scene * (F.weights(t, dt, F.lcd(F.light(t, c, CHIP, black, t0), dt, tau) if tau else
                                       F.light(t, c, CHIP, black, t0), T_EXP, starts).sum(1))[:, None]
    coded = shot(CODE) * (1 + 0.005 * rng.standard_normal((rows, 160)))
    const = shot("1" * len(CODE)) * (1 + 0.005 * rng.standard_normal((rows, 160)))
    ro, t0_fit, rms, spread = real.fit_readout(coded, const, CODE, CHIP, T_EXP, 0, black, tau)
    assert abs(ro / readout - 1) < 0.02 and abs(t0_fit - t0) < 1e-3 and rms < 0.2 * spread, (ro, t0_fit, rms, spread)


def test_align_turns_the_reference_to_the_blade():
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    turn = 0.9                                                                 # rad, counter-clockwise
    ref, x = blade(rows, cols, START + turn), blade(rows, cols, START)
    scene = grid(0.0, [0.0], 1.0)
    turned, angle = real.align(ref, x, HUB, scene.r, scene.th)
    inside = F.wrap(scene, np.ones((len(scene.r), len(scene.th)))) > 0.5
    same_path = F.wrap(scene, F.unwrap(scene, x)[0])                           # x through the same resampling
    assert abs(angle - turn) < 0.01 and np.abs(turned - same_path)[inside].mean() < 0.01, angle


def test_find_hub_from_two_stills():
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    turn = np.radians(10)
    f0, f1 = blade(rows, cols, START), blade(rows, cols, START + turn)
    scene = grid(0.0, [0.0], 1.0)
    th = np.arange(START - HALF - 0.3, START + HALF + 0.3, scene.dtheta)
    hub = real.find_hub(f0, f1, turn, (HUB[0] + 8, HUB[1] - 6), scene.r, th)
    assert np.hypot(hub[0] - HUB[0], hub[1] - HUB[1]) < 0.5, hub


def test_reference_scoring_ignores_the_still_angle_and_exposure():
    """a still of the stopped blade at another angle and half the brightness scores the deblur as the true
    scene does. Not exactly: the still is sampled on the pixel grid at another angle, which moved PSNR by
    0.3-0.6 dB here (SSIM 0.0002), the same for every method. Unturned, the still's blade is not in the arc"""
    img, truth = simulate()
    rpm = OMEGA * 60 / (2 * np.pi)
    x, scene, _ = capture.deblur(img, calib(), CODE, CHIP, T_EXP, rpm, "global", "tikhonov", 1e-3)
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    _, p_true, q_true, _ = capture.compare(x, scene, truth)
    turned, p_still, q_still, _ = capture.compare(x, scene, 0.5 * blade(rows, cols, START + 1.0))
    inside = F.wrap(scene, np.ones((len(scene.r), len(scene.th)))) > 0.5
    assert np.corrcoef(turned[inside], truth[inside])[0, 1] > 0.99
    assert p_true > 25 and abs(p_still - p_true) < 1 and abs(q_still - q_true) < 0.005, (p_true, p_still)
