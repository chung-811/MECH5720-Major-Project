import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import capture
import ocr
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


def simulate(readout=0.0, texture=True, omega=OMEGA, dt=CHIP / 40, half=HALF, lag=None):
    """capture of the blade under the code, finer in time than capture.py's own grid. Global shutter
    when readout is 0, else line (row) l opens at l * readout / rows. Scaled like capture.brightness.
    lag: (rise, fall) of the monitor (F.lcd), from black; None = a screen that switches at once"""
    t = (np.arange(int(np.ceil((T_EXP + readout) / dt))) + 0.5) * dt
    L = F.light(t, CODE, CHIP)
    if lag:
        L = F.lcd(L, dt, *lag, start=0.0)
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


def test_rolling_shutter_timing_from_the_run_log():
    """no measured readout in the calibration: the run log's readout and code start stand in for it"""
    readout = 4 * CHIP
    img, _ = simulate(readout)
    rpm = OMEGA * 60 / (2 * np.pi)
    measured = capture.deblur(img, calib(readout), CODE, CHIP, T_EXP, rpm, "rs-row", "tikhonov", 1e-3)[0]
    logged = capture.deblur(img, calib(), CODE, CHIP, T_EXP, rpm, "rs-row", "tikhonov", 1e-3,
                            logged=dict(readout_s=readout, t0=0.0))[0]
    assert np.allclose(measured, logged)
    with pytest.raises(SystemExit):
        capture.deblur(img, calib(), CODE, CHIP, T_EXP, rpm, "rs-row", "tikhonov", 1e-3)


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
    assert real.log_timing(log) == {}


def test_run_log_from_the_capture_gui(tmp_path):
    """the json the data_capture GUI writes next to each DNG (the fields used, from a real OV5647 log)"""
    log = tmp_path / "coded_rs.json"
    log.write_text(json.dumps(dict(code=dict(bits=[1, 0, 1, 1], bit_duration_s=0.0625), actual_exposure_us=1113805,
                                   readout_s=0.064, frame_boundary_monotonic_ns=655709467092,
                                   t0_monotonic_ns=655730755946, late_bits=0, aborted=False)))
    assert real.run_log(log) == ("1011", 0.0625, 1.113805)
    timing = real.log_timing(log)
    assert timing["readout_s"] == 0.064 and abs(timing["t0"] - (0.064 + 0.021288854)) < 1e-9
    assert timing["late_bits"] == 0 and timing["aborted"] is False


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
    run("calibrate.py", "grid", DNG, "--out", tmp_path)
    assert (tmp_path / f"{DNG.stem}_grid.png").exists()
    run("calibrate.py", "init", calib, "--hub", 480, 640, "--radius", 150, 200, "--arc", 200, 230, "--readout", 0.0632)
    run("calibrate.py", "hub", calib, DNG, DNG, "--steps", 0, "--steps-per-rev", 2038)
    with pytest.raises(subprocess.CalledProcessError):
        run("calibrate.py", "hub", calib, DNG, DNG, "--steps", 100)       # steps need --steps-per-rev
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


@pytest.mark.parametrize("black,tau,fall", [(0.0, 0.0, None), (0.03, 3e-3, None), (0.0, 0.044, 0.0)])
def test_fit_readout_recovers_line_time_and_start(black, tau, fall):
    """still scene under the code and under constant light on a rolling shutter, exposure = code length,
    the code starting 21 ms after the first line opens (as coded_illumination.py does)"""
    readout, t0, rows = 0.0632, 0.021, 240
    rng = np.random.default_rng(0)
    scene = 0.2 + 0.6 * rng.random((rows, 1)) * np.ones((1, 160))           # a different shade on every row
    dt = CHIP / 64
    t = np.arange(-0.1, T_EXP + 0.3, dt)
    starts = np.arange(rows) * readout / rows
    shot = lambda c: scene * (F.weights(t, dt, F.lcd(F.light(t, c, CHIP, black, t0), dt, tau, fall) if tau else
                                       F.light(t, c, CHIP, black, t0), T_EXP, starts).sum(1))[:, None]
    coded = shot(CODE) * (1 + 0.005 * rng.standard_normal((rows, 160)))
    const = shot("1" * len(CODE)) * (1 + 0.005 * rng.standard_normal((rows, 160)))
    ro, t0_fit, rms, spread = real.fit_readout(coded, const, CODE, CHIP, T_EXP, 0, black, tau, fall)
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


def stills(turn, seed=0):
    """two noisy stills of the stopped blade, turned by `turn` rad between them, with a static bright
    patch in the background that must not pull the hub"""
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    rng = np.random.default_rng(seed)
    f0, f1 = (blade(rows, cols, START + a) + rng.normal(0, 0.005, SHAPE) for a in (0.0, turn))
    f0[5:20, 90:120] += 0.5
    f1[5:20, 90:120] += 0.5
    return f0, f1


@pytest.mark.parametrize("turn", [0.3, -0.3])
def test_hub_from_stills_needs_no_guess(turn):
    f0, f1 = stills(turn)
    hub, sign, _ = real.hub_from_stills(f0, f1, abs(turn))
    assert np.hypot(hub[0] - HUB[0], hub[1] - HUB[1]) < 0.3 and sign == np.sign(turn)
    r0, r1 = real.radii_from_stills(f0, f1, hub)
    assert r0 <= R0 and R1 <= r1 <= R1 + 8


def test_find_arc_covers_the_blade_and_its_smear():
    img, _ = simulate()
    a0, a1 = np.radians(real.find_arc(img, HUB, np.arange(R0 - 3.0, R1 + 4.0)))
    lo, hi = START - HALF, START + HALF + OMEGA * T_EXP
    assert a0 < lo and hi < a1 and (a1 - a0) - (hi - lo) < np.radians(30)


def test_quick_calibration_end_to_end(tmp_path):
    """the command line, as on the rig: calibrate.py auto from two stills, then deblur a capture whose arc
    is found in it. Simulated images as .npy, the run log as the capture GUI writes it"""
    here = Path(__file__).parents[1]
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], cwd=here, check=True, capture_output=True, text=True)
    f0, f1 = stills(0.3)
    np.save(tmp_path / "still0.npy", f0)
    np.save(tmp_path / "still1.npy", f1)
    img, truth = simulate(lag=(real.MONITOR["rise"], real.MONITOR["fall"]))     # the rig's monitor, as auto writes
    np.save(tmp_path / "cap.npy", img + np.random.default_rng(1).normal(0, 0.002, SHAPE))
    (tmp_path / "cap.json").write_text(json.dumps(dict(
        code=dict(bits=[int(c) for c in CODE], bit_duration_s=CHIP), actual_exposure_us=T_EXP * 1e6, readout_s=0.0,
        frame_boundary_monotonic_ns=0, t0_monotonic_ns=0, late_bits=0, aborted=False)))
    calib = tmp_path / "gs.json"
    out = run("calibrate.py", "auto", calib, tmp_path / "still0.npy", tmp_path / "still1.npy", "--deg",
              np.degrees(0.3), "--out", tmp_path).stdout
    assert "counter-clockwise" in out and (tmp_path / "still1_auto.png").exists()
    run("global-shutter.py", tmp_path / "cap.npy", "--calib", calib, "--rpm", OMEGA * 60 / (2 * np.pi),
        "--solver", "tikhonov", "--lam", 1e-3, "--out", tmp_path)
    x = np.load(tmp_path / "cap_global.npy")
    assert blade_psnr(x, truth) > blade_psnr(img, truth) + 10


def test_auto_flags_a_camera_bump_between_stills(tmp_path):
    """three stills, the camera nudged 2 px before the middle one: two stills cannot show it, the pairs
    of three disagree and auto says so"""
    here = Path(__file__).parents[1]
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    rng = np.random.default_rng(3)
    for i in range(3):
        f = blade(rows, cols, START + 0.3 * i) + rng.normal(0, 0.005, SHAPE)
        np.save(tmp_path / f"s{i}.npy", np.roll(f, (2, -2), (0, 1)) if i == 1 else f)
    out = subprocess.run([sys.executable, "calibrate.py", "auto", tmp_path / "c.json", *[tmp_path / f"s{i}.npy" for i in range(3)],
                          "--deg", str(np.degrees(0.3)), "--out", tmp_path], cwd=here, capture_output=True, text=True, check=True).stdout
    assert "the pairs disagree" in out


def test_wrong_rpm_sign_is_caught_and_corrected(capsys):
    """--rpm typed with the wrong sign: the direction check deblurs the right way round and says so"""
    img, truth = simulate()
    rpm = OMEGA * 60 / (2 * np.pi)
    right = capture.deblur(img, calib(), CODE, CHIP, T_EXP, rpm, "global", "tikhonov", 1e-3)[0]
    assert "other way" not in capsys.readouterr().out                   # the right sign is left alone
    flipped, _, used = capture.deblur(img, calib(), CODE, CHIP, T_EXP, -rpm, "global", "tikhonov", 1e-3)
    assert "the fan turns the other way" in capsys.readouterr().out and used > 0
    assert np.allclose(flipped, right)


def test_blind_speed_at_the_edge_of_its_search_is_flagged(capsys):
    """--rpm 40% low: the true speed is outside the +-20% search. (30% low can instead land on a shallow
    false dip inside it, which is why --find-speed needs --rpm within 20%: see the README)"""
    img, _ = simulate()
    rpm = OMEGA * 60 / (2 * np.pi)
    capture.deblur(img, calib(), CODE, CHIP, T_EXP, 0.6 * rpm, "global", "tikhonov", 1e-3, find_speed=True)
    assert "at the edge of its search" in capsys.readouterr().out


def gui_log(path, readout=0.0):
    path.write_text(json.dumps(dict(code=dict(bits=[int(c) for c in CODE], bit_duration_s=CHIP),
                                    actual_exposure_us=T_EXP * 1e6, readout_s=readout, frame_boundary_monotonic_ns=0,
                                    t0_monotonic_ns=0, late_bits=0, aborted=False)))


def test_monitor_refuses_a_blade_narrower_than_its_smear(tmp_path):
    """a thin rod's two edges smear into each other, which the edge method cannot read: it must not save"""
    here = Path(__file__).parents[1]
    img, _ = simulate(texture=False, half=0.05)            # 0.1 rad wide, smear 0.33 rad
    np.save(tmp_path / "plain.npy", img)
    gui_log(tmp_path / "plain.json")
    c = tmp_path / "c.json"
    c.write_text(json.dumps(calib()))
    r = subprocess.run([sys.executable, "calibrate.py", "monitor", tmp_path / "plain.npy", "--calib", c,
                        "--rpm", str(OMEGA * 60 / (2 * np.pi))], cwd=here, capture_output=True, text=True)
    assert r.returncode != 0 and "not saved" in r.stderr + r.stdout
    assert json.loads(c.read_text())["monitor"] == calib()["monitor"]


def test_auto_keeps_measured_monitor_values(tmp_path):
    here = Path(__file__).parents[1]
    f0, f1 = stills(0.3)
    np.save(tmp_path / "s0.npy", f0)
    np.save(tmp_path / "s1.npy", f1)
    c = tmp_path / "c.json"
    c.write_text(json.dumps({**calib(), "monitor": dict(black=0.01, tau=0.003, t0=0.001), "readout_s": 0.06}))
    subprocess.run([sys.executable, "calibrate.py", "auto", c, tmp_path / "s0.npy", tmp_path / "s1.npy", "--deg",
                    str(np.degrees(0.3)), "--out", tmp_path], cwd=here, capture_output=True, check=True)
    saved = json.loads(c.read_text())
    assert saved["monitor"] == dict(black=0.01, tau=0.003, t0=0.001) and saved["readout_s"] == 0.06


def test_a_file_that_is_not_a_run_log_says_so(tmp_path):
    log = tmp_path / "x.json"
    log.write_text(json.dumps(dict(something="else")))
    with pytest.raises(SystemExit, match="not a run log"):
        real.run_log(log)
    log.write_text(json.dumps(dict(code=dict(bits=[1], bit_duration_s=0.1), actual_exposure_us=1)))
    with pytest.raises(SystemExit, match="not a complete run log"):
        real.log_timing(log)


def test_samples_outside_the_image_carry_no_data():
    """a polar sample off the image gets a NaN line from unwrap and an all-zero row in A, so it constrains
    nothing (it used to be read as 0 and fitted as black); samples inside keep their rows"""
    dt = T_EXP / 40
    t = (np.arange(40) + 0.5) * dt
    scene = F.grid((100.0, 63.5), np.arange(R0, R1 + 1.0), START - HALF, START + HALF, 1 / R1, SHAPE, 0, t, dt, OMEGA)
    lines = F.unwrap(scene, np.ones(SHAPE))[1]
    off = np.isnan(lines).ravel()
    assert off.any() and not off.all()
    A, _ = F.forward_model(scene, F.light(t, CODE, CHIP), T_EXP, lines)
    rows = np.asarray(abs(A).sum(1)).ravel()
    assert np.all(rows[off] == 0) and np.all(rows[~off] > 0)


def test_dpir_deblur_does_not_depend_on_brightness():
    """the same noisy capture at 0.15x brightness deblurs to the same image, scaled: raw levels vary with
    gain and exposure, and DRUNet is not scale-invariant (dpir scales the data to its range)"""
    img, truth = simulate()
    noisy = F.sensor(img, 0.5 * 10000 / 0.8, np.random.default_rng(0))
    rpm = OMEGA * 60 / (2 * np.pi)
    a = capture.deblur(noisy, calib(), CODE, CHIP, T_EXP, rpm, "global", "dpir")[0]
    b = capture.deblur(0.15 * noisy, calib(), CODE, CHIP, T_EXP, rpm, "global", "dpir")[0] / 0.15
    assert blade_psnr(a, truth) > 25
    assert abs(blade_psnr(a, truth) - blade_psnr(b, truth)) < 0.01


def test_blade_noise_reads_the_blade_not_the_dark_background():
    """shot noise makes the blade noisier than the background; blade_noise must read the blade's, where
    a whole-image estimate (what DPIR's noise level used to come from) reads the background's"""
    from skimage.restoration import estimate_sigma
    clean = np.zeros((200, 200))
    clean[50:150, 50:150] = 0.8
    img = clean + np.random.default_rng(0).normal(0, 1, clean.shape) * np.where(clean > 0, 0.02, 0.002)
    assert abs(F.blade_noise(img, np.ones(clean.shape, bool)) / 0.02 - 1) < 0.1
    assert estimate_sigma(img) < 0.5 * 0.02
    few = np.zeros(clean.shape, bool)
    few[60:70, 60:70] = True                                   # 25 blocks: too few to trust
    assert F.blade_noise(img, few) is None


def test_compare_refuses_a_reference_of_another_size():
    """a reference still in another camera mode used to score NaN without a word"""
    with pytest.raises(SystemExit, match="same camera mode"):
        capture.compare(np.zeros((128, 128)), None, np.zeros((64, 64)))


@pytest.mark.parametrize("before", ["none", "no lag"])
def test_deblur_py_one_command(tmp_path, before):
    """deblur.py, as in the lab: a stills folder (3 stills turned the same steps apart, then a reference) and
    a capture folder named with its speed. It calibrates, saves calib.json with the stills, deblurs and scores"""
    here = Path(__file__).parents[1]
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    still_dir, cap_dir = data / "gs_stills", data / f"gs_{OMEGA * 60 / (2 * np.pi):.4f}rpm"
    still_dir.mkdir(parents=True), cap_dir.mkdir()
    for i, a in enumerate((0.0, 0.3, 0.6, 0.0)):                 # 3 stills 0.3 rad apart, then the reference
        np.save(still_dir / f"coded_gs_20260101_00000{i}.npy", blade(rows, cols, START + a) + rng.normal(0, 0.005, SHAPE))
    img, truth = simulate(lag=(real.MONITOR["rise"], real.MONITOR["fall"]))    # the rig's monitor
    np.save(cap_dir / "coded_gs_20260101_000100.npy", img + rng.normal(0, 0.002, SHAPE))
    calib_path = tmp_path / "results" / "gs_stills" / "calib.json"        # next to the results, not in the capture folder
    if before == "no lag":                                           # one from before: a screen that switches at once
        calib_path.parent.mkdir(parents=True)
        calib_path.write_text(json.dumps({**calib(), "static": "calib_static.npy"}))
    (cap_dir / "coded_gs_20260101_000100.json").write_text(json.dumps(dict(
        code=dict(bits=[int(c) for c in CODE], bit_duration_s=CHIP), actual_exposure_us=T_EXP * 1e6, readout_s=0.0,
        frame_boundary_monotonic_ns=0, t0_monotonic_ns=0, late_bits=0, aborted=False)))
    for d in (still_dir, cap_dir):
        d.chmod(0o555)                                           # read-only, as OneDrive leaves them
    try:
        out = subprocess.run([sys.executable, "deblur.py", str(cap_dir), "--stills", str(still_dir), "--steps", "1000",
                              "--steps-per-rev", str(2 * np.pi / 0.3 * 1000), "--out", str(tmp_path / "results" / "run")],
                             cwd=here, capture_output=True, text=True)
    finally:
        for d in (still_dir, cap_dir):
            d.chmod(0o755)
    assert out.returncode == 0, out.stdout + out.stderr
    assert ("treats the monitor as switching instantly" in out.stdout) == (before == "no lag")   # so redone
    assert calib_path.exists() and "PSNR" in out.stdout                   # calibrated, and scored on the reference
    saved = json.loads(calib_path.read_text())
    assert saved["static"] == "calib_static.npy" and saved["monitor"] == real.MONITOR
    assert "static scene taken out" in out.stdout
    x = np.load(tmp_path / "results" / "run" / "coded_gs_20260101_000100_global.npy")
    assert blade_psnr(x, truth) > blade_psnr(img, truth) + 10


def test_remove_static_leaves_only_what_moves():
    """the static scene is fitted on the pixels outside the blade circle and taken out: under half the light a
    fixed object is half as bright, inside the circle too, and only the moving part is left"""
    static = np.full(SHAPE, 0.05)
    static[5:20, 90:120] = 0.6                                 # a bright fixed object outside the circle
    static[60:70, 30:40] = 0.4                                 # and one inside it, like the rig's bolts
    moving = np.zeros(SHAPE)
    moving[40:50, 60:70] = 0.3
    img = 0.5 * static + 0.01 + moving + np.random.default_rng(0).normal(0, 0.001, SHAPE)
    out, a, b = real.remove_static(img, static, HUB, 30)
    assert abs(a - 0.5) < 0.01 and abs(b - 0.01) < 0.002
    assert np.abs(out - moving).max() < 0.01
    with pytest.raises(ValueError, match="fills the picture"):
        real.remove_static(img, static, HUB, 200)
    with pytest.raises(ValueError, match="calibrate from stills"):
        real.remove_static(img, static[:-2], HUB, 30)


def test_lcd_by_default_is_the_old_single_lag():
    """F.lcd(L, dt, tau) gives exactly what it did before the rise and fall were split"""
    def before(L, dt, tau):
        out, a = np.empty_like(L), 1 - np.exp(-dt / tau)
        out[0] = L[0]
        for k in range(1, len(L)):
            out[k] = out[k - 1] + a * (L[k] - out[k - 1])
        return out
    dt = CHIP / 16
    t = (np.arange(int(np.ceil(T_EXP / dt))) + 0.5) * dt
    L = F.light(t, CODE, CHIP, black=0.04, t0=0.3 * CHIP)
    for tau in (1e-3, 3e-3, 0.044):
        assert np.array_equal(F.lcd(L, dt, tau), before(L, dt, tau))
        assert np.array_equal(F.lcd(L, dt, tau, tau), before(L, dt, tau))


def test_lcd_rises_slowly_and_falls_at_once():
    """the rig's monitor: a 67 ms flash from black climbs as 1 - exp(-t / rise) and goes dark at once"""
    dt, rise = 1e-3, 0.044
    L = np.r_[np.zeros(10), np.ones(67), np.zeros(10)]
    out = F.lcd(L, dt, rise, 0.0, start=0.0)
    assert np.allclose(out[10:77], 1 - np.exp(-np.arange(1, 68) * dt / rise), rtol=0, atol=1e-12)
    assert np.all(out[:10] == 0) and np.all(out[77:] == 0)
    slow_fall = F.lcd(L, dt, 0.0, rise, start=0.0)                       # and the other way round
    assert np.all(slow_fall[10:77] == 1)
    assert np.allclose(slow_fall[77:], np.exp(-np.arange(1, 11) * dt / rise), rtol=0, atol=1e-12)
    assert np.array_equal(F.lcd(L, dt, 0.0, 0.0), L)                     # 0 = instant
    on = np.ones(5)                                                      # start: what was shown before
    assert F.lcd(on, dt, rise)[0] == 1 and F.lcd(on, dt, rise, start=0.0)[0] == pytest.approx(1 - np.exp(-dt / rise))


@pytest.mark.parametrize("monitor,lag", [
    (dict(black=0.0, rise=0.044, fall=0.0, t0=0.0), (0.044, 0.0)),        # what auto writes
    (dict(black=0.01, tau=0.003, t0=0.0), (0.003, 0.003)),                # fitted by calibrate.py monitor
    (dict(black=0.0, tau=0.0, t0=0.0), (0.0, 0.0)),                       # a calibration from before
    ({}, (0.0, 0.0)),
    (dict(rise=0.044), (0.044, 0.0)),
    (dict(rise=0.02, fall=0.01, tau=0.5), (0.02, 0.01)),                 # rise and fall win over tau
    (dict(tau=0.003, fall=0.0), (0.003, 0.0)),                            # each falls back to tau on its own
])
def test_monitor_lag_reads_the_calibration(monitor, lag):
    assert real.monitor_lag(monitor) == lag


@pytest.mark.parametrize("monitor", [dict(rise=-0.01), dict(fall=1.0), dict(tau=float("nan")), dict(rise="44 ms"),
                                     dict(rise=None), None, [0.044]])
def test_monitor_lag_refuses_nonsense(monitor):
    with pytest.raises(SystemExit):
        real.monitor_lag(monitor)


def test_brightness_starts_the_lag_from_black():
    """the screen shows black for the pre-roll, so even the first bit has to brighten"""
    dt = CHIP / 16
    t = (np.arange(int(np.ceil(T_EXP / dt))) + 0.5) * dt
    scene = grid(OMEGA, t, dt)
    assert CODE[0] == "1"
    ideal = capture.brightness(scene, CODE, CHIP, T_EXP, calib()["monitor"])
    slow = capture.brightness(scene, CODE, CHIP, T_EXP, real.MONITOR)
    assert ideal[0] == ideal.max() and slow[0] < 0.2 * slow.max()
    for L in (ideal, slow):                                              # both scaled the same way
        assert F.weights(scene.t, scene.dt, L, T_EXP, [0.0]).sum() == pytest.approx(1)


def test_deblur_uses_the_monitors_rise():
    """a capture under the rig's monitor deblurs better with its rise in the calibration than with a screen that
    switches at once: the calibration's monitor reaches the forward model"""
    img, truth = simulate(lag=(real.MONITOR["rise"], real.MONITOR["fall"]))
    rpm = OMEGA * 60 / (2 * np.pi)
    ideal = capture.deblur(img, calib(), CODE, CHIP, T_EXP, rpm, "global", "tikhonov", 1e-3)[0]
    rig = capture.deblur(img, {**calib(), "monitor": dict(real.MONITOR)}, CODE, CHIP, T_EXP, rpm, "global", "tikhonov",
                         1e-3)[0]
    assert blade_psnr(rig, truth) > blade_psnr(ideal, truth) + 5, (blade_psnr(rig, truth), blade_psnr(ideal, truth))  # 10 dB


def test_auto_writes_the_measured_monitor(tmp_path):
    """auto writes the rig's monitor over one that switches at once, keeping a measured code start"""
    here = Path(__file__).parents[1]
    f0, f1 = stills(0.3)
    np.save(tmp_path / "s0.npy", f0)
    np.save(tmp_path / "s1.npy", f1)
    c = tmp_path / "c.json"
    auto = [sys.executable, "calibrate.py", "auto", c, tmp_path / "s0.npy", tmp_path / "s1.npy", "--deg",
            str(np.degrees(0.3)), "--out", tmp_path]
    subprocess.run(auto, cwd=here, capture_output=True, check=True)
    assert json.loads(c.read_text())["monitor"] == real.MONITOR
    c.write_text(json.dumps({**calib(), "monitor": dict(black=0.0, tau=0.0, t0=0.021)}))
    subprocess.run(auto, cwd=here, capture_output=True, check=True)
    assert json.loads(c.read_text())["monitor"] == {**real.MONITOR, "t0": 0.021}


def test_brightness_with_the_code_starting_before_the_exposure():
    """t0 < 0 (a rolling shutter's code can start before the first line opens): the screen has been brightening
    for -t0 already. Same as the lag run on a long grid from well before the code; t0 = 0 is unchanged"""
    dt = CHIP / 16
    t = (np.arange(int(np.ceil(T_EXP / dt))) + 0.5) * dt
    scene = grid(OMEGA, t, dt)
    rise, fall = real.MONITOR["rise"], real.MONITOR["fall"]
    for t0 in (-0.3 * CHIP, -2.5 * CHIP, 0.0, 0.4 * CHIP):
        k = 200                                                          # samples of black before the grid
        long = np.r_[t[0] - dt * np.arange(k, 0, -1), t]
        ref = F.lcd(F.light(long, CODE, CHIP, 0.0, t0), dt, rise, fall, start=0.0)[k:]
        ref /= F.weights(t, dt, ref, T_EXP, [0.0]).sum()
        got = capture.brightness(scene, CODE, CHIP, T_EXP, {**real.MONITOR, "t0": t0})
        assert np.allclose(got, ref, rtol=0, atol=1e-12), t0
    before = F.lcd(F.light(t, CODE, CHIP, 0.0, 0.0), dt, rise, fall, start=0.0)     # the lag as first written
    assert np.array_equal(capture.brightness(scene, CODE, CHIP, T_EXP, real.MONITOR),
                          before / F.weights(t, dt, before, T_EXP, [0.0]).sum())


def test_scripts_warn_about_a_calibration_without_the_lag(tmp_path):
    """global-shutter.py and the rolling shutter scripts take a calibration as given: one from before the
    measured monitor is flagged, not silently deblurred with a screen that switches at once"""
    here = Path(__file__).parents[1]
    img, _ = simulate()
    np.save(tmp_path / "cap.npy", img)
    for monitor, warned in ((calib()["monitor"], True), (dict(real.MONITOR), False)):
        c = tmp_path / "c.json"
        c.write_text(json.dumps({**calib(), "monitor": monitor}))
        out = subprocess.run([sys.executable, "global-shutter.py", tmp_path / "cap.npy", "--calib", c, "--rpm", "1",
                              "--check", "--out", tmp_path], cwd=here, capture_output=True, text=True, check=True)
        assert ("switching at once" in out.stdout) == warned


def fan(rows, cols, angle, mark=True):
    """three look-alike blades 120 deg apart (the rig's), told apart only by a dark patch near the tip of the
    first, as the rig's are only by their letters"""
    img = sum(blade(rows, cols, angle + k * 2 * np.pi / 3, texture=False) - 0.05 for k in range(3)) + 0.05
    if mark:
        a = angle
        rr, cc = HUB[0] - 48 * np.sin(a), HUB[1] + 48 * np.cos(a)
        img = np.where(np.hypot(rows - rr, cols - cc) < 5, 0.1, img)
    return img


def fan_capture(angle=START, omega=OMEGA, dt=CHIP / 40):
    """the fan under the code, global shutter, as simulate()"""
    t = (np.arange(int(np.ceil(T_EXP / dt))) + 0.5) * dt
    L = F.light(t, CODE, CHIP)
    L = L / F.weights(t, dt, L, T_EXP, [0.0]).sum()
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    w = F.weights(t, dt, L, T_EXP, [0.0])[0]
    return sum(wk * fan(rows, cols, angle + omega * tk) for wk, tk in zip(w, t)), t, w


def test_blade_angle_tells_look_alike_blades_apart():
    """the whole capture, every blade, against the whole still blurred the same way: the marked blade lands
    on the marked blade, not on a look-alike 120 deg off"""
    img, t, w = fan_capture()
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    r = np.arange(R0, R1 + 1.0)
    for turn in (2.0, -1.0):
        ref = fan(rows, cols, START + turn)                         # the still, at another angle
        angle, margin = real.blade_angle(ref, img, HUB, r, OMEGA, t, w)
        off = (angle - turn + np.pi) % (2 * np.pi) - np.pi
        assert abs(off) < np.radians(1.5) and margin > 1.02, (np.degrees(off), margin)


def test_align_near_stays_on_that_blade():
    """align, told roughly where the blade is, keeps to it even where a look-alike matches about as well"""
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    x, ref = fan(rows, cols, START, mark=False), fan(rows, cols, START + 1.0, mark=False)
    r, th = np.arange(R0, R1 + 1.0), np.arange(START - 0.5, START + 0.5, 1 / R1)
    for k in (0, 1, 2):                                             # each of the three look-alikes on request
        want = 1.0 + k * 2 * np.pi / 3
        got = real.align(ref, x, HUB, r, th, near=want + 0.05)[1]
        assert abs((got - want + np.pi) % (2 * np.pi) - np.pi) < np.radians(1), (k, np.degrees(got))


def test_speed_from_blur_finds_the_speed():
    """the speed from the blur of the still, searched from guesses 15% off, within 0.5% of the true one
    (nearest-step blur on bilinear samples read 1.2% low here). Only for these two still angles: on this
    small blade, whose 1 px edges turn into a different pixel staircase at every angle, other angles move
    the estimate by up to 9%. On the real captures (5, 8 rpm) the three stills agreed within 1.2%"""
    img, _ = simulate()
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    rpm = OMEGA * 60 / (2 * np.pi)
    for guess, turn in ((1.15, 1.0), (0.85, -2.0)):
        ref = blade(rows, cols, START + turn)                       # the stopped blade, somewhere else
        est, edge = capture.speed_from_blur(img, calib(), ref, CODE, CHIP, T_EXP, guess * rpm)
        assert not edge and abs(est / rpm - 1) < 0.005, (guess, turn, est / rpm)


@pytest.mark.skipif(subprocess.run(["which", "tesseract"], capture_output=True).returncode != 0,
                    reason="tesseract not installed")
def test_read_letter_reads_the_card():
    """an upright letter on a card, with dark background and card edges round it, reads as that letter; a blank
    card reads as nothing"""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=60)
    for ch in "ABC":
        im = Image.new("L", (85, 85), 10)                           # background
        d = ImageDraw.Draw(im)
        d.rectangle((8, 6, 80, 78), fill=230)                       # the card, cut by the square's edge
        d.text((44, 42), ch, fill=20, font=font, anchor="mm")
        assert ocr.read_letter(np.asarray(im, float) / 255) == ch
    assert ocr.read_letter(np.full((85, 85), 0.9)) == ""


def test_card_crop_turns_the_blade_to_point_down():
    """a bright spot further out along the blade than the card's centre ends up straight below it"""
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    thc, rc = 0.7, 40.0
    spot = np.hypot(rows - (HUB[0] - 52 * np.sin(thc)), cols - (HUB[1] + 52 * np.cos(thc))) < 2
    c = ocr.card_crop(spot.astype(float), HUB, rc, thc, half=20)
    r, k = np.unravel_index(np.argmax(c), c.shape)
    assert abs(k - 20) <= 1 and r > 20 + 9, (r, k)


@pytest.mark.parametrize("code,chip,sign", [(CODE, CHIP, -1), ("1", T_EXP, 1), ("1", T_EXP, -1)])
def test_speed_from_blur_steady_light_and_either_way(code, chip, sign):
    """steady light (one bit as long as the exposure: the sub-frames are then set by the speed, the case that
    made the match jump between candidate speeds) and the fan turning either way: never off the search, the
    parabola never more than a step, coded within 0.5%, steady within 3%"""
    omega = sign * OMEGA
    dt = CHIP / 40
    t = (np.arange(int(np.ceil(T_EXP / dt))) + 0.5) * dt
    L = F.light(t, code, chip)
    w = F.weights(t, dt, L / F.weights(t, dt, L, T_EXP, [0.0]).sum(), T_EXP, [0.0])[0]
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    img = sum(wk * blade(rows, cols, START + omega * tk) for wk, tk in zip(w, t))
    rpm = omega * 60 / (2 * np.pi)
    est, edge = capture.speed_from_blur(img, calib(), blade(rows, cols, START + 1.0), code, chip, T_EXP, 1.1 * rpm)
    assert not edge and abs(est / rpm - 1) < (0.005 if code == CODE else 0.03), est / rpm


def test_steady_light_keeps_the_given_direction(capsys):
    """steady light cannot show which way the fan turns, so the deblur never flips --rpm's sign on it"""
    dt = CHIP / 40
    t = (np.arange(int(np.ceil(T_EXP / dt))) + 0.5) * dt
    L = F.light(t, "1", T_EXP)
    w = F.weights(t, dt, L / F.weights(t, dt, L, T_EXP, [0.0]).sum(), T_EXP, [0.0])[0]
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    img = sum(wk * blade(rows, cols, START + OMEGA * tk) for wk, tk in zip(w, t))
    rpm = OMEGA * 60 / (2 * np.pi)
    for given in (rpm, -rpm):
        used = capture.deblur(img, calib(), "1", T_EXP, T_EXP, given, "global", "tikhonov", 1e-3)[2]
        assert np.sign(used) == np.sign(given)
    assert "cannot be told" in capsys.readouterr().out


def test_compare_with_the_capture_scores_the_right_blade(monkeypatch):
    """compare given the capture scores the deblurred blade against the blade real.blade_angle names (which
    tells look-alikes apart, test above), not whichever blade of the still best matches the deblur: here the
    right one, then a look-alike it is told to use, which align on its own would not pick"""
    img, t, w = fan_capture()
    rows, cols = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    rpm = OMEGA * 60 / (2 * np.pi)
    cal = {**calib(), "arc_deg": list(np.degrees([START - 0.6, START + 0.25 + OMEGA * T_EXP + 0.1]))}
    x, scene, _ = capture.deblur(img, cal, CODE, CHIP, T_EXP, rpm, "global", "tikhonov", 1e-3)
    for turn in (2.0, -1.0):
        capture.compare(x, scene, fan(rows, cols, START + turn), img)
        # the turned still against the fan with its mark on the capture's blade, and on each look-alike
        err = [np.abs(scene.truth - fan(rows, cols, START + k * 2 * np.pi / 3))[scene.mask].mean() for k in (0, 1, 2)]
        assert np.argmin(err) == 0 and err[0] < 0.5 * min(err[1:]), (turn, err)
        assert scene.blade_margin > 1.02
    turn = 2.0
    monkeypatch.setattr(real, "blade_angle", lambda *a: (turn + 2 * np.pi / 3, 9.0))      # names a look-alike
    capture.compare(x, scene, fan(rows, cols, START + turn), img)
    err = [np.abs(scene.truth - fan(rows, cols, START + k * 2 * np.pi / 3))[scene.mask].mean() for k in (0, 1, 2)]
    assert np.argmin(err) != 0 and scene.blade_margin == 9.0, err
