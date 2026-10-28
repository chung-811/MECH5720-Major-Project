"""Regenerates every figure in the report, plus numbers.tex (the numbers the text quotes), so no
number is typed by hand.

    cd software/deconvolution
    uv run python ../../report/figures/make_figures.py

Inputs:
  - code files: software/data_capture/coded/params/          (in the repo)
  - real captures and run logs: output/gs_cd, output/rs_cd   (Ollie's 21 Oct hand-spun CD session)
  - rig scores: figures/data/rig_psnr.csv                    (deblur.py on the 26 and 28 Oct captures, in the repo)
  - simulation results: software/deconvolution/out/demo_*.npz and rows_*_axis*.npz, written by
        uv run python demo.py renders/blade.npz
        uv run python demo.py renders/blade_tex.npz
        uv run python rows.py renders/blade_tex.npz 0
        uv run python rows.py renders/blade_tex.npz 1
    (the renders come from blender/render_fan.py, see software/deconvolution/STUDIES.md)

A figure whose input is missing is skipped and listed at the end (exit code 1).
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECONV = ROOT / "software/deconvolution"
PARAMS = ROOT / "software/data_capture/coded/params"
OUTPUT = ROOT / "output"
SIM = DECONV / "out"
sys.path.insert(0, str(DECONV))
import flutter as F  # noqa: E402  (RASKAR52, CHIP: the same constants the pipeline uses)

COL, TEXT = 3.3125, 6.875          # ICCP column and text width, inches
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})

numbers = {}                        # macro name -> text, written to numbers.tex
missing = []


def num(name, value, fmt="{:.1f}"):
    numbers[name] = fmt.format(value) if not isinstance(value, str) else value


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}.pdf")


def need(path):
    if not path.exists():
        missing.append(str(path.relative_to(ROOT)))
        return False
    return True


def bits(name):
    j = json.loads((PARAMS / name).read_text())
    return np.array([int(c) for c in j["sequence"] if c in "01"]), j["total_duration_s"]


# ---------------------------------------------------------------- designed code and its spectrum
def fig_code():
    rig, rig_T = bits("code_16bit_best_4refresh.json")
    codes = {f"rig code ({len(rig)} bits)": rig,
             f"box, steady light ({len(rig)} bits)": np.ones(len(rig)),
             "Raskar et al. (52 bits)": np.array([int(c) for c in F.RASKAR52])}
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.45), width_ratios=(1, 1.35))
    for i, (k, c) in enumerate(codes.items()):
        t = np.arange(len(c) + 1) / len(c)
        a.step(t, np.r_[c, c[-1]] * 0.8 + 1.2 * (2 - i), where="post", color=f"C{i}")
        pad = 64 * len(c)                                       # zero-padded DFT: a smooth curve
        H = np.abs(np.fft.rfft(c, pad)) / c.sum()               # 1 at zero frequency
        f = np.fft.rfftfreq(pad)                                # cycles per bit
        b.semilogy(f[1:], np.maximum(H[1:], 1e-3), color=f"C{i}", label=k)
        m = H[1:].min()
        num({0: "rigMinMTF", 1: "boxMinMTF", 2: "raskarMinMTF"}[i], m, "{:.3f}" if m > 1e-3 else "0")
    a.set_yticks([]), a.set_xlabel("time / exposure"), a.set_title("(a) light in one exposure")
    b.set_xlabel("frequency (cycles per bit)"), b.set_ylabel("magnitude"), b.set_ylim(1e-3, 1.2)
    b.set_title("(b) spectrum of the blur"), b.legend(loc="lower left", fontsize=5.5, frameon=False)
    num("rigBits", len(rig), "{}"), num("rigCodeMs", 1e3 * rig_T / len(rig), "{:.1f}"), num("rigCodeS", rig_T, "{:.3f}")
    save(fig, "code_spectrum")


# ---------------------------------------------------------------- exposure timing
def fig_timing():
    rig, Tc = bits("code_16bit_best_4refresh.json")
    Tro, pad, rows = 0.25, 0.05, 6     # readout drawn 4x the OV5647's (~64 ms) so the stagger is visible
    fig, axs = plt.subplots(2, 1, figsize=(COL, 2.0), sharex=True)
    for a, (title, T, first) in zip(axs, [
            ("(a) exposure = code length: rows see different slices", Tc, -Tro / 2),
            ("(b) exposure = code + readout + padding: every row sees all",
             Tc + Tro + pad, -Tro)]):
        for k, v in enumerate(rig):                             # the code, shaded behind the rows
            if v:
                a.axvspan(k * Tc / len(rig), (k + 1) * Tc / len(rig), color="0.85", lw=0)
        for r in range(rows):
            s = first + r * Tro / (rows - 1)
            a.barh(r, T, left=s, height=0.55, color="C0" if r else "k", alpha=0.75)
        a.axvline(0, color="k", lw=0.5), a.axvline(Tc, color="k", lw=0.5)
        a.set_yticks([0, rows - 1], ["first row", "last row"]), a.invert_yaxis()
        a.set_title(title, fontsize=6.5, loc="left")
    axs[1].set_xlabel("time from the start of the code (s)")
    save(fig, "timing")


# ---------------------------------------------------------------- real data: CD capture and monitor timing
def fig_real():
    logs = sorted(glob.glob(str(OUTPUT / "*_cd/*.json")))
    png = OUTPUT / "gs_cd/coded_gs_20261021_222612.png"
    if not logs or not need(png):
        if not logs:
            missing.append("output/*_cd/*.json")
        return
    late, idx, cam = [], [], []
    for f in logs:
        j = json.loads(Path(f).read_text())
        for b in j["bits"]:
            late.append(1e3 * b["late_s"]), idx.append(b["index"]), cam.append(j["shutter"])
    late, idx, cam = map(np.array, (late, idx, cam))
    bit_ms = 1e3 * json.loads(Path(logs[0]).read_text())["code"]["bit_duration_s"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.5), width_ratios=(1, 1.25))
    a.imshow(plt.imread(png)), a.axis("off"), a.set_title("(a) hand-spun CD, coded light")
    for c, m, k in (("global", "o", "C0"), ("rolling", "s", "C1")):
        s = cam == c
        b.scatter(idx[s] + (0.15 if c == "rolling" else -0.15), late[s], s=4, marker=m, color=k,
                  label=f"{c} shutter runs")
    b.set_xlabel("bit of the code"), b.set_ylabel("shown late (ms)")
    b.set_title(f"(b) bit timing, {bit_ms:.1f} ms bits"), b.legend(frameon=False, fontsize=5.5)
    num("realRuns", len(logs), "{}"), num("realBits", late.size, "{}"), num("realBitMs", bit_ms)
    num("realWithinOneMs", int((late <= 1).sum()), "{}"), num("realMaxLateMs", late.max())
    num("realMaxLatePct", 100 * late.max() / bit_ms, "{:.0f}")
    slow = sorted(set(idx[late > 1].tolist()))          # which bits were more than 1 ms late
    num("realLateBits", ", ".join(map(str, slow)) if slow else "none")
    save(fig, "real_cd")


# ---------------------------------------------------------------- real rig: coded against steady light
WORD = {3: "Three", 5: "Five", 8: "Eight"}


def fig_rig():
    """PSNR against speed on the rig (global shutter), coded and steady light, from data/rig_psnr.csv"""
    path = HERE / "data/rig_psnr.csv"
    if not need(path):
        return
    rows = [l.split(",") for l in path.read_text().splitlines() if l and not l.startswith(("#", "date"))]
    score = {}
    for date, light, rpm, _, p, _ in rows:
        score.setdefault((date, light, int(rpm)), []).append(float(p))
    fig, a = plt.subplots(figsize=(COL, 1.7))
    for light, col, mk, dx in (("coded", "C0", "o", -0.12), ("steady", "C3", "s", 0.12)):
        rpms = sorted(r for d, l, r in score if d == "2026-10-28" and l == light)
        for r in rpms:
            p = np.array(score[("2026-10-28", light, r)])
            a.scatter(np.full(p.size, r + dx) + np.linspace(-0.05, 0.05, p.size), p, s=3, color=col, alpha=0.35, lw=0)
            num(f"rig{light.title()}{WORD[r]}", p.mean()), num(f"rig{light.title()}{WORD[r]}Sd", p.std())
            num(f"rig{light.title()}{WORD[r]}N", p.size, "{}")
        m = [np.mean(score[("2026-10-28", light, r)]) for r in rpms]
        sd = [np.std(score[("2026-10-28", light, r)]) for r in rpms]
        a.errorbar(np.array(rpms) + dx, m, sd, fmt=mk + "-", color=col, ms=3, capsize=2,
                   label=f"{light} light + deblur")
    gaps = [np.mean(score[("2026-10-28", "coded", r)]) - np.mean(score[("2026-10-28", "steady", r)])
            for r in sorted(WORD) if ("2026-10-28", "steady", r) in score]
    num("rigGapMin", min(gaps)), num("rigGapMax", max(gaps))
    p26 = np.array(score[("2026-10-26", "coded", 3)])
    num("rigOctTwentySix", p26.mean()), num("rigOctTwentySixSd", p26.std()), num("rigOctTwentySixN", p26.size, "{}")
    a.set_xticks(sorted(WORD)), a.set_xlabel("fan speed (rpm)"), a.set_ylabel("PSNR (dB)")
    a.legend(frameon=False, fontsize=6), a.grid(alpha=0.3, lw=0.4)
    save(fig, "rig_psnr")


# ---------------------------------------------------------------- simulation: deblurred images and speed
def crop(truth, margin=12):
    r, c = np.nonzero(truth > 0.02 * truth.max())
    return slice(max(r.min() - margin, 0), r.max() + margin), slice(max(c.min() - margin, 0), c.max() + margin)


def demo(stem):
    path = SIM / f"demo_{stem}.npz"
    return np.load(path) if need(path) else None


def fig_demo():
    d = {s: demo(s) for s in ("blade", "blade_tex")}
    tags = {"blade": "Letters", "blade_tex": "Tex"}
    keys = {"GS coded, per-frame": "Gs", "RS coded, per-frame": "RsFrame", "RS coded, per-row": "RsRow",
            "GS constant, deblurred": "Const", "GS short (1 chip, x16)": "Short"}
    for s, z in d.items():
        if z is None:
            continue
        for m, (p, q) in zip(z["methods"], z["psnr_ssim"]):
            num(f"psnr{keys[str(m)]}{tags[s]}", p), num(f"ssim{keys[str(m)]}{tags[s]}", q, "{:.3f}")
        if "tik_methods" not in z:                   # an npz from before demo.py saved Tikhonov
            missing.append(f"Tikhonov scores in out/demo_{s}.npz (re-run demo.py)")
            continue
        for m, (p, q) in zip(z["tik_methods"], z["tik_psnr_ssim"]):
            num(f"tik{keys[str(m)]}{tags[s]}", p), num(f"tikSsim{keys[str(m)]}{tags[s]}", q, "{:.3f}")
        err = 100 * (float(z["est"]) - 1)
        num(f"speedErr{tags[s]}", abs(err), "{:.1f}"), num(f"speedErrSigned{tags[s]}", err, "{:+.1f}")

    z = d["blade_tex"] if d["blade_tex"] is not None else d["blade"]
    if z is None:
        return
    names, panels, vmax = [str(n) for n in z["names"]], z["panels"].astype(np.float32), float(z["vmax"])
    score = {str(m): p for m, (p, _) in zip(z["methods"], z["psnr_ssim"])}
    rr, cc = crop(panels[names.index("truth")])
    pick = lambda n: panels[names.index(n)][rr, cc]

    # teaser: coded capture, deblurred, stopped fan
    fig, axs = plt.subplots(1, 3, figsize=(COL, 1.55))
    for a, (n, t) in zip(axs, [("GS coded (raw)", "coded capture"),
                               ("GS coded, per-frame", f"deblurred, {score['GS coded, per-frame']:.1f} dB"),
                               ("truth", "fan stopped")]):
        a.imshow(pick(n), cmap="gray", vmin=0, vmax=vmax), a.set_title(t), a.axis("off")
    save(fig, "teaser")

    # every method, raw above, deblurred below
    order = [("GS coded (raw)", "GS coded, per-frame", "global shutter, coded"),
             ("RS coded (raw)", "RS coded, per-frame", "rolling, coded, per frame"),
             ("RS coded (raw)", "RS coded, per-row", "rolling, coded, per row"),
             ("GS constant (raw)", "GS constant, deblurred", "global shutter, steady light"),
             ("GS short (raw)", "GS short (1 chip, x16)", "short exposure, $\\times$16")]
    fig, axs = plt.subplots(2, 5, figsize=(TEXT, 2.6))
    for j, (raw, out, title) in enumerate(order):
        axs[0, j].imshow(pick(raw), cmap="gray", vmin=0, vmax=vmax), axs[0, j].set_title(title)
        lab = f"{score[out]:.1f} dB" if out in score else "no deblurring"
        axs[1, j].imshow(pick(out), cmap="gray", vmin=0, vmax=vmax), axs[1, j].set_title(lab)
    for a in axs.ravel():
        a.axis("off")
    axs[0, 0].text(-0.08, 0.5, "capture", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right")
    axs[1, 0].text(-0.08, 0.5, "result", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right")
    save(fig, "qualitative")

    # blind speed: residual against candidate speed, both scenes
    fig, a = plt.subplots(figsize=(COL, 1.6))
    for s, lab, mk in (("blade", "letters", "o"), ("blade_tex", "photo + letters", "s")):
        if d[s] is not None:
            r = d[s]["residual"]
            a.plot(d[s]["omegas"], r / r.min(), f"{mk}-", ms=2.5, label=f"{lab}: estimate {100 * (float(d[s]['est']) - 1):+.1f}%")
    a.axvline(1, color="k", lw=0.5), a.set_xlabel("candidate speed / true speed")
    a.set_ylabel("data residual / minimum"), a.legend(frameon=False)
    save(fig, "speed")


# ---------------------------------------------------------------- simulation: criterion 1, PSNR by row band
def fig_rows():
    for axis in (0, 1):
        path = SIM / f"rows_blade_tex_axis{axis}.npz"
        if not need(path):
            continue
        z = np.load(path)
        keys, bands = [str(k) for k in z["keys"]], z["bands"]
        res = dict(zip(keys, bands))
        codes = list(dict.fromkeys(k.split("|")[0] for k in keys))
        ros = [str(r) for r in z["readout_names"]]
        chip = float(z["chip"])
        fig, axs = plt.subplots(1, len(codes), figsize=(TEXT, 2.2), sharey=True)
        for a, cname, code in zip(axs, codes, z["codes"]):
            a.plot(res[f"{cname}|global|"], "k-", lw=1.6, label="global shutter")
            for ro, col in zip(ros, ("C0", "C1", "C3")):
                a.plot(res[f"{cname}|{ro}|per-frame"], "--", color=col, label=f"rolling {ro}, per frame")
                a.plot(res[f"{cname}|{ro}|per-row"], "-", color=col, label=f"rolling {ro}, per row")
            a.set_title(f"{len(str(code))}-bit code, {1e3 * len(str(code)) * chip:.0f} ms exposure")
            a.set_xlabel("band of sensor lines, first to last read")
        axs[0].set_ylabel("PSNR over the blade (dB)")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
        save(fig, f"rows_axis{axis}")
        # numbers: mean and worst band per case, plus the largest per-row shortfall against global
        A = "ABCDEFGHIJ"[axis]
        gap = mean_gap = -np.inf                    # global minus per-row: worst band, worst mean
        for cname in codes:
            g = res[f"{cname}|global|"]
            for ro in ros:
                r = res[f"{cname}|{ro}|per-row"]
                gap, mean_gap = max(gap, np.nanmax(g - r)), max(mean_gap, np.nanmean(g) - np.nanmean(r))
        num(f"rowsMaxGap{A}", gap, "{:.1f}"), num(f"rowsMaxMeanGap{A}", mean_gap, "{:.1f}")
        worst = res[f"{codes[-1]}|{ros[-1]}|per-frame"]
        num(f"rowsShortFrameMean{A}", np.nanmean(worst)), num(f"rowsShortFrameWorst{A}", np.nanmin(worst))
        num(f"rowsShortRowMean{A}", np.nanmean(res[f"{codes[-1]}|{ros[-1]}|per-row"]))
        num(f"rowsShortGlobalMean{A}", np.nanmean(res[f"{codes[-1]}|global|"]))
        num(f"rowsShortDrop{A}", np.nanmean(res[f"{codes[-1]}|global|"]) - np.nanmean(worst))


if __name__ == "__main__":
    print("figures:")
    for f in (fig_code, fig_timing, fig_real, fig_rig, fig_demo, fig_rows):
        f()
    (HERE / "numbers.tex").write_text(
        "% written by make_figures.py: do not edit by hand\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(numbers.items())))
    print(f"  numbers.tex ({len(numbers)} numbers)")
    if missing:
        print("\nskipped, inputs missing (see the docstring for how to make them):\n  " + "\n  ".join(missing))
        sys.exit(1)
