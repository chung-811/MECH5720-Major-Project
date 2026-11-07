"""MECH5720 Group 10 -- flashing code analysis.
Code from Raskar, Agrawal & Tumblin, "Coded Exposure Photography:
Motion Deblurring using Fluttered Shutter", SIGGRAPH 2006.
Reproduces their noise-amplification figures and adds an LCD-lag model."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

RASKAR = np.array([int(c) for c in "1010000111000001010000110011110111010111001001100111"])
BOX = np.ones(52)

def smear_matrix(code, n=300):
    """(n+k-1) x n matrix: each unknown pixel smeared by the normalised code."""
    c = code / code.sum(); k = len(c)
    A = np.zeros((n + k - 1, n))
    for j in range(n): A[j:j + k, j] = c
    return A

def noise_amp_db(code, n=300):
    """Paper's metric: max of (A^T A)^-1, in dB (sigma^2 = 1)."""
    A = smear_matrix(code, n)
    return 10 * np.log10(np.linalg.inv(A.T @ A).max())

def lcd_kernel(code, refresh_hz=60, tau_ms=4.0, dt=0.05):
    """Code as the monitor actually emits it: first-order lag of time constant
    tau_ms, then averaged back to one sample per chop. The monitor we measured
    on the rig is asymmetric (about 44 ms to brighten, dims at once); that
    model lives in software/deconvolution/real.py (MONITOR)."""
    chop = 1000 / refresh_hz; L = int(round(chop / dt)); a = min(dt / tau_ms, 1.0)
    s = np.r_[np.repeat(code, L), np.zeros(int(6 * tau_ms / dt))].astype(float)
    y = np.zeros_like(s)
    for i in range(1, len(s)): y[i] = y[i - 1] + a * (s[i] - y[i - 1])
    y = np.r_[y, np.zeros((-len(y)) % L)]
    return y.reshape(-1, L).mean(1)

def spectrum_db(code, pad=300):
    F = np.abs(np.fft.rfft(code / code.sum(), len(code) + pad))
    return 20 * np.log10(np.maximum(F, 1e-12))

if __name__ == "__main__":
    lag = lcd_kernel(RASKAR, tau_ms=4)
    rows = [("Box (constant light)", BOX), ("Raskar code, ideal", RASKAR),
            ("Raskar code, 60 Hz LCD, 4 ms lag", lag)]
    for name, c in rows: print(f"{name:34s} noise amplification {noise_amp_db(c):6.2f} dB")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    t = np.arange(52)
    ax[0].step(t, RASKAR + 1.2, where="post", label="Code (ideal)")
    ax[0].plot(np.arange(len(lag)) + 0.5, lag / lag.max(), ".-", ms=3, label="LCD, 4 ms lag")
    ax[0].set(xlabel="Chop (1 monitor frame = 16.7 ms)", yticks=[], title="Flashing sequence")
    ax[0].legend(fontsize=8, loc="upper right")
    for name, c in rows:
        f = np.linspace(0, np.pi, len(spectrum_db(c)))
        ax[1].plot(f, spectrum_db(c), label=f"{name} ({noise_amp_db(c):.1f} dB)")
    ax[1].set(ylim=(-60, 2), xlabel="Spatial frequency (rad/pixel)", ylabel="Magnitude (dB)",
              title="Blur frequency response (zero-padded to 352)")
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig("code_spectrum.png", dpi=160)
