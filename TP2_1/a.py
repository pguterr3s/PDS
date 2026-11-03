import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

FICHEIRO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ECG_with_interf_26_27.wav")


def carregar_ecg():
    fs, x = wavfile.read(FICHEIRO)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float64) / 32768.0
    return fs, x


def espetro(x, fs):
    X = np.abs(np.fft.rfft(x - np.mean(x))) / len(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return f, X


def detetar_interferencia(x, fs):
    f, X = espetro(x, fs)
    return f[np.argmax(X)]


def projetar_notch(f0, fs, rp):
    W0 = 2 * np.pi * f0 / fs
    b = np.array([1.0, -2 * np.cos(W0), 1.0])
    a = np.array([1.0, -2 * rp * np.cos(W0), rp ** 2])
    return b, a


if __name__ == "__main__":
    fs, x = carregar_ecg()
    f0 = detetar_interferencia(x, fs)

    print(f"Fs = {fs} Hz | duracao = {len(x) / fs:.1f} s | {len(x)} amostras")
    print(f"Frequencia do sinal interferente: f0 = {f0:.2f} Hz")
    print(f"W0 = 2*pi*f0/fs = {2 * np.pi * f0 / fs:.4f} rad/amostra (= pi/{fs / (2 * f0):.0f})")

    f, X = espetro(x, fs)
    XdB = 20 * np.log10(X + 1e-12)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    ax1.plot(f, XdB, lw=0.6)
    ax1.axvline(f0, color="r", ls="--", lw=1, label=f"f0 = {f0:.0f} Hz")
    ax1.set_title("Espetro do ECG ruidoso (0 - fs/2)")
    ax1.set_xlabel("Frequencia (Hz)")
    ax1.set_ylabel("|X(f)| (dB)")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(f, XdB, lw=0.6)
    ax2.axvline(f0, color="r", ls="--", lw=1, label=f"f0 = {f0:.0f} Hz")
    ax2.set_xlim(0, 3000)
    ax2.set_title("Zoom 0 - 3000 Hz (ECG concentrado nas baixas frequencias)")
    ax2.set_xlabel("Frequencia (Hz)")
    ax2.set_ylabel("|X(f)| (dB)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(FICHEIRO), "a_espetro_original.png"), dpi=150)
    plt.show()