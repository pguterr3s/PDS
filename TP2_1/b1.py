import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from TP2_1.a import carregar_ecg, detetar_interferencia, projetar_notch, FICHEIRO

RPS = [0.7, 0.9, 0.99]


def largura_banda_3db(w, H):
   
    mag = np.abs(H) / np.max(np.abs(H))
    abaixo = w[mag < 1 / np.sqrt(2)]
    return abaixo.max() - abaixo.min() if len(abaixo) else 0.0


if __name__ == "__main__":
    fs, x = carregar_ecg()
    f0 = detetar_interferencia(x, fs)
    print(f"f0 = {f0:.0f} Hz, fs = {fs} Hz\n")

    fig, (ax_mag, ax_fase) = plt.subplots(2, 1, figsize=(11, 8))
    fig_zoom, ax_zoom = plt.subplots(figsize=(11, 4.5))

    for rp in RPS:
        b, a = projetar_notch(f0, fs, rp)
        w, H = freqz(b, a, worN=2 ** 16)
        f = w * fs / (2 * np.pi)
        HdB = 20 * np.log10(np.abs(H) + 1e-12)

        bw = largura_banda_3db(w, H)
        ganho_dc = np.abs(np.sum(b) / np.sum(a))
        print(f"rp = {rp:<5}  b = {np.round(b, 4)}  a = {np.round(a, 4)}")
        print(f"           |H(0)| = {ganho_dc:.4f} ({20 * np.log10(ganho_dc):.2f} dB) | "
              f"BW(-3dB) = {bw * fs / (2 * np.pi):.1f} Hz  (aprox. teorica 2(1-rp) -> "
              f"{(1 - rp) * fs / np.pi:.1f} Hz)\n")

        ax_mag.plot(f, HdB, label=f"rp = {rp}")
        ax_fase.plot(f, np.unwrap(np.angle(H)), label=f"rp = {rp}")
        ax_zoom.plot(f, np.abs(H), label=f"rp = {rp}")

    ax_mag.axvline(f0, color="k", ls=":", lw=1)
    ax_mag.set_title("Resposta em frequencia do filtro notch (freqz) - modulo")
    ax_mag.set_xlabel("Frequencia (Hz)")
    ax_mag.set_ylabel("|H| (dB)")
    ax_mag.set_ylim(-60, 10)
    ax_mag.grid(True)
    ax_mag.legend()

    ax_fase.axvline(f0, color="k", ls=":", lw=1)
    ax_fase.set_title("Fase")
    ax_fase.set_xlabel("Frequencia (Hz)")
    ax_fase.set_ylabel("Fase (rad)")
    ax_fase.grid(True)
    ax_fase.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(FICHEIRO), "b1_freqz.png"), dpi=150)

    ax_zoom.axvline(f0, color="k", ls=":", lw=1)
    ax_zoom.axhline(1 / np.sqrt(2), color="gray", ls="--", lw=1, label="-3 dB")
    ax_zoom.set_xlim(0, 2 * f0)
    ax_zoom.set_title("Modulo linear (0 - 2*f0): quanto maior rp, mais estreito o notch")
    ax_zoom.set_xlabel("Frequencia (Hz)")
    ax_zoom.set_ylabel("|H|")
    ax_zoom.grid(True)
    ax_zoom.legend()
    fig_zoom.tight_layout()
    fig_zoom.savefig(os.path.join(os.path.dirname(FICHEIRO), "b1_freqz_zoom.png"), dpi=150)

    plt.show()