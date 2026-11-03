import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import lfilter
from TP2_1.a import carregar_ecg, detetar_interferencia, projetar_notch, espetro, FICHEIRO

RPS = [0.5, 0.99]
PASTA = os.path.dirname(FICHEIRO)
SEG_OUVIR = 5  # segundos a reproduzir de cada sinal


def guardar_wav(nome, y, fs):
    y16 = np.clip(y * 32768, -32768, 32767).astype(np.int16)
    wavfile.write(os.path.join(PASTA, nome), fs, y16)


def ouvir(nome, y, fs):
    print(f"  > a reproduzir: {nome} ({SEG_OUVIR} s)")
    sd.play(y[: SEG_OUVIR * fs], fs)
    sd.wait()


if __name__ == "__main__":
    fs, x = carregar_ecg()
    f0 = detetar_interferencia(x, fs)
    print(f"f0 = {f0:.0f} Hz, fs = {fs} Hz\n")

    # Filtragem: lfilter implementa a equacao as diferencas de projetar_notch()
    saidas = {}
    for rp in RPS:
        b, a = projetar_notch(f0, fs, rp)
        saidas[rp] = lfilter(b, a, x)
        guardar_wav(f"ECG_filtrado_rp{rp}.wav", saidas[rp], fs)

    # Verificacao numerica: amplitude da componente em f0 antes/depois
    f, X = espetro(x, fs)
    k0 = np.argmin(np.abs(f - f0))
    print(f"Componente em {f0:.0f} Hz:")
    print(f"  original     : {X[k0]:.3e}")
    for rp, y in saidas.items():
        _, Y = espetro(y, fs)
        print(f"  rp = {rp:<5}   : {Y[k0]:.3e}  (atenuacao {20 * np.log10(Y[k0] / X[k0]):.1f} dB)")
    print()

    # ---- Ouvir ----
    ouvir("original (com interferencia)", x, fs)
    for rp, y in saidas.items():
        ouvir(f"filtrado rp = {rp}", y, fs)

    # ---- Espetro antes / depois ----
    fig, axs = plt.subplots(len(RPS) + 1, 1, figsize=(11, 9), sharex=True)
    sinais = [("Original (com interferencia)", x)] + [(f"Filtrado rp = {rp}", y) for rp, y in saidas.items()]
    for ax, (titulo, s) in zip(axs, sinais):
        fs_, S = espetro(s, fs)
        ax.plot(fs_, 20 * np.log10(S + 1e-12), lw=0.6)
        ax.axvline(f0, color="r", ls="--", lw=1)
        ax.set_title(titulo)
        ax.set_ylabel("|X(f)| (dB)")
        ax.set_ylim(-160, 0)
        ax.grid(True)
    axs[-1].set_xlim(0, 3000)
    axs[-1].set_xlabel("Frequencia (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "b2_espetros.png"), dpi=150)

    # ---- Dominio do tempo ----
    t = np.arange(len(x)) / fs
    janela = (t >= 10) & (t < 13)            # 3 s -> varios batimentos
    zoom = (t >= 10) & (t < 10.01)           # 10 ms -> ver a sinusoide de 2 kHz

    fig, axs = plt.subplots(len(RPS) + 1, 2, figsize=(13, 9), sharey="row")
    for i, (titulo, s) in enumerate(sinais):
        axs[i, 0].plot(t[janela], s[janela], lw=0.6)
        axs[i, 0].set_title(f"{titulo} - 3 s")
        axs[i, 1].plot(t[zoom] * 1e3, s[zoom], lw=0.8)
        axs[i, 1].set_title(f"{titulo} - zoom 10 ms")
        for ax in axs[i]:
            ax.grid(True)
        axs[i, 0].set_ylabel("Amplitude")
    axs[-1, 0].set_xlabel("Tempo (s)")
    axs[-1, 1].set_xlabel("Tempo (ms)")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "b2_tempo.png"), dpi=150)

    # Sobreposicao dos dois filtrados para comparar a distorcao do ECG
    fig, ax = plt.subplots(figsize=(11, 4))
    for rp, y in saidas.items():
        ax.plot(t[janela], y[janela], lw=0.8, label=f"rp = {rp}")
    ax.set_title("ECG filtrado: rp = 0.5 vs rp = 0.99")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "b2_comparacao.png"), dpi=150)

    plt.show()