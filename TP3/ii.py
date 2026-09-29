import numpy as np
import matplotlib.pyplot as plt
from i import projetar_lpf_ideal, FS, FC, N1, N2, N3

NFFT = 8192 


def resposta_em_frequencia(h, fs, nfft=NFFT):
    """FFT (com zero-padding) da resposta ao impulso -> resposta em frequencia."""
    H = np.fft.rfft(h, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    mag = np.abs(H)
    return freqs, mag


if __name__ == "__main__":
    h1 = projetar_lpf_ideal(N1, FC, FS)
    h2 = projetar_lpf_ideal(N2, FC, FS)
    h3 = projetar_lpf_ideal(N3, FC, FS)

    plt.figure(figsize=(10, 6))
    for h, label in [(h1, f"N1 = {N1}"), (h2, f"N2 = {N2}"), (h3, f"N3 = {N3}")]:
        freqs, mag = resposta_em_frequencia(h, FS)
        plt.plot(freqs, mag, label=label)

    plt.axvline(FC, color="gray", linestyle="--", label=f"fc = {FC} Hz")
    plt.xlabel("Frequência [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.title(f"Resposta em frequência do LPF (Fs={FS} Hz, fc={FC} Hz)")
    plt.xlim(0, FS / 2)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("resposta_frequencia.png", dpi=150)
    print("Grafico guardado em resposta_frequencia.png")
    plt.show()