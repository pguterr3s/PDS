import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from i import projetar_lpf_ideal, FIRFilter, FC, N1, N3

MUSICA = "RuiVelosoPortoSentido.wav"


DURACAO_SEGUNDOS = 30


def analisar_espetro(sinal, fs):
    N = len(sinal)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    mag = np.abs(np.fft.rfft(sinal)) / N
    return freqs, mag


if __name__ == "__main__":
    fs, musica = wavfile.read(MUSICA)

    if musica.ndim > 1:
        musica = musica.mean(axis=1)

    musica = musica.astype(float)
    musica = musica / np.max(np.abs(musica))

    if DURACAO_SEGUNDOS is not None:
        musica = musica[: int(DURACAO_SEGUNDOS * fs)]

    print(f"Fs (do ficheiro) = {fs} Hz | duracao usada = {len(musica)/fs:.1f} s")

    h1 = projetar_lpf_ideal(N1, FC, fs)
    h3 = projetar_lpf_ideal(N3, FC, fs)

    fir1 = FIRFilter(h1)
    fir3 = FIRFilter(h3)

    print(f"A filtrar musica com N1={N1}... (mais lento)")
    saida_n1 = np.array(fir1.filter_signal(musica))

    print(f"A filtrar musica com N3={N3}...")
    saida_n3 = np.array(fir3.filter_signal(musica))

    wavfile.write("musica_filtrada_N1.wav", fs, saida_n1.astype(np.float32))
    wavfile.write("musica_filtrada_N3.wav", fs, saida_n3.astype(np.float32))
    print("Ficheiros guardados: musica_filtrada_N1.wav, musica_filtrada_N3.wav")

    freqs_in, mag_in = analisar_espetro(musica, fs)
    freqs_n1, mag_n1 = analisar_espetro(saida_n1, fs)
    freqs_n3, mag_n3 = analisar_espetro(saida_n3, fs)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs_in[1:], mag_in[1:], label="Original", alpha=0.6)
    plt.plot(freqs_n1[1:], mag_n1[1:],linestyle="--", label=f"Filtrado N1={N1}", linewidth=1.5)
    plt.plot(freqs_n3[1:], mag_n3[1:],linestyle=":", label=f"Filtrado N3={N3}", linewidth=1)
    plt.axvline(FC, color="gray", linestyle="--", label=f"fc = {FC} Hz")
    plt.xlabel("Frequência [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Espetro da música: original vs. filtrada (N1 vs N3)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("espetro_musica_N1_N3.png", dpi=150)
    print("Grafico guardado em espetro_musica_N1_N3.png")
    plt.show()