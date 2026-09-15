import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Ler o ficheiro
fs, audio = wavfile.read("RuiVelosoPortoSentido.wav")

# Número de amostras
N = len(audio)
# FFT
X = np.fft.fft(audio)
# Eixo das frequências
f = np.fft.fftfreq(N, 1/fs)

# Considerar apenas frequências positivas
mask = f >= 0

# Plot
plt.plot(f[mask], np.abs(X[mask]))
plt.xlabel("Frequência [Hz]")
plt.ylabel("Magnitude")
plt.title("Espectro do sinal de áudio")
plt.grid()
plt.show()