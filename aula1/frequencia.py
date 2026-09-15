import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Ler o ficheiro
fs, audio = wavfile.read("RuiVelosoPortoSentido.wav")
print(fs)

# Se o áudio for estéreo (2 canais), escolhemos apenas o primeiro canal (esquerdo)
if len(audio.shape) > 1:
    audio = audio[:, 0]

audio = audio - np.mean(audio)

# Número de amostras
N = len(audio)

# FFT (agora sim, ao longo do tempo)
X = np.fft.fft(audio)

# Criar um eixo de frequências a ir de 0 até Fs
f = np.arange(N) * (fs / N)

# Plot
plt.plot(f, np.abs(X))
plt.xlim(0, fs)
plt.xlabel("Frequência [Hz]")
plt.ylabel("Amplitude")
plt.title("Espectro do sinal de áudio (0 até Fs)")
plt.grid()
plt.show()