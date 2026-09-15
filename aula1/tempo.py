import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Ler o ficheiro
fs, audio = wavfile.read("RuiVelosoPortoSentido.wav")

print("Frequência de amostragem:", fs, "Hz")
print("Número de amostras:", len(audio))
print("Duração:", len(audio) / fs, "segundos")

# Criar eixo temporal
t = np.arange(len(audio)) / fs

# Plot
plt.plot(t, audio)
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.title("plot tempo")
plt.grid()
plt.show()