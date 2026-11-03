import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

class FIRFilter:
    def __init__(self, coefficients):
        self.h = list(coefficients)
        self.N = len(coefficients)
        self.buffer = [0.0] * self.N 

    def process_sample(self, x_n) -> float:
        self.buffer = [x_n] + self.buffer[:-1]
        y_n = 0.0
        for k in range(self.N): 
            y_n += self.buffer[k] * self.h[k]
        return y_n

    def filter_signal(self, signal):
        return [self.process_sample(x) for x in signal]

def analisar_sinal(tempo, sinal_in, sinal_out, fs, titulo_janela):
    
    
   
    N = len(sinal_in)
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    fft_in = np.abs(np.fft.rfft(sinal_in)) / N
    fft_out = np.abs(np.fft.rfft(sinal_out)) / N

    plt.figure(figsize=(12, 8))
    plt.suptitle(titulo_janela, fontsize=14, fontweight='bold')

    
    inicio = 0
    fim = 60 * fs 
    
   
    if fim > N:
        fim = N

    plt.subplot(2, 1, 1)
   
    plt.plot(tempo[inicio:fim], sinal_in[inicio:fim], label="Entrada (Original)", alpha=0.7)
    plt.plot(tempo[inicio:fim], sinal_out[inicio:fim], label="Saída (Filtrado)", linewidth=2)
    plt.title("Domínio do Tempo (1 Minuto)")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True)

    
    plt.subplot(2, 1, 2)
    plt.plot(freqs[1:], fft_in[1:], label="Espectro de Entrada", alpha=0.7)
    plt.plot(freqs[1:], fft_out[1:], label="Espectro de Saída", linewidth=2)
    plt.title("Domínio da Frequência (FFT)")
    plt.xlabel("Frequência [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def teste_ruido_branco(fir, N_taps):
    print("A processar Ruído Branco...")
    fs = 44100
    duracao = 3.0 
    tempo = np.linspace(0, duracao, int(fs * duracao), endpoint=False)
    
    ruido_in = np.random.randn(len(tempo))
    
    
    ruido_in = ruido_in / np.max(np.abs(ruido_in))
    
    
    ruido_out = fir.filter_signal(ruido_in)
    
 
    ruido_out_array = np.array(ruido_out)
    if np.max(np.abs(ruido_out_array)) > 0:
        ruido_out_array = ruido_out_array / np.max(np.abs(ruido_out_array))
    
    
    wavfile.write("1_ruido_original.wav", fs, ruido_in.astype(np.float32))
    wavfile.write("2_ruido_filtrado.wav", fs, ruido_out_array.astype(np.float32))
    print("-> Áudios '1_ruido_original.wav' e '2_ruido_filtrado.wav' guardados na pasta!")
    
    analisar_sinal(tempo, ruido_in, ruido_out, fs, "Teste 1: Ruído Branco")

def teste_musica(fir):
    print("A processar Música...")
    nome_ficheiro = "musica.wav" 
    
    try:
        fs, musica_in = wavfile.read(nome_ficheiro)
        
        
        if len(musica_in.shape) > 1:
            musica_in = musica_in.mean(axis=1)
            
        
        musica_in = musica_in / np.max(np.abs(musica_in))
        
        tempo = np.linspace(0, len(musica_in)/fs, len(musica_in), endpoint=False)
        
        musica_out = fir.filter_signal(musica_in)
        
        wavfile.write("musica_filtrada.wav", fs, np.array(musica_out, dtype=np.float32))
        
        analisar_sinal(tempo, musica_in, musica_out, fs, "Teste 2: Sinal de Áudio (Música)")
        
    except FileNotFoundError:
        print(f"ERRO: Ficheiro '{nome_ficheiro}' não encontrado. Adicione um ficheiro .wav na diretoria.")

if __name__ == "__main__":
    h_n = [-0.045, 0, 0.075, 0.1592, 0.2251, 0.250, 0.2251, 0.1592, 0.075, 0, -0.045]
    
    meu_fir = FIRFilter(h_n)
    
    teste_ruido_branco(meu_fir, len(h_n))
    
    meu_fir.buffer = [0.0] * len(h_n)
    
    teste_musica(meu_fir)