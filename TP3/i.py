import numpy as np

# Dados do enunciado
FS = 44100
FC = 2000
N1, N2, N3 = 255, 33, 11


def projetar_lpf_ideal(N, fc, fs):
    """
    Formula do quadro (Wc = frequencia de corte angular digital, rad/amostra):
        h[n] = (Wc/pi) * sinc(Wc*n/pi)          -- infinita, centrada em n=0

    Substituindo Wc = 2*pi*fc/fs e atrasando por
    M=(N-1)/2 amostras para caber em indices 0..N-1:
        h[n] = 2*(fc/fs) * sinc(2*(fc/fs)*(n - M)),  n = 0..N-1
    """
    M = (N - 1) / 2
    n = np.arange(N)
    h = 2 * (fc / fs) * np.sinc(2 * (fc / fs) * (n - M))
    return h


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


if __name__ == "__main__":
    h1 = projetar_lpf_ideal(N1, FC, FS)
    h2 = projetar_lpf_ideal(N2, FC, FS)
    h3 = projetar_lpf_ideal(N3, FC, FS)

    fir1 = FIRFilter(h1)
    fir2 = FIRFilter(h2)
    fir3 = FIRFilter(h3)

    print(f"Fs = {FS} Hz, fc = {FC} Hz\n")
    for nome, h in [("N1 = 255", h1), ("N2 = 33", h2), ("N3 = 11", h3)]:
        print(f"{nome}: {len(h)} coeficientes | ganho DC (soma de h) = {np.sum(h):.4f}")