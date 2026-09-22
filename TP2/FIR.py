class FIRFilter:
    def __init__(self, coefficients):
        self.h = list(coefficients)
        self.N = len(coefficients)

        self.buffer = [0.0] * self.N

    def process_sample(self, x_n):
        self.buffer = [x_n] + self.buffer[:-1]
        y_n = 0.0
        for k in range(self.N):
            y_n += self.buffer[k] * self.h[k]

        return y_n
    
    def filter_signal(self, signal):
        return [self.process_sample(x_n) for x_n in signal]
       
def validar_fir():
    h_teste = [1, 2, 3, 4]

    fir_impulso = FIRFilter(h_teste)
    impulso = [1.0, 0.0, 0.0, 0.0, 0.0]
    saida_impulso = fir_impulso.filter_signal(impulso)
    
    print("--- Teste 1: Resposta ao Impulso ---")
    print(f"Esperado: {h_teste}")
    print(f"Obtido:   {saida_impulso}")
    assert saida_impulso[:len(h_teste)] == h_teste, "Falha no Teste do Impulso!"
    print("-> Teste do Impulso: OK\n")

    # -----------------------------------------------------------
    # Teste 2: Resposta ao Degrau Unitário (Ganho DC)
    # A saída em regime estacionário deve ser a soma de todos os h[k].
    # -----------------------------------------------------------
    fir_degrau = FIRFilter(h_teste)
    degrau = [1.0] * 10
    saida_degrau = fir_degrau.filter_signal(degrau)
    ganho_dc_esperado = sum(h_teste)
    
    print("--- Teste 2: Resposta ao Degrau (Ganho DC) ---")
    print(f"Ganho DC Esperado: {ganho_dc_esperado}")
    print(f"Última saída obtida: {saida_degrau[-1]}")
    assert abs(saida_degrau[-1] - ganho_dc_esperado) < 1e-9, "Falha no Ganho DC!"
    print("-> Teste do Degrau: OK\n")


if __name__ == "__main__":
    validar_fir()