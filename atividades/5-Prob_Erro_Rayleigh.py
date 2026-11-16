import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

'''
    Nessa atividade, é simulada a taxa de erro de símbolo em função da SNR média por símbolo.
    A constelação analisada é a 16-QAM, comparando os canais AWGN e Rayleigh.
    Para isso, a energia média de símbolo E_s é fixada e varia-se N_0.
    É considerada demodulação coerente com CSI perfeita: o receptor conhece a envoltória β.

    - Symbol Error Probability - SEP: expressão matemática derivada da análise probabilística
    - Symbol Error Rate - SER: resultado da simulação de Monte Carlo associada à SEP
'''

# N_t >> 1/SEP
N_t = 10_000
E_s = 1
m = 16

# Valores da SNR média por símbolo (dB)
gamma_s_db = np.linspace(-5, 30, 25)
gamma_s = 10**(gamma_s_db / 10)

# Posições da constelação QAM simétrica nos dois eixos
posicoes_QAM = np.arange(-(np.sqrt(m) - 1), np.sqrt(m), 2)

# Normaliza para energia E_s = 1
I, Q = np.meshgrid(posicoes_QAM, posicoes_QAM)
E = I**2 + Q**2
posicoes_QAM /= np.sqrt(np.mean(E) / E_s)

s_i = np.random.choice(posicoes_QAM, size=(N_t, 2))

# Pontos para o critério MAP:
pontos_QAM = np.array([(i, q)
                       for q in posicoes_QAM[::-1]
                       for i in posicoes_QAM])

# Ruído aditivo:
N_0 = E_s / gamma_s
mu = 0

SER_AWGN = []
SER_Rayleigh = []

for pot_ruido in N_0:
    # Vetor de ruído
    n = np.random.normal(mu, np.sqrt(pot_ruido / 2),
                         size=(N_t, 2))

    # Distribuição Rayleigh da envoltória do canal com E[β²] = 1
    # Uma amostra independente por símbolo, aplicada aos dois eixos
    sigma = np.sqrt(1/2)
    beta_Rayleigh = np.sqrt(np.random.normal(0, sigma, N_t)**2
                           + np.random.normal(0, sigma, N_t)**2)

    # Sinal recebido, corrompido por ruído (mesmo caso da atividade 3):
    r = s_i + n

    # Critério MAP (símbolos equiprováveis):
    distancias = np.sum(
        (r[:, None, :] - pontos_QAM[None, :, :])**2,
        axis=2
    )

    indices_MAP = np.argmin(distancias, axis=1)
    s_estimado = pontos_QAM[indices_MAP]

    # Taxa de erro de símbolo (SER) - AWGN:
    N_e = np.count_nonzero(np.any(s_estimado != s_i, axis=1))
    SER_AWGN.append(N_e / N_t)

    # Sinal recebido com desvanecimento Rayleigh e ruído:
    r = beta_Rayleigh[:, None] * s_i + n

    # Critério MAP com CSI perfeita: aplica β na constelação de referência
    distancias = np.sum(
        (r[:, None, :] - beta_Rayleigh[:, None, None]
         * pontos_QAM[None, :, :])**2,
        axis=2
    )

    indices_MAP = np.argmin(distancias, axis=1)
    s_estimado = pontos_QAM[indices_MAP]

    # Taxa de erro de símbolo (SER) - Rayleigh:
    N_e = np.count_nonzero(np.any(s_estimado != s_i, axis=1))
    SER_Rayleigh.append(N_e / N_t)

SER_AWGN = np.array(SER_AWGN)
SER_Rayleigh = np.array(SER_Rayleigh)

# Probabilidade de erro de símbolo (SEP) - AWGN, expressão exata da atividade 3:
argumento = np.sqrt(3 * gamma_s / (m - 1))
Q_func = 0.5 * erfc(argumento / np.sqrt(2))

SEP_AWGN = (4 * (1 - 1 / np.sqrt(m)) * Q_func
            - 4 * (1 - 1 / np.sqrt(m))**2 * Q_func**2)

# Probabilidade de erro de símbolo média (SEP) - Rayleigh, Eq. (53) do PDF:
C_m = np.sqrt(1.5 * gamma_s / (m - 1 + 1.5 * gamma_s))
SEP_Rayleigh = (2 * (1 - 1 / np.sqrt(m)) * (1 - C_m)
                - (1 - 1 / np.sqrt(m))**2
                * (1 - 4 / np.pi * C_m * np.arctan(1 / C_m)))

# Plotar SEP e SER:
plt.figure(figsize=(9, 5.5))
plt.semilogy(gamma_s_db, SEP_AWGN, 'b-', label='SEP (AWGN) - Teórica')
plt.semilogy(gamma_s_db, SEP_Rayleigh, 'b--', label='SEP (Rayleigh) - Teórica (Exata)')
plt.semilogy(gamma_s_db, np.where(SER_AWGN > 0, SER_AWGN, np.nan),
             'k+', markersize=5, label='Simulação (AWGN)')
plt.semilogy(gamma_s_db, np.where(SER_Rayleigh > 0, SER_Rayleigh, np.nan),
             'ko', markersize=5, markerfacecolor='none', label='Simulação (Rayleigh)')

plt.xlabel(r'SNR média por símbolo - $\bar\gamma_s$ (dB)')
plt.ylabel('Probabilidade de erro')
plt.title(f'SEP e SER - {m}-QAM')
plt.ylim(1e-5, 1)
plt.xlim(-5, 30)
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
