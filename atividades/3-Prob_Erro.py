
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

'''
    Nessa atividade, é simulada a taxa de erro de símbolo em função da SNR por símbolo γ_s = E_s/N_0.
    Para isso, a energia média de símbolo E_s é fixada e varia-se N_0. A constelação analisada é a M-QAM.

    - Symbol Error Probability - SEP: expressão matemática derivada da análise probabilística
    - Symbol Error Rate - SER: resultado da simulação de Monte Carlo associada à SEP
'''

# N_t >> 1/SEP
N_t = 10_000
E_s = 1

# Valores da SNR por símbolo (dB)
gamma_s_db = np.linspace(-5, 40, 30)
gamma_s = 10**(gamma_s_db / 10)

plt.figure(figsize=(9, 5.5))

# Será simulado o desempenho para 4, 16, 64 e 256-QAM
for m in [4, 16, 64, 256]:
    # Posições da constelação QAM simétrica nos dois eixos
    posicoes_QAM = np.arange(-(np.sqrt(m) - 1), np.sqrt(m), 2)

    # Normaliza para energia E_s = 1
    I, Q = np.meshgrid(posicoes_QAM, posicoes_QAM)
    E = I**2 + Q**2
    posicoes_QAM /= np.sqrt(np.mean(E) / E_s)

    s_i = np.array([[np.random.choice(posicoes_QAM),
                     np.random.choice(posicoes_QAM)]
                     for i in range(N_t)])

    # Pontos para o critério MAP:
    pontos_QAM = np.array([(i, q)
                           for q in posicoes_QAM[::-1]
                           for i in posicoes_QAM])

    # Ruído aditivo:
    N_0 = E_s / gamma_s
    mu = 0

    SER = []

    for pot_ruido in N_0:

        # Vetor de ruído
        n = np.random.normal(mu, np.sqrt(pot_ruido / 2),
                             size=(N_t, 2))

        # Sinal recebido, corrompido por ruído:
        r = s_i + n

        # Critério MAP (símbolos equiprováveis):
        distancias = np.sum(
            (r[:, None, :] - pontos_QAM[None, :, :])**2,
            axis=2
        )

        indices_MAP = np.argmin(distancias, axis=1)
        s_estimado = pontos_QAM[indices_MAP]

        # Taxa de erro de símbolo (SER):
        N_e = np.count_nonzero(np.any(s_estimado != s_i, axis=1))
        SER.append(N_e / N_t)

    SER = np.array(SER)

    # Probabilidade de erro de símbolo (SEP) - expressão exata:
    argumento = np.sqrt(3 * gamma_s / (m - 1))
    Q_func = 0.5 * erfc(argumento / np.sqrt(2))

    SEP = (4 * (1 - 1 / np.sqrt(m)) * Q_func
           - 4 * (1 - 1 / np.sqrt(m))**2 * Q_func**2)

    # Plotar SEP e SER:
    linha, = plt.semilogy(gamma_s_db - 10 * np.log10( np.log2(m) ), SEP, label=f'SEP - {m}-QAM')
    if (m==256):
        plt.semilogy(gamma_s_db, np.where(SER > 0, SER, np.nan),
                    'x', color='k',
                    markersize=5, label=f'SER')
    else:
        plt.semilogy(gamma_s_db, np.where(SER > 0, SER, np.nan),
                    'x', color='k', markersize=5)        

plt.xlabel(r'SNR por símbolo - $\gamma_s$ (dB)')
plt.ylabel('Probabilidade de erro')
plt.title('SEP e SER - M-QAM')
plt.ylim(1e-5, 1)
plt.xlim(-5, 30)
plt.grid(True, which='both', alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
