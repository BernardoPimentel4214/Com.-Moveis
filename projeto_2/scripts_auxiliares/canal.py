import numpy as np
import matplotlib.pyplot as plt


def init_constelacao_M_QAM(M, E_s):
    # Posições da constelação QAM simétrica nos dois eixos
    posicoes_QAM = np.arange(-(np.sqrt(M) - 1), np.sqrt(M), 2)

    # Normaliza para energia E_s = 1
    I, Q = np.meshgrid(posicoes_QAM, posicoes_QAM)
    E = I**2 + Q**2
    posicoes_QAM /= np.sqrt(np.mean(E) / E_s)

    # Pontos para o critério MAP:
    pontos_QAM = np.array([(i, q)
                           for q in posicoes_QAM[::-1]
                           for i in posicoes_QAM])

    return pontos_QAM


def init_r_beta_phi_l(l):
    r_l = []
    beta_l = []
    phi_l = []
    for ramo in range(l + 1): r_l.append([]), beta_l.append([]), phi_l.append([])
    return r_l, beta_l, phi_l


def Tx_simbolos(rng, desvanecimento, s_i, SNR, E_s, N_t, K_r=5):
    if desvanecimento == 'Rice' and K_r < 0:
        raise ValueError('K_r deve ser não negativo (escala linear).')
    r_l = []
    beta_l = []
    phi_l = []
    N_0 = E_s / SNR
    mu = 0
    
    for pot_ruido in N_0:
            # Realização do canal
        if (desvanecimento == 'Rayleigh'):
            # Distribuição Rayleigh da envoltória do canal com E[β²] = 1
            # Uma amostra independente por símbolo, aplicada aos dois eixos
            sigma = np.sqrt(1/2)
            h = (rng.normal(0, sigma, N_t) + 1j*rng.normal(0, sigma, N_t))
            phi = np.angle(h)
            beta = np.abs(h)

        elif desvanecimento == 'Rice':
            # K_r = potência LOS / potência difusa; E[|h|²] = 1.
            sigma = np.sqrt(1 / (2 * (K_r + 1)))
            h = (np.sqrt(K_r / (K_r + 1))
                 + rng.normal(0, sigma, N_t)
                 + 1j * rng.normal(0, sigma, N_t))
            phi = np.angle(h)
            beta = np.abs(h)
        else:
            raise ValueError(f'Desvanecimento desconhecido: {desvanecimento}')

        # Vetor de ruído
        s_complex = s_i[:, 0] + 1j * s_i[:, 1]

        n = rng.normal(mu, np.sqrt(pot_ruido / 2), size=(N_t, 2))
        n_complex = n[:, 0] + 1j * n[:, 1]

        # Sinal recebido, corrompido por ruído (mesmo caso da atividade 3):
        r_complex = h * s_complex + n_complex
        r_N_0_fixo = np.stack((r_complex.real, r_complex.imag), axis=-1)

        r_l.append(r_N_0_fixo)
        beta_l.append(beta)
        phi_l.append(phi)

    return np.array(r_l), np.array(beta_l), np.array(phi_l)


def SEP_QAM_Rayleigh(gamma_s, M, K, tecnica):
    nos, pesos = np.polynomial.legendre.leggauss(100)
    a = 1 - 1 / np.sqrt(M)

    def integral(limite):
        theta = (nos + 1) * limite / 2
        t = 3 / (2 * (M - 1) * np.sin(theta)**2)
        z = gamma_s[:, None] * t[None, :]
        if tecnica == 'MRC':
            mgf = (1 + z)**(-K)
        else:
            mgf = np.ones_like(z)
            for k in range(1, K + 1):
                mgf *= k / (k + z)
        return limite / 2 * np.sum(mgf * pesos, axis=1)

    return 4 * a / np.pi * integral(np.pi / 2) - 4 * a**2 / np.pi * integral(np.pi / 4)
