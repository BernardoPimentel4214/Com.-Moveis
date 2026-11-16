"""Diagramas receptores e ensaio Rayleigh para 1 a 4 ramos lineares.

theta é medido a partir do eixo z do arranjo, com posições z_l = l*d.
O diagrama de abertura circular uniforme aproxima um refletor parabólico:
Balanis, Antenna Theory, 3ª ed., seção 12.6, p. 683, Tabela 12.2.
O refletor parabólico é discutido na seção 15.4, p. 893. A aproximação
não inclui alimentação, bloqueio, perdas ou polarização.
"""

import numpy as np
from scipy.special import j1


def ganho_dipolo(theta):
    return 1.5 * np.sin(theta)**2


def ganho_parabola(theta, diametro_lambda=3.0):
    def potencia(t):
        # Ângulo em relação ao eixo transversal: sin(pi/2 - theta) = cos(theta).
        x = np.pi * diametro_lambda * np.cos(t)
        amplitude = np.ones_like(x, dtype=float)
        np.divide(2 * j1(x), x, out=amplitude, where=np.abs(x) > 1e-10)
        return np.where((t >= 0) & (t <= np.pi), amplitude**2, 0.0)

    grade = np.linspace(0, np.pi, 4097)
    media = np.trapezoid(potencia(grade) * np.sin(grade), grade) / 2
    return potencia(np.asarray(theta)) / media


def sortear_angulos(rng, n_ramos, n_amostras, espalhamento_graus=0.0):
    """AoA (ramos, amostras); uma direção comum se espalhamento=0."""
    centro = np.arccos(rng.uniform(-1, 1, n_amostras))
    if espalhamento_graus == 0:
        return np.broadcast_to(centro, (n_ramos, n_amostras)).copy()
    desvio = rng.normal(0, np.deg2rad(espalhamento_graus),
                        (n_ramos, n_amostras))
    return np.arccos(np.cos(centro + desvio))


def receber_rayleigh_antenas(rng, simbolos, gamma_s, n_ramos,
                             diagrama='dipolo', diametro_lambda=3.0,
                             espacamento_lambda=0.5, espalhamento_graus=360,
                             E_s=1.0):
    simbolos = np.asarray(simbolos, dtype=complex)
    gamma_s = np.asarray(gamma_s, dtype=float)
    theta = sortear_angulos(rng, n_ramos, simbolos.size, espalhamento_graus)

    if diagrama == 'dipolo':
        Gr = ganho_dipolo(theta)
    elif diagrama == 'parabólica':
        Gr = ganho_parabola(theta, diametro_lambda)

    shape = (n_ramos, gamma_s.size, simbolos.size)
    h = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    fase = np.exp(-2j * np.pi * espacamento_lambda
                  * np.arange(n_ramos)[:, None] * np.cos(theta))
    h_efetivo = np.sqrt(Gr)[:, None, :] * h * fase[:, None, :]

    sigma = np.sqrt(E_s / (2 * gamma_s))[None, :, None]
    ruido = sigma * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    r = h_efetivo * simbolos[None, None, :] + ruido

    return r, np.abs(h_efetivo), np.angle(h_efetivo), Gr, theta


def snr_combinadores(beta, gamma_s):
    """SNRs para 1 até n ramos, chaves SC/MRC/EGC."""
    beta = np.asarray(beta)
    media = np.asarray(gamma_s)[None, :, None]
    return {
        'SC': media * np.maximum.accumulate(beta**2, axis=0),
        'MRC': media * np.cumsum(beta**2, axis=0),
        'EGC': media * np.cumsum(beta, axis=0)**2
               / np.arange(1, beta.shape[0] + 1)[:, None, None],}
