import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import exp1, i0e

'''
    Nessa atividade, é simulada a capacidade ergódica em função da SNR média.
    São comparados os canais AWGN, Rayleigh e Rice, considerando CSI perfeita no receptor.
    A capacidade é normalizada pela largura de banda: C_erg/B_w = E[log2(1 + γ)].
    Para isso, são geradas amostras da envoltória com E[β²] = 1, de modo que γ = γ_média * β².
'''

# Número de amostras
N = 200_000

# Valores da SNR média (dB)
gamma_db = np.linspace(-10, 30, 201)
gamma_media = 10**(gamma_db / 10)

# Pontos da simulação
gamma_db_amostras = np.linspace(-10, 30, 21)
gamma_media_amostras = 10**(gamma_db_amostras / 10)

plt.figure(figsize=(9, 5.5))

'''
    Curvas exatas
'''
# Capacidade normalizada do canal AWGN (limitante)
C_AWGN = np.log2(1 + gamma_media)
plt.plot(gamma_db, C_AWGN, 'k--', label='AWGN (limitante)')

# K_r em escala linear, como na figura dos slides
for K_r, cor in zip([0, 5, 10, 30], ['c', 'g', 'b', 'r']):
    if K_r == 0:
        # Capacidade ergódica Rayleigh - Eq. (63)
        C_exata = np.exp(1 / gamma_media) * exp1(1 / gamma_media) / np.log(2)
        nome = 'Rayleigh (K = 0)'
    else:
        # FDP do ganho x = β² para Rice, com E[β²] = 1
        # i0e(z) = exp(-z) * I_0(z), evitando overflow no produto da Eq. (65)
        def fdp_ganho(x):
            z = 2 * np.sqrt(K_r * (K_r + 1) * x)
            return ((K_r + 1)
                    * np.exp(-(np.sqrt((K_r + 1) * x) - np.sqrt(K_r))**2)
                    * i0e(z))

        # Capacidade ergódica Rice - integral da Eq. (65), com γ = γ_média * x
        C_exata = []
        for media in gamma_media:
            capacidade, _ = quad(lambda x: np.log2(1 + media * x) * fdp_ganho(x),
                                 0, np.inf)
            C_exata.append(capacidade)
        C_exata = np.array(C_exata)
        nome = f'Rice (K = {K_r})'

    '''
        Curvas geradas com amostras
    '''
    # Distribuição da envoltória do canal com E[β²] = 1
    # Para K_r = 0, a distribuição Rice se reduz à Rayleigh
    mu = np.sqrt(K_r / (K_r + 1))
    sigma = np.sqrt(1 / (2 * (K_r + 1)))
    beta = np.sqrt(np.random.normal(mu, sigma, N)**2
                   + np.random.normal(0, sigma, N)**2)

    # Check de E[β²] = 1
    print(f'{nome}: E[β²] = {np.mean(beta**2)}')

    C_amostrada = []
    for media in gamma_media_amostras:
        # SNR instantânea e média da capacidade normalizada
        gamma = media * beta**2
        C_amostrada.append(np.mean(np.log2(1 + gamma)))
    C_amostrada = np.array(C_amostrada)

    # Plotar curva exata e pontos simulados sobre a curva
    plt.plot(gamma_db, C_exata, color=cor, label=nome)
    plt.plot(gamma_db_amostras, C_amostrada, 'o', color=cor,
             markersize=4, markerfacecolor='none', zorder=3,
             label='Simulação' if K_r == 0 else None)

plt.xlabel(r'SNR média - $\bar\gamma$ (dB)')
plt.ylabel('Capacidade normalizada (bits/s/Hz)')
plt.title('Capacidade Ergódica (CSI-R)')
plt.xlim(-10, 30)
plt.ylim(0, 10.5)
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
