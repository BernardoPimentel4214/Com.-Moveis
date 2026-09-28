import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

'''
    Dada a geometria do problema, podemos determinar a curva de probabilidade de visada direta (LoS),
    assim como os demais parâmetros de larga escala. Para o cenário UMi - Street Canyon

        ◦Pr_LoS = {1 ,                               d ≤ 18m
                  (18/d) + exp(-d/36)(1 - 18/d),    18m < d ,

        ◦ Espalhamento de atraso - σ_τ                          - esp_atraso
        ◦ Fator de Rice - K_R (Nulo caso o enlace seja NLoS)    - K_R
        ◦ Espalhamento azimutal de saída - σ_ϕ;AoD              - esp_az_saida
        ◦ Espalhamento azimutal de chegada - σ_ϕ;AoA            - esp_az_chegada
        ◦ Espalhamento em elevação de saída - σ_θ;AoD           - esp_el_saida
        ◦ Espalhamento em elevação de chegada - σ_θ;AoA         - esp_el_chegada.
    
    Excetuando-se Pr_LoS, esses parâmetros são dados por VAs Gaussianas em escala logarítmica (B ou dB quando especificado).
    Sabendo a média e o desvio padrão delas, podemos obter amostras com valores em escala linear após conversão.
'''

def init_est_larga_escala(f_GHz, UE, BS, d, indoor, rng):
    # Probabilidade de visada direta:
    Pr_LoS = lambda d: (1 if d <= 18
                        else (18/d) + np.exp(-d/36) * (1 - 18/d))

    Pr_LoS_0 = Pr_LoS(d)

    # Distribuição de Bernoulli com chance Pr_Los_0% de ser 1 e (1 - Pr_Los_0)% de ser 0:
    LoS = rng.binomial(n=1, p=Pr_LoS_0, size=1)

    if(LoS):
        # Espalhamento de atraso - σ_τ
        mu_esp_atraso  = -0.24 * np.log10(1 + f_GHz) - 7.14
        sig_exp_atraso = 0.38

        # Fator Rice (dB)
        mu_K_R  = 9
        sig_K_R = 5

        # Espalhamento azimutal de saída - σ_ϕ;AoD
        mu_esp_az_saida  = -0.05 * np.log10(1 + f_GHz) + 1.21
        sig_esp_az_saida = 0.41

        # Espalhamento azimutal de chegada - σ_ϕ;AoA
        mu_esp_az_chegada  = -0.08 * np.log10(1 + f_GHz) + 1.73
        sig_esp_az_chegada = 0.014 * np.log10(1 + f_GHz) + 0.28

        # Espalhamento em elevação de saída - σ_θ;AoD
        mu_esp_el_saida  = np.max([-0.21, -14.8 * (d/1000) + 0.01 * np.abs(UE[2] - BS[2]) + 0.83])
        sig_esp_el_saida = 0.35

        # Espalhamento em elevação de chegada - σ_θ;AoA
        mu_esp_el_chegada  = -0.1 * np.log10(1 + f_GHz) + 0.73
        sig_esp_el_chegada = -0.04 * np.log10(1 + f_GHz) + 0.34
    else:
        # Espalhamento de atraso - σ_τ
        mu_esp_atraso  = -0.24 * np.log10(1 + f_GHz) - 6.83
        sig_exp_atraso = 0.16 * np.log10(1 + f_GHz) + 0.28

        # Fator Rice (dB)
        mu_K_R  = None
        sig_K_R = None

        # Espalhamento azimutal de saída - σ_ϕ;AoD
        mu_esp_az_saida  = -0.23 * np.log10(1 + f_GHz) + 1.53
        sig_esp_az_saida = 0.11 * np.log10(1 + f_GHz) + 0.33

        # Espalhamento azimutal de chegada - σ_ϕ;AoA
        mu_esp_az_chegada  = -0.08 * np.log10(1 + f_GHz) + 1.81
        sig_esp_az_chegada = 0.05 * np.log10(1 + f_GHz) + 0.3

        # Espalhamento em elevação de saída - σ_θ;AoD
        mu_esp_el_saida  = np.max([-0.5, -3.1 * (d/1000) + 0.01 * np.abs(UE[2] - BS[2]) + 0.2])
        sig_esp_el_saida = 0.35

        # Espalhamento em elevação de chegada - σ_θ;AoA
        mu_esp_el_chegada  = -0.04 * np.log10(1 + f_GHz) + 0.92
        sig_esp_el_chegada = -0.07 * np.log10(1 + f_GHz) + 0.41

    if(indoor):
        # Espalhamento de atraso - σ_τ
        mu_esp_atraso  = -6.62
        sig_exp_atraso = 0.32

        # Fator Rice (dB)
        mu_K_R  = None
        sig_K_R = None

        # Espalhamento azimutal de saída - σ_ϕ;AoD
        mu_esp_az_saida  = 1.25
        sig_esp_az_saida = 0.42

        # Espalhamento azimutal de chegada - σ_ϕ;AoA
        mu_esp_az_chegada  = 1.76
        sig_esp_az_chegada = 0.16

        # Espalhamento em elevação de chegada - σ_θ;AoA
        mu_esp_el_chegada  = 1.01
        sig_esp_el_chegada = 0.43

    est_larga_escala = {'esp_atraso':     (mu_esp_atraso, sig_exp_atraso),
                        'fator_Rice':     (mu_K_R, sig_K_R),
                        'esp_az_saida':   (mu_esp_az_saida, sig_esp_az_saida),
                        'esp_az_chegada': (mu_esp_az_chegada, sig_esp_az_chegada),
                        'esp_el_saida':   (mu_esp_el_saida, sig_esp_el_saida),
                        'esp_el_chegada': (mu_esp_el_chegada, sig_esp_el_chegada)}

    return est_larga_escala, LoS


def amostras_larga_escala(est_larga_escala, N, rng):
    amostras = {'esp_atraso':     0,
                'fator_Rice':     0,
                'esp_az_saida':   0,
                'esp_az_chegada': 0,
                'esp_el_saida':   0,
                'esp_el_chegada': 0}
    
    for parametro in est_larga_escala:
        mu    = est_larga_escala[parametro][0]
        sigma = est_larga_escala[parametro][1]

        # Obtém amostras usando distribuição normal
        if(not(mu == None and sigma == None)):
            amostras[parametro] = rng.normal(mu, sigma)
        else:
            amostras[parametro] = None

        # Converte amostras em B ou dB para linear
        if(parametro != 'fator_Rice' and not(mu == None and sigma == None)):
            amostras[parametro] = 10**amostras[parametro]
        elif(parametro == 'fator_Rice' and not(mu == None and sigma == None)):
            amostras[parametro] = 10**(amostras[parametro]/10)

        # Limita valores de espalhamento azimutais e de elevação
        if(parametro == 'esp_az_saida' or parametro == 'esp_az_chegada'):
            amostras[parametro] = np.clip(amostras[parametro], None, 104)
        if(parametro == 'esp_el_saida' or parametro == 'esp_el_chegada'):
            amostras[parametro] = np.clip(amostras[parametro], None, 52)

    return amostras


def atrasos_multipercurso(sigma_tau, N, LoS, indoor, rng):
    # r_tau fator de proporcionalidade para a distribuição exponensial de atrasos
    if(LoS):
        r_tau = 3
    else:
        r_tau = 2.1
    if(indoor):
        r_tau = 2.2

    # Gera atrasos exponenciais:
    tau_doub_prime = np.zeros(N)
    mu_tau = r_tau * sigma_tau
    tau_doub_prime = rng.exponential(mu_tau, N)

    # Normalizando e ordenando
    tau_prime = tau_doub_prime - np.min(tau_doub_prime)
    tau = tau_prime[np.argsort(tau_prime)]

    return tau, r_tau, np.argsort(tau_prime)


def potencia_multipercurso(tau, sigma_tau, r_tau, K_R, LoS, N, rng):
    # Termos de sombreamento
    sigma_SF = 4 # Tabela 7.4.1-1
    SF = rng.normal(0, sigma_SF, N)

    # Potência preliminar
    alpha_hat_quad = np.exp(-tau * (r_tau - 1)/(r_tau*sigma_tau)) * 10**(SF/10)

    # Normalizando potências
    if(LoS):
        pot_dispersa = np.sum(alpha_hat_quad[1:])
        alpha_quad = (1/(K_R+1)) * (alpha_hat_quad/pot_dispersa)
        alpha_quad[0] = K_R/(K_R + 1)
    else:
        pot_dispersa = np.sum(alpha_hat_quad)
        alpha_quad = alpha_hat_quad/pot_dispersa

    return alpha_quad, pot_dispersa


def plot_potencia_multipercurso(alpha_quad, tau, sigma_tau):
    fig, ax = plt.subplots(figsize = (8, 4.5), layout = 'constrained')

    ax.stem(1e6*tau, alpha_quad, markerfmt='^')
    ax.plot(0, 0, label=fr'$\sigma_\tau = {1e9*sigma_tau:.2f}$ ns')
    ax.grid(True)

    ax.set_title('Potência Multipercurso')
    ax.set_xlabel(r'$\tau$ ($\mu$s)')
    ax.set_ylabel(r'$\alpha_n^2$', rotation=0)
    ax.set_yscale('log')
    ax.legend(handlelength=0, handletextpad=0)

    plt.show()


def azimute_chegada(alpha_quad, sigma_phi_AoA, phi_chegada, LoS, N, rng):
    # Ângulos azimutais iniciais
    phi_doub_prime = 1.42*sigma_phi_AoA * np.sqrt(-np.log(alpha_quad/np.max(alpha_quad)))

    # Sinais aleatórios
    U_n = rng.choice([-1, 1], N)

    # Flutuações aleatórias
    Y_n = rng.normal(0, sigma_phi_AoA/7, N)

    # Resultados finais
    phi_prime = U_n*phi_doub_prime + Y_n + phi_chegada

    # Caso haja visada direta:
    if(LoS):
        phi_prime[0] = phi_chegada

    # Normaliza dentro de -180 a 180 graus
    for n in range(N):
        while(phi_prime[n] < -180):
            phi_prime[n] += 360
        while(phi_prime[n] > 180):
            phi_prime[n] -= 360

    return phi_prime


def plot_azimute_chegada(alpha_quad, phi_prime, sigma_phi_AoA):
    fig, ax = plt.subplots(figsize = (8, 4.5), layout = 'constrained', subplot_kw={'projection': 'polar'})
    ax.stem(np.deg2rad(phi_prime), alpha_quad, bottom=alpha_quad.min() / 10, linefmt='purple')
    ax.plot(0, 0, color='k', label=rf'σφ;AoA = {sigma_phi_AoA:.2f}$^o$')

    ticks = np.linspace(alpha_quad.min(), alpha_quad.max(), 5)
    ticks = np.trunc(ticks * 100) / 100
    ax.set_rticks(ticks) 
    ax.set_rscale('log')
    ax.legend(handlelength=0, handletextpad=0)
    ax.grid(True)

    ax.set_title('Espectro Angular de Potência (Azimute)')
    plt.show()


def elev_chegada(alpha_quad, sigma_theta_AoA, theta_chegada, LoS, N, rng):
    theta_doub_prime = -sigma_theta_AoA * np.log(alpha_quad/np.max(alpha_quad))
    U_n = rng.choice([-1, 1], N)
    Y_n = rng.normal(0, sigma_theta_AoA/7, N)

    theta_prime = U_n*theta_doub_prime + Y_n + theta_chegada

    if(LoS):
        theta_prime[0] = theta_chegada
    for n in range(N):
        while(theta_prime[n] < -180):
            theta_prime[n] += 360
        while(theta_prime[n] > 180):
            theta_prime[n] -= 360

    return theta_prime


def plot_elev_chegada(alpha_quad, theta_prime, theta_phi_AoA):
    fig, ax = plt.subplots(figsize = (8, 4.5), layout = 'constrained', subplot_kw={'projection': 'polar'})
    ax.stem(np.deg2rad(theta_prime), alpha_quad, bottom=alpha_quad.min() / 10, linefmt='orange')
    ax.plot(0, 0, color='k', label=rf'σ_θ;AoA = {theta_phi_AoA:.2f}$^o$')

    ticks = np.linspace(alpha_quad.min(), alpha_quad.max(), 5)
    ticks = np.trunc(ticks * 100) / 100
    ax.set_rticks(ticks) 
    ax.set_rscale('log')
    ax.legend(handlelength=0, handletextpad=0)
    ax.grid(True)

    ax.set_title('Espectro Angular de Potência (Elevação)')
    plt.show()


def dir_chegada(phi_prime, theta_prime):
    phi_prime = np.deg2rad(phi_prime)
    theta_prime = np.deg2rad(theta_prime)

    r_x = np.cos(phi_prime) * np.sin(theta_prime)
    r_y = np.sin(phi_prime) * np.sin(theta_prime)
    r_z = np.cos(theta_prime)

    return (r_x, r_y, r_z)


def plot_dir_chegada(r_n, alpha_quad):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(projection='3d')

    cores = (alpha_quad - alpha_quad.min()) / (alpha_quad.max() - alpha_quad.min())
    c_map = plt.cm.viridis(cores)
    c_map_flat = c_map.reshape(-1, 4)
    r_n = r_n/np.linalg.norm(r_n)

    ax.quiver(0, 0, 0, r_n[0], r_n[1], r_n[2],
              length=1, normalize=True, color=c_map_flat, arrow_length_ratio=0.1, linewidths=1.2)

    sm = cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(vmin=cores.min(), vmax=cores.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.1)
    cbar.set_label(r'Potência Multipercurso $\alpha_n^2$ (normalizada)')
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.set_title('Direções de Chegada das Componentes Multipercurso')
    ax.view_init(6, 35)

    plt.show()


def desvio_doppler(f_GHz, r_n, v_rx):
    c = 299_792_458

    f = 1e9 * f_GHz
    comp_onda = c/f

    r_n = np.column_stack((r_n[0], r_n[1], r_n[2]))
    nu_n = np.zeros(len(r_n))

    for i in range(len(r_n)):
        nu = (1/comp_onda) * np.dot(r_n[i], v_rx)
        nu_n[i] = nu

    return nu_n


def plot_doppler(alpha_quad, nu_n):
    fig, ax = plt.subplots(figsize = (8, 4.5), layout = 'constrained')

    ax.stem(nu_n, alpha_quad, markerfmt='^') # mudando os parametros da curva
    ax.grid(True) #adicinando linhas

    ax.set_title('Desvio Doppler')
    ax.set_xlabel(r'$\nu$ (Hz)')
    ax.set_ylabel(r'$\alpha_n^2$', rotation=0)
    ax.set_yscale('log')

    plt.show()


def fases_multipercurso(f_GHz, nu_n, tau):
    return 2*np.pi*((1e9*f_GHz + nu_n)*tau)


def plot_fases_multipercurso(nu_n, phi_n_bar):
    t_i = np.linspace(0, 1e-3, 1000)

    fig, ax = plt.subplots(figsize = (8, 4.5), layout = 'constrained')

    for n in range(10, 100, 10):
        ax.plot(1e3*t_i, (phi_n_bar[n] - 2*np.pi * nu_n[n] * t_i), linestyle ='--', label=f'CM #{n}')

    ax.grid(True)
    ax.set_title('Fases Multipercurso')
    ax.set_xlabel(r'tempo ($m\;s$)')
    ax.set_ylabel(r'Defasagem total ($rad$)')
    ax.legend()

    plt.show()
