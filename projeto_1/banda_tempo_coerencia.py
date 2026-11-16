import numpy as np
import matplotlib.pyplot as plt

def autocorrelacao_canal_norm(alpha_quad, atrasos, k, sig, nu_n):
    rho = 0
    for n in range(len(alpha_quad)):
        rho += alpha_quad[n] * np.exp(-1j*2*np.pi*atrasos[n]*k) * np.exp(1j*2*np.pi*nu_n[n]*sig)

    return rho/np.sum(alpha_quad)

def banda_de_coerencia(alpha_quad, atrasos, nu_n, sigma_tau):
    k = np.logspace(0, 9, 10000)
    rho_k = np.abs(autocorrelacao_canal_norm(alpha_quad, atrasos, k, 0, nu_n))

    fig, ax = plt.subplots(figsize = (8, 4.5), layout = 'constrained')

    corte = np.argwhere(rho_k<=0.95)
    corte_0 = corte[0][0] if corte.size else 0
    k_corte = k[corte_0] if corte.size else np.nan

    ax.plot(k, rho_k, linewidth=2)
    plt.axhline(y=0.95, color='k', linestyle='-.')
    plt.axvline(x=k_corte, color='k', linestyle='--')

    ax.grid(True)
    ax.set_xscale('log')
    ax.set_title(fr'BC(95%) = {1e-6*k_corte:.2f} MHz, $\sigma_\tau = {1e9*sigma_tau:.2f} ns$')
    ax.set_xlabel(r'Desvio de Frequência - $\kappa$ (Hz)')
    ax.set_ylabel(r'$|\rho_T(\kappa, 0)|$')
    ax.legend()

    plt.show()

def tempo_de_coerencia(alpha_quad, atrasos, nu_n):
    sig = np.logspace(-6, 0, 10000)
    rho_sig = np.abs(autocorrelacao_canal_norm(alpha_quad, atrasos, 0, sig, nu_n))

    fig, ax = plt.subplots(figsize = (8, 4.5), layout = 'constrained')

    corte = np.argwhere(rho_sig<=0.95)
    corte_0 = corte[0][0] if corte.size else 0
    sig_corte = sig[corte_0] if corte.size else np.nan

    ax.plot(sig, rho_sig, linewidth=2)
    plt.axhline(y=0.95, color='k', linestyle='-.')
    plt.axvline(x=sig_corte, color='k', linestyle='--')

    ax.grid(True)
    ax.set_xscale('log')
    ax.set_title(fr'TC(95%) = {1e3*sig_corte:.2f} ms, $\max(\nu_n) = {np.max(nu_n):.2f} Hz$, $v_U = 3\;km/h$')
    ax.set_xlabel(r'Desvio de Tempo - $\sigma$ (s)')
    ax.set_ylabel(r'$|\rho_T(0, \sigma)|$')
    ax.legend()

    plt.show()
