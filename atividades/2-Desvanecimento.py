import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from scipy.special import gamma,eval_laguerre

N = 1000000                     # Número de amostras da normal
x = np.linspace(0, 15, 1000)    # Intervalo para o eixo x dos gráfios

fig, axs = plt.subplots(2, 3, layout='constrained')

# Rayleigh
for n in range(3):
    sig_quad = 1 + 2*n
    sigma = np.sqrt(sig_quad)
    
    X_1 = normal(0, sigma, N)
    X_2 = normal(0, sigma, N)
    beta = np.sqrt(X_1**2 + X_2**2)

    # FDP Rayleigh
    axs[0][0].hist(beta, bins=100, density=True, alpha=0.5, label=rf'$\sigma^2 = {sig_quad}$')

    # FDA Rayleigh
    counts, bin_edge = np.histogram(beta, bins=100)
    bin_widths = np.diff(bin_edge)
    fda = np.cumsum(counts * bin_widths)
    fda = fda/fda[-1]
    axs[0][1].plot(bin_edge[1:], fda, label=rf'$\sigma^2 = {sig_quad}$')

    # Momentos Rayleigh
    momentos = np.zeros(10)
    m = np.linspace(1, 10, 10)

    for k in m:
        momentos[int(k)-1] = np.mean(beta**k)
    axs[0][2].scatter(m, momentos, marker='x', label=rf'$Sim \; \sigma^2 = {sig_quad}$')

    momento = lambda i: (2*sig_quad)**(i/2) * gamma(1 + i/2)
    axs[0][2].plot(m, momento(m), label=rf'$Eq. \; \sigma^2 = {sig_quad}$')

for n in range(3):
    sig_quad = 1 + 2*n
    sigma = np.sqrt(sig_quad)
    mu_1 = 1 + 5*n
    mu_2 = 0
    
    X_1 = normal(mu_1, sigma, N)
    X_2 = normal(mu_2, sigma, N)
    beta = np.sqrt(X_1**2 + X_2**2)

    # FDP Rice
    axs[1][0].hist(beta, bins=100, density=True, alpha=0.5, label=rf'$\sigma^2 = {sig_quad}$, $\mu_1 = {mu_1}$')

    # FDA Rice
    counts, bin_edge = np.histogram(beta, bins=100)
    bin_widths = np.diff(bin_edge)
    fda = np.cumsum(counts * bin_widths)
    fda = fda/fda[-1]
    axs[1][1].plot(bin_edge[1:], fda, label=rf'$\sigma^2 = {sig_quad}$, $\mu_1 = {mu_1}$')

    # Momentos Rayleigh
    momentos = np.zeros(10)
    m = np.linspace(1, 10, 10)

    for k in m:
        momentos[int(k)-1] = np.mean(beta**k)
    axs[1][2].scatter(m, momentos, marker='x')

    K_r = (mu_1**2) / (2*sig_quad)
    momento = lambda i: (2*sig_quad)**(i/2) * gamma(1 + i/2) * eval_laguerre(i/2, -K_r)
    axs[1][2].plot(m, momento(m))

axs[0][0].set_xlabel(r'Domínio da VA $x$')
axs[0][0].set_ylabel(r'$p_\beta(\beta)$')
axs[0][0].set_title('FDPs Rayleigh')
axs[0][0].grid(True, alpha=0.5)
axs[0][0].legend()

axs[0][1].set_xlabel(r'Domínio da VA $\beta$')
axs[0][1].set_ylabel(r'$F_\beta(\beta)$')
axs[0][1].set_title('FDAs Rayleigh')
axs[0][1].grid(True, alpha=0.5)
axs[0][1].legend()

axs[0][2].set_xlabel(r'Ordem ($n$)')
axs[0][2].set_ylabel(r'$E[\beta^n]$')
axs[0][2].set_title('Momentos de n-ésima Ordem')
axs[0][2].grid(True, alpha=0.5)
axs[0][2].set_yscale('log')
axs[0][2].legend()

axs[1][0].set_xlabel(r'Domínio da VA $x$')
axs[1][0].set_ylabel(r'$p_\beta(\beta)$')
axs[1][0].set_title(r'FDPs Rice ($\mu_2 = 0)$')
axs[1][0].grid(True, alpha=0.5)
axs[1][0].legend()

axs[1][1].set_xlabel(r'Domínio da VA $\beta$')
axs[1][1].set_ylabel(r'$F_\beta(\beta)$')
axs[1][1].set_title(r'FDAs Rice ($\mu_2 = 0$)')
axs[1][1].grid(True, alpha=0.5)
axs[1][1].legend()

axs[1][2].set_xlabel(r'Ordem ($n$)')
axs[1][2].set_ylabel(r'$E[\beta^n]$')
axs[1][2].set_title('Momentos de n-ésima Ordem')
axs[1][2].grid(True, alpha=0.5)
axs[1][2].set_yscale('log')

plt.show()