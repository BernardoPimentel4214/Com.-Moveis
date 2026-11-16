import numpy as np
import matplotlib.pyplot as plt

def init_pulso(delta_t, N_amostras):
    t_i = np.linspace(0, 5*delta_t, N_amostras)
    f_i = np.linspace(-100/(5*delta_t), 100/(5*delta_t), N_amostras) / (2*np.pi)
    pulso = lambda t: np.heaviside(t, 1/2) - np.heaviside(t - 2*delta_t, 1/2)

    # Transformada de Fourier:
    dt = t_i[1] - t_i[0]
    P = dt*np.exp(-1j*2*np.pi*np.outer(f_i, t_i)) @ pulso(t_i)

    # Obtém espectro de amplitude:
    re = P.real
    im = P.imag
    amplitude = np.sqrt(re**2 + im**2)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), layout='constrained')
    axs[0].plot(1e6*t_i, pulso(t_i), color='red', linewidth=3)
    axs[0].set_ylabel(r'$s(t)$', rotation=1)
    axs[0].set_xlabel(r'Tempo absluto - $t\;(\mu s)$')
    axs[0].set_title('Sinal Transmitido no Tempo')
    axs[0].grid()

    axs[1].plot(1e-6*f_i, amplitude, color='red', linewidth=3)
    axs[1].set_ylabel(r'$|S(f)|$', rotation=1)
    axs[1].set_xlabel(r'Frequência - $f\;(MHz)$')
    axs[1].set_title("Espectro de Amplitude")
    axs[1].grid()

    plt.show()

    return pulso(t_i), amplitude


def transmite_pulso(delta_t, alpha_quad, phi_n_bar, nu_n, tau, N_amostras):
    t_i = np.linspace(0, 5*delta_t, N_amostras)
    pulso = lambda t: np.heaviside(t, 1/2) - np.heaviside(t - 2*delta_t, 1/2)

    r = 0
    for n in range(len(alpha_quad)):
        r += np.sqrt(alpha_quad[n]) * np.exp(-1j*(phi_n_bar[n] - 2*np.pi*nu_n[n]*t_i)) * pulso(t_i - tau[n])

    return r

def plot_pulso_recebido(pulso_transmitido, delta_t, sigma_tau, N_amostras):
    t_i = np.linspace(0, 5*delta_t, N_amostras)
    pulso = lambda t: np.heaviside(t, 0) - np.heaviside(t - 2*delta_t, 1)

    fig, ax = plt.subplots(figsize = (8, 4.5), layout = 'constrained')

    ax.plot(1e6*t_i, np.abs(pulso_transmitido), linewidth=3, label=f'Sinal Recebido')
    ax.plot(1e6*t_i, pulso(t_i), linestyle='--', linewidth=2, label=f'Sinal Trasmitido')

    ax.grid(True)
    ax.set_title(fr'Sinal Recebido, $\delta t = {delta_t}\;s$, $BW \approx {1e-6/delta_t}\;MHz$, $\sigma_\tau = {1e9*sigma_tau:.2f}\;ns$')
    ax.set_xlabel(r'tempo absoluto - t ($\mu\;s$)')
    ax.set_ylabel(r'Sinal Recebido (amplitude)')
    ax.set_xticks([0, 1e6*delta_t, 2*1e6*delta_t, 3*1e6*delta_t, 4*1e6*delta_t, 5*1e6*delta_t])
    ax.set_xticklabels([0, r'$\delta t$', r'$2\delta t$', r'$3\delta t$', r'$4\delta t$', r'$5\delta t$'])

    plt.show()
