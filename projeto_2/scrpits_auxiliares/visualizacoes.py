import numpy as np
import matplotlib.pyplot as plt

def plot_p_out(K_r, SNR, p_out_sim, p_out_ext, L, snr_media_dB, tecnica):
    fig, ax = plt.subplots(1, 1)

    if(K_r == 0):
        for i in range(L):
            curva = p_out_sim[i] if p_out_ext is None else p_out_ext[i]
            ax.plot(SNR, curva, color ='C'+ str(i) , linewidth=3, linestyle ='-', zorder=1, label=f'L={i+1}')
            ax.scatter(SNR, p_out_sim[i], color ='k', marker='+', zorder=10, label='Simulação' if i == L - 1 else None)
        
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-5, 1.1)
    ax.set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)')
    ax.set_ylabel('OP', rotation=1)
    ax.set_yscale('log')
    ax.set_title(f'OP, {tecnica}, SNR média por ramo = {snr_media_dB} dB')
    ax.legend()
    fig.tight_layout()

    plt.show()


def plot_p_erro(SNR, ser, sep, L, M, tecnica):
    fig, ax = plt.subplots(1, 1)
    snr_dB = 10 * np.log10(SNR)

    for i in range(L):
        simulada = np.asarray(ser[i])
        simulada = np.where(simulada > 0, simulada, np.nan)
        curva = simulada if sep is None else sep[i]
        ax.semilogy(snr_dB, curva, color=f'C{i}', label=f'L={i+1}')
        ax.scatter(snr_dB, simulada, color='k', marker='+',
                   label='Simulação' if i == L - 1 else None, zorder=10)

    ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)')
    ax.set_ylabel('Probabilidade de erro de símbolo')
    ax.set_title(f'SEP e SER, {tecnica}, Rayleigh, {M}-QAM')
    ax.set_ylim(1e-5, 1)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plt.show()


def plot_capacidade(SNR, c_sim, c_teo, L, tecnica):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    snr_dB = 10 * np.log10(SNR)

    for i in range(L):
        curva = c_sim[i] if c_teo is None else c_teo[i]
        ax.plot(snr_dB, curva, color=f'C{i}', label=f'L={i+1}')
        ax.scatter(snr_dB, c_sim[i], color='k', marker='+', zorder=10,
                   label='Simulação' if i == L - 1 else None)

    ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)')
    ax.set_ylabel('Capacidade normalizada (bits/s/Hz)')
    ax.set_title(f'Capacidade ergódica, {tecnica}, Rayleigh')
    ax.set_xlim(snr_dB[0], snr_dB[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plt.show()

def _plot_Rice(SNR_dB, curvas, L, titulo, xlabel, ylabel, limites=None, log_y=False):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for j, tecnica in enumerate(('SC', 'MRC', 'EGC')):
        for i in range(L):
            curva = np.asarray(curvas[j][i])
            if log_y:
                curva = np.where(curva > 0, curva, np.nan)
            axs[j].plot(SNR_dB, curva, color=f'C{i}', linestyle='--',
                        marker='+', label=f'L={i+1}')
        axs[j].set_title(tecnica)
        axs[j].set_xlabel(xlabel)
        axs[j].set_xlim(SNR_dB[0], SNR_dB[-1])
        if log_y:
            axs[j].set_yscale('log')
        if limites is not None:
            axs[j].set_ylim(*limites)
        else:
            axs[j].set_ylim(bottom=0)
        axs[j].grid(True, which='both', alpha=0.3)
        axs[j].legend()
    axs[0].set_ylabel(ylabel)
    fig.suptitle(titulo)
    fig.tight_layout()
    plt.show()


def plot_p_out_K(th_dB, out_SC, out_MRC, out_EGC, L, snr_media_dB, K_r):
    _plot_Rice(th_dB, (out_SC, out_MRC, out_EGC), L,
               f'Outage simulado — Rice, K={K_r} (linear), SNR média por ramo = {snr_media_dB} dB',
               r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)',
               'Probabilidade de outage', limites=(1e-5, 1.1), log_y=True)


def plot_p_erro_K(SNR, ser_SC, ser_MRC, ser_EGC, L, M, K_r):
    _plot_Rice(10 * np.log10(SNR), (ser_SC, ser_MRC, ser_EGC), L,
               f'SER simulada — Rice, K={K_r} (linear), {M}-QAM',
               r'SNR média por ramo $\bar{\gamma}_s$ (dB)',
               'Probabilidade de erro de símbolo', limites=(1e-5, 1), log_y=True)


def plot_capacidade_K(SNR, c_SC, c_MRC, c_EGC, L, K_r):
    _plot_Rice(10 * np.log10(SNR), (c_SC, c_MRC, c_EGC), L,
               f'Capacidade ergódica simulada — Rice, K={K_r} (linear)',
               r'SNR média por ramo $\bar{\gamma}_s$ (dB)',
               'Capacidade normalizada (bits/s/Hz)')


def plot_ganhos_canais(SNR_dB, ganhos_rayleigh, ganhos_rice, metrica, L, K_r, snr_media_dB=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, tecnica in enumerate(('SC', 'MRC', 'EGC')):
        for ganhos, canal, estilo in ((ganhos_rayleigh, 'Rayleigh', '-'),
                                      (ganhos_rice, 'Rice', '--')):
            curva = np.asarray(ganhos[i])
            curva = np.where(np.isfinite(curva), curva, np.nan)
            ax.plot(SNR_dB, curva, color=f'C{i}', linestyle=estilo,
                    marker='+' if canal == 'Rice' else None,
                    label=f'{tecnica} — {canal}')

    if metrica == 'Probabilidade de Outage':
        ax.set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)')
        detalhe = f', SNR média por ramo = {snr_media_dB} dB'
    else:
        ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)')
        detalhe = ''
    ax.set_ylabel('Ganho relativo (dB)')
    ax.set_title(f'Ganhos relativos de 1 a {L} ramos — {metrica}\n'
                 f'Rayleigh e Rice, K={K_r} (linear){detalhe}')
    ax.set_xlim(SNR_dB[0], SNR_dB[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    plt.show()


def plot_desempenho_canais(SNR_dB, curvas_rayleigh, curvas_rice, metrica, K_r, ramos, snr_media_dB=None, M=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    probabilidade = metrica in ('Probabilidade de Outage', 'SER')
    for i, tecnica in enumerate(('SC', 'MRC', 'EGC')):
        for curvas, canal, estilo in ((curvas_rayleigh, 'Rayleigh', '-'),
                                      (curvas_rice, 'Rice', '--')):
            curva = np.asarray(curvas[i][ramos - 1])
            # Zero eventos observados não é uma probabilidade exata igual a zero.
            if probabilidade:
                curva = np.where(curva > 0, curva, np.nan)
            ax.plot(SNR_dB, curva, color=f'C{i}', linestyle=estilo,
                    marker='+' if canal == 'Rice' else None,
                    label=f'{tecnica} — {canal}')

    detalhe = ''
    if metrica == 'Probabilidade de Outage':
        ax.set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)')
        ax.set_ylabel('Probabilidade de outage')
        detalhe = f', SNR média por ramo = {snr_media_dB} dB'
    else:
        ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)')
        if metrica == 'SER':
            ax.set_ylabel('Probabilidade de erro de símbolo')
            detalhe = f', {M}-QAM'
        else:
            ax.set_ylabel('Capacidade normalizada (bits/s/Hz)')
    if probabilidade:
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1.1)
    else:
        ax.set_ylim(bottom=0)
    ax.set_title(f'{metrica} — {ramos} ramos, resultados simulados\n'
                 f'Rayleigh e Rice, K={K_r} (linear){detalhe}')
    ax.set_xlim(SNR_dB[0], SNR_dB[-1])
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    plt.show()


def plot_ganhos(SNR, curvas_metrica, metrica):
    fig, ax = plt.subplots(1, 1)

    ax.plot(SNR, curvas_metrica[0], linewidth=3, linestyle ='-', zorder=1, label=f'SC')
    ax.plot(SNR, curvas_metrica[1], linewidth=3, linestyle ='-', zorder=1, label=f'MRC')
    ax.plot(SNR, curvas_metrica[2], linewidth=3, linestyle ='-', zorder=1, label=f'EGC')
        
    ax.grid(True, which='both', alpha=0.3)
    #ax.set_ylim(1e-5, 1.1)
    if (metrica == 'Probabilidade de Outage'):
        ax.set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB), SNR média por ramo = 10 dB')
    else: 
        ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)')
    ax.set_ylabel('Ganho Relativo (dB)')
    ax.set_title(rf'Ganhos relativos de 1 a 4 ramos - {metrica}')
    ax.legend()
    fig.tight_layout()

    plt.show()
