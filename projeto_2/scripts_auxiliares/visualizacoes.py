import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ARTEFATOS = Path(__file__).resolve().parents[1] / 'artefatos'
_NUMERO_GRAFICO = 0


def _salvar(fig, nome):
    global _NUMERO_GRAFICO
    _NUMERO_GRAFICO += 1
    ARTEFATOS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ARTEFATOS / f'{_NUMERO_GRAFICO}-{nome}',
                dpi=200, bbox_inches='tight')


def plot_diagramas_polares(theta_chegada, diametro_lambda=3.0, n_raios=80):
    """Diagramas de potência e AoAs sorteadas no corte polar do primeiro ramo."""
    from scrpits_auxiliares.antenas import ganho_dipolo, ganho_parabola

    theta_chegada = np.asarray(theta_chegada)
    if theta_chegada.ndim == 2:
        theta_chegada = theta_chegada[0]
    if theta_chegada.ndim != 1 or theta_chegada.size == 0:
        raise ValueError('theta_chegada deve conter ângulos de chegada')

    grade = np.linspace(0, np.pi, 721)
    if n_raios <= 0:
        raise ValueError('n_raios deve ser positivo')
    indices = np.linspace(0, theta_chegada.size - 1,
                          min(n_raios, theta_chegada.size), dtype=int)
    direcoes = theta_chegada[indices]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': 'polar'})
    for ax, nome, ganho in ((axs[0], 'Dipolo curto', ganho_dipolo),
                           (axs[1], 'Abertura circular (parabólica)',
                            lambda t: ganho_parabola(t, diametro_lambda))):
        raio = ganho(grade)
        # Espelha o corte para exibir a seção completa do diagrama.
        ax.plot(grade, raio, color='C0', linewidth=3, label='Ganho de potência')
        if nome == 'Dipolo curto':
            ax.plot(2 * np.pi - grade, raio, color='C0', linewidth=3)
        comprimento_raios = raio.max() * 1.05
        for angulo in direcoes:
            ax.plot([angulo, angulo], [0, comprimento_raios],
                    color='C1', alpha=0.10, linewidth=1, zorder=1)
        ax.scatter(direcoes, np.full_like(direcoes, comprimento_raios),
                   color='C1', alpha=0.25, s=12, zorder=4,
                   label='Ângulos de chegada sorteados')
        ax.set_theta_zero_location('N')
        ax.set_title(nome)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15))
    fig.suptitle('Diagramas receptores e ângulos de chegada sorteados')
    fig.tight_layout()
    _salvar(fig, 'diagramas_polares_antenas.png')
    plt.show()


def plot_p_out(K_r, SNR, p_out_sim, p_out_ext, L, snr_media_dB, tecnica):
    fig, ax = plt.subplots(1, 1)

    if (K_r == 0):
        for i in range(L):
            curva = p_out_sim[i] if p_out_ext is None else p_out_ext[i]
            if (tecnica != 'EGC'):
                ax.plot(SNR, curva, color='C' + str(i), linewidth=3,
                        linestyle='-', zorder=1, label=f'L={i+1}')
            else:
                ax.plot(SNR, curva, color='C' + str(i), linewidth=3,
                        linestyle='--', zorder=1, label=f'L={i+1}')
            ax.scatter(SNR, p_out_sim[i], color='k', marker='+', zorder=10,
                       label='Simulação' if i == L - 1 else None)

    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-5, 1.1)
    ax.set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)')
    ax.set_ylabel('OP', rotation=1)
    ax.set_yscale('log')
    ax.set_title(f'OP, {tecnica}, SNR média por ramo = {snr_media_dB} dB')
    ax.legend()
    fig.tight_layout()
    _salvar(fig, f'outage_rayleigh_{tecnica.lower()}.png')
    plt.show()


def plot_p_erro(SNR, ser, sep, L, M, tecnica):
    fig, ax = plt.subplots(1, 1)
    snr_dB = 10 * np.log10(SNR)

    for i in range(L):
        simulada = np.asarray(ser[i])
        simulada = np.where(simulada > 0, simulada, np.nan)
        curva = simulada if sep is None else sep[i]
        if (tecnica != 'EGC'):
            ax.semilogy(snr_dB, curva, color=f'C{i}', label=f'L={i+1}', linewidth=3)
        else:
            ax.semilogy(snr_dB, curva, color=f'C{i}', label=f'L={i+1}', linestyle='--', linewidth=3)
        ax.scatter(snr_dB, simulada, color='k', marker='x',
                   label='Simulação' if i == L - 1 else None, zorder=10)

    ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)')
    ax.set_ylabel('Probabilidade de erro de símbolo')
    ax.set_title(f'SEP e SER, {tecnica}, Rayleigh, {M}-QAM')
    ax.set_ylim(1e-5, 1)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _salvar(fig, f'ser_rayleigh_{tecnica.lower()}.png')
    plt.show()


def plot_capacidade(SNR, c_sim, c_teo, L, tecnica):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    snr_dB = 10 * np.log10(SNR)

    for i in range(L):
        curva = c_sim[i] if c_teo is None else c_teo[i]
        if (tecnica != 'EGC'):
            ax.plot(snr_dB, curva, color=f'C{i}', label=f'L={i+1}', linewidth=3)
        else:
            ax.plot(snr_dB, curva, color=f'C{i}', label=f'L={i+1}', linestyle='--', linewidth=3)
        ax.scatter(snr_dB, c_sim[i], color='k', marker='x', zorder=10,
                   label='Simulação' if i == L - 1 else None)

    ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)')
    ax.set_ylabel('Capacidade normalizada (bits/s/Hz)')
    ax.set_title(f'Capacidade ergódica, {tecnica}, Rayleigh')
    ax.set_xlim(snr_dB[0], snr_dB[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _salvar(fig, f'capacidade_rayleigh_{tecnica.lower()}.png')
    plt.show()


def _plot_Rice(SNR_dB, curvas, L, titulo, xlabel, ylabel, nome_arquivo,
               limites=None, log_y=False):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for j, tecnica in enumerate(('SC', 'MRC', 'EGC')):
        for i in range(L):
            curva = np.asarray(curvas[j][i])
            if log_y:
                curva = np.where(curva > 0, curva, np.nan)
            axs[j].plot(SNR_dB, curva, color=f'C{i}', linestyle='--',
                        label=f'L={i+1}', linewidth=3)
            axs[j].scatter(SNR_dB, curva, color='k', marker='x', zorder=10)
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
    _salvar(fig, nome_arquivo)
    plt.show()


def plot_p_out_K(th_dB, out_SC, out_MRC, out_EGC, L, snr_media_dB, K_r):
    _plot_Rice(th_dB, (out_SC, out_MRC, out_EGC), L,
               f'Outage simulado — Rice, K={K_r} (linear), SNR média por ramo = {snr_media_dB} dB',
               r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)',
               'Probabilidade de outage', 'outage_rice_tres_tecnicas.png',
               limites=(1e-4, 1.1), log_y=True)


def plot_p_erro_K(SNR, ser_SC, ser_MRC, ser_EGC, L, M, K_r):
    _plot_Rice(10 * np.log10(SNR), (ser_SC, ser_MRC, ser_EGC), L,
               f'SER simulada — Rice, K={K_r} (linear), {M}-QAM',
               r'SNR média por ramo $\bar{\gamma}_s$ (dB)',
               'Probabilidade de erro de símbolo', 'ser_rice_tres_tecnicas.png',
               limites=(1e-4, 1), log_y=True)


def plot_capacidade_K(SNR, c_SC, c_MRC, c_EGC, L, K_r):
    _plot_Rice(10 * np.log10(SNR), (c_SC, c_MRC, c_EGC), L,
               f'Capacidade ergódica simulada — Rice, K={K_r} (linear)',
               r'SNR média por ramo $\bar{\gamma}_s$ (dB)',
               'Capacidade normalizada (bits/s/Hz)',
               'capacidade_rice_tres_tecnicas.png')


def plot_ganhos_canais(SNR_dB, ganhos_rayleigh, ganhos_rice, metrica, L, K_r, snr_media_dB=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, tecnica in enumerate(('SC', 'MRC', 'EGC')):
        for ganhos, canal, estilo in ((ganhos_rayleigh, 'Rayleigh', '-'),
                                      (ganhos_rice, 'Rice', '--')):
            curva = np.asarray(ganhos[i])
            curva = np.where(np.isfinite(curva), curva, np.nan)
            ax.plot(SNR_dB, curva, color=f'C{i}', linestyle=estilo,
                    linewidth=3,
                    label=f'{tecnica} — {canal}')
            if canal == 'Rice':
                ax.scatter(SNR_dB, curva, color='k', marker='x', zorder=10)

    if metrica == 'Probabilidade de Outage':
        ax.set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)')
        detalhe = f', SNR média por ramo = {snr_media_dB} dB'
    else:
        ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)')
        detalhe = ''
    ax.set_ylabel('Ganho relativo (dB)')
    ax.set_xlim(SNR_dB[0], SNR_dB[-1])
    ax.set_title(f'Ganhos relativos de 1 a {L} ramos — {metrica}\n'
                 f'Rayleigh e Rice, K={K_r} (linear){detalhe}')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    nome = {'Probabilidade de Outage': 'outage', 'SER': 'ser',
            'Capacidade Ergódica': 'capacidade'}[metrica]
    _salvar(fig, f'ganho_relativo_{nome}_rayleigh_rice.png')
    plt.show()


def plot_desempenho_canais(SNR_dB, curvas_rayleigh, curvas_rice, metrica,
                          K_r, ramos, snr_media_dB=None, M=None):
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
                    linewidth=3,
                    label=f'{tecnica} — {canal}')
            if canal == 'Rice':
                ax.scatter(SNR_dB, curva, color='k', marker='x', zorder=10)

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
    nome = {'Probabilidade de Outage': 'outage', 'SER': 'ser',
            'Capacidade Ergódica': 'capacidade'}[metrica]
    _salvar(fig, f'desempenho_absoluto_{nome}_{ramos}_ramos_rayleigh_rice.png')
    plt.show()


def plot_ganhos(SNR, curvas_metrica, metrica):
    fig, ax = plt.subplots(1, 1)

    ax.plot(SNR, curvas_metrica[0], linewidth=3, linestyle='-', zorder=1, label=f'SC')
    ax.plot(SNR, curvas_metrica[1], linewidth=3, linestyle='-', zorder=1, label=f'MRC')
    ax.plot(SNR, curvas_metrica[2], linewidth=3, linestyle='-', zorder=1, label=f'EGC')

    ax.grid(True, which='both', alpha=0.3)
    # ax.set_ylim(1e-2, 1.1)
    if (metrica == 'Probabilidade de Outage'):
        ax.set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB), SNR média por ramo = 10 dB')
    else:
        ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)')
    ax.set_ylabel('Ganho Relativo (dB)')
    ax.set_title(rf'Ganhos relativos de 1 a 4 ramos - {metrica}')
    ax.legend()
    fig.tight_layout()
    nome = {'Probabilidade de Outage': 'outage', 'SER': 'ser',
            'Capacidade Ergódica': 'capacidade'}[metrica]
    _salvar(fig, f'ganho_relativo_{nome}_rayleigh.png')
    plt.show()


def plot_desempenho_antenas(resultados_ant, L, snr_media_dB):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, tecnica in enumerate(('SC', 'MRC', 'EGC')):
        for diagrama, estilo in (('dipolo', '-'), ('parabólica', '--')):
            outage, capacidade = resultados_ant[diagrama][tecnica]
            axs[0].plot(range(1, L + 1), outage, color=f'C{i}',
                        linestyle=estilo, linewidth=3,
                        label=f'{tecnica} — {diagrama}')
            axs[1].plot(range(1, L + 1), capacidade, color=f'C{i}',
                        linestyle=estilo, linewidth=3,
                        label=f'{tecnica} — {diagrama}')
    axs[0].set(title='Outage absoluto (limiar = 5 dB)', ylabel='Probabilidade de outage')
    axs[1].set(title='Capacidade ergódica', ylabel='bits/s/Hz')
    for ax in axs:
        ax.set(xlabel='Número de ramos', xticks=range(1, L + 1))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'Efeito do diagrama do receptor — Rayleigh, SNR média por ramo = {snr_media_dB} dB')
    fig.tight_layout()
    _salvar(fig, 'desempenho_diagramas_antenas_rayleigh.png')
    plt.show()
