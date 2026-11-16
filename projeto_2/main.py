'''
    Neste projeto, são estudadas três técnicas de diversidade:
        - Selection Combining (SC)
        - Maximal Ratio Combining (MRC)
        - Equal Gain Combining (EGC)

    Para cada uma delas, serão analisadas as métricas de desempenho de probabilidade de outage, probabilidade de
    erro de símbolo e capacidade ergódica em diferentes cenários. Com isso, espera-se:
        - observar melhorias de performance do sistema, resultantes de cada uma das técnicas supracitadas;
        - comparar como a eficácia desses mecanismos evolui à medida que se acrescenta ramos de diversidade;
        - avaliar ganhos relativos de desempenho para diferentes canais;
        - e como uma investigação complementar, determinar se há influência da antena receptora no comportamento
          do SCC, MRC e EGC.
'''

import numpy as np
import matplotlib.pyplot as plt
from math import factorial, comb
from scipy.special import exp1, expn

def dB(x) : return 10 * np.log10(x)

def lin(x): return 10**(x/10)

def ganho_relativo_Rice(curvas):
    inicial = np.asarray(curvas[0])
    final = np.asarray(curvas[-1])
    ganho = np.full(inicial.shape, np.nan)
    validos = (inicial > 0) & (final > 0)
    ganho[validos] = dB(final[validos] / inicial[validos])
    return ganho

######################
#                    #
#    0 - ENTRADAS    #
#                    #
######################

snr_media_dB = 10
SEED = 4214
N_t  = 100_000
E_s  = 1
M    = 4
L    = 4
K_r  = 10

rng = np.random.default_rng(SEED)
rng_K = np.random.default_rng(SEED + 1)

######################################################
#                                                    #
#    1 - MÉTRICAS DE DESEMPENHO NO CANAL RAYLEIGH    #
#                                                    #
######################################################

'''
    Na avaliação das métricas de desempenho, a energia média de símbolo será fixa enquanto a potência de ruído
    é variada. Os símbolos transmitidos serão de uma constelação 16-QAM.
'''

from scrpits_auxiliares.canal import (init_constelacao_M_QAM, Tx_simbolos,
                                    init_r_beta_phi_l, SEP_QAM_Rayleigh)
from scrpits_auxiliares.diversidade import (SC, MRC, EGC)
from scrpits_auxiliares.visualizacoes import (plot_p_out, plot_p_erro, plot_capacidade, plot_ganhos,
                                             plot_p_out_K, plot_p_erro_K, plot_capacidade_K,
                                             plot_ganhos_canais, plot_desempenho_canais,
                                             plot_desempenho_antenas, plot_diagramas_polares)

K_r = lin(K_r)
N_g = 30
th = lin(np.linspace(-10, 20, N_g))
gamma_s = lin(np.linspace(-5, 40, N_g))

constelacao = init_constelacao_M_QAM(M, E_s)
s_i = rng.choice([l[0] for l in constelacao], size=(N_t, 2))

# Dados para desvanecimento Rayleigh:
out_SC  = []
out_MRC = []
out_EGC = []
out_SC_analiticas  = []
out_MRC_analiticas = []

SER_SC  = []
SER_MRC = []
SER_EGC = []
SEP_SC  = []
SEP_MRC = []

C_SC  = []
C_MRC = []
C_EGC = []
C_SC_analitica  = []
C_MRC_analitica = []

# Dados para desvanecimento Rice
out_SC_K  = []
out_MRC_K = []
out_EGC_K = []
SER_SC_K  = []
SER_MRC_K = []
SER_EGC_K = []
C_SC_K    = []
C_MRC_K   = []
C_EGC_K   = []


for l in range(L):
    r_l, beta_l, phi_l = init_r_beta_phi_l(l)

    for ramo in range(l+1):
        r_l[ramo], beta_l[ramo], phi_l[ramo] = Tx_simbolos(desvanecimento='Rayleigh',
                                                           SNR=gamma_s,
                                                           rng=rng,
                                                           N_t=N_t,
                                                           E_s=E_s,
                                                           s_i=s_i)

    r_l = np.asarray(r_l)
    beta_l = np.asarray(beta_l)
    phi_l = np.asarray(phi_l)

    r_SC  = SC(beta_l, r_l)
    r_MRC = MRC(beta_l, phi_l, r_l)
    r_EGC = EGC(phi_l, r_l)

    #######################################
    #                                     #
    #    1.1 - PROBABILIDADE DE OUTAGE    #
    #                                     #
    #######################################

    # SC:
    beta_SC = np.max(beta_l, axis=0)
    snr_SC = gamma_s[:, None] * beta_SC**2           # (N_g, N_t)
    snr_SC_out = lin(snr_media_dB) * beta_SC[0]**2   # (N_t,)

    # Fração dos símbolos cuja SNR ficou abaixo do limiar
    p_out_SC = np.array([np.mean(snr_SC_out < limiar) for limiar in th])
    out_SC.append(p_out_SC)

    # Expressão analítica: Rayleigh independente com E[beta²] = 1
    p_out_analitica = (1 - np.exp(-th / lin(snr_media_dB)))**(l + 1)
    out_SC_analiticas.append(p_out_analitica)

    # MRC:
    beta_MRC = np.sum(beta_l**2, axis=0)             # (N_g, N_t)
    snr_MRC = gamma_s[:, None] * beta_MRC
    snr_MRC_out = lin(snr_media_dB) * beta_MRC[0]    # (N_t,)

    p_out_MRC = np.array([np.mean(snr_MRC_out < limiar) for limiar in th])
    out_MRC.append(p_out_MRC)

    x = th / lin(snr_media_dB)
    p_out_MRC_analitica = 1 - np.exp(-x) * sum(x**k / factorial(k) for k in range(l + 1))
    out_MRC_analiticas.append(p_out_MRC_analitica)

    # EGC:
    beta_EGC = np.sum(beta_l, axis=0)**2 / (l + 1)  # (N_g, N_t)
    snr_EGC = gamma_s[:, None] * beta_EGC
    snr_EGC_out = lin(snr_media_dB) * beta_EGC[0]   # (N_t,)

    p_out_EGC = np.array([np.mean(snr_EGC_out < limiar) for limiar in th])
    out_EGC.append(p_out_EGC)

    ################################################
    #                                              #
    #    1.2 - PROBABILIDADE DE ERRO DE SÍMBOLO    #
    #                                              #
    ################################################

    # Fase do mesmo ramo selecionado pelo SC
    idx_SC = np.argmax(beta_l, axis=0)
    phi_SC = np.take_along_axis(phi_l, idx_SC[None, ...], axis=0)[0]
    amplitude_EGC = np.sum(beta_l, axis=0)
    ser_SC, ser_MRC, ser_EGC = [], [], []

    for g in range(N_g):
        sc_complex = (r_SC[g, :, 0] + 1j * r_SC[g, :, 1]) * np.exp(-1j * phi_SC[g])
        sc_coerente = np.stack((sc_complex.real, sc_complex.imag), axis=-1)

        for recebido, ganho, curva in ((sc_coerente, beta_SC[g], ser_SC),
                                       (r_MRC[g], beta_MRC[g], ser_MRC),
                                       (r_EGC[g], amplitude_EGC[g], ser_EGC),):
            # MAP para símbolos equiprováveis, como na atividade 5.
            distancias = np.sum((recebido[:, None, :] - ganho[:, None, None]
                               * constelacao[None, :, :])**2, axis=2)
            s_estimado = constelacao[np.argmin(distancias, axis=1)]
            curva.append(np.mean(np.any(s_estimado != s_i, axis=1)))

    SER_SC.append(ser_SC)
    SER_MRC.append(ser_MRC)
    SER_EGC.append(ser_EGC)
    SEP_SC.append(SEP_QAM_Rayleigh(gamma_s, M, l + 1, 'SC'))
    SEP_MRC.append(SEP_QAM_Rayleigh(gamma_s, M, l + 1, 'MRC'))

    ###################################
    #                                 #
    #    1.3 - CAPACIDADE ERGÓDICA    #
    #                                 #
    ###################################

    # Capacidade de Shannon normalizada por B_w
    C_SC.append(np.mean(np.log2(1 + snr_SC), axis=1))
    C_MRC.append(np.mean(np.log2(1 + snr_MRC), axis=1))
    C_EGC.append(np.mean(np.log2(1 + snr_EGC), axis=1))

    K = l + 1
    # SC:
    C_SC_analitica.append(K / np.log(2) * sum(comb(K - 1, k) * (-1)**k / (k + 1)
                                            * np.exp((k + 1) / gamma_s) * exp1((k + 1) / gamma_s)
                                              for k in range(K)))

    # MRC:
    C_MRC_analitica.append(np.exp(1 / gamma_s) / np.log(2) * sum(expn(k, 1 / gamma_s) for k in range(1, K + 1)))

    # Rice: mesmas métricas simuladas, para l + 1 ramos e K_r fixo.
    r_l_K, beta_l_K, phi_l_K = init_r_beta_phi_l(l)
    for ramo in range(l + 1):
        r_l_K[ramo], beta_l_K[ramo], phi_l_K[ramo] = Tx_simbolos(desvanecimento='Rice',
                                                                 SNR=gamma_s,
                                                                 rng=rng_K,
                                                                 N_t=N_t,
                                                                 E_s=E_s,
                                                                 s_i=s_i,
                                                                 K_r=K_r)

    r_l_K = np.asarray(r_l_K)
    beta_l_K = np.asarray(beta_l_K)
    phi_l_K = np.asarray(phi_l_K)
    r_SC_K = SC(beta_l_K, r_l_K)
    r_MRC_K = MRC(beta_l_K, phi_l_K, r_l_K)
    r_EGC_K = EGC(phi_l_K, r_l_K)

    beta_SC_K = np.max(beta_l_K, axis=0)
    beta_MRC_K = np.sum(beta_l_K**2, axis=0)
    amplitude_EGC_K = np.sum(beta_l_K, axis=0)
    ganho_EGC_K = amplitude_EGC_K**2 / (l + 1)

    # Outage: SNR média fixa, variando o limiar, como no caso Rayleigh.
    for ganho, curva_out, curva_C in ((beta_SC_K**2, out_SC_K, C_SC_K),
                                      (beta_MRC_K, out_MRC_K, C_MRC_K),
                                      (ganho_EGC_K, out_EGC_K, C_EGC_K),):

        snr_out_K = lin(snr_media_dB) * ganho[0]
        curva_out.append(np.array([np.mean(snr_out_K < limiar) for limiar in th]))
        snr_K = gamma_s[:, None] * ganho
        curva_C.append(np.mean(np.log2(1 + snr_K), axis=1))

    # SER: detecção coerente com CSI perfeita e símbolos equiprováveis.
    idx_SC_K = np.argmax(beta_l_K, axis=0)
    phi_SC_K = np.take_along_axis(phi_l_K, idx_SC_K[None, ...], axis=0)[0]
    ser_SC_K, ser_MRC_K, ser_EGC_K = [], [], []
    for g in range(N_g):
        sc_complex_K = (r_SC_K[g, :, 0] + 1j * r_SC_K[g, :, 1]) * np.exp(-1j * phi_SC_K[g])
        sc_coerente_K = np.stack((sc_complex_K.real, sc_complex_K.imag), axis=-1)
        for recebido, ganho, curva in ((sc_coerente_K, beta_SC_K[g], ser_SC_K),
                                       (r_MRC_K[g], beta_MRC_K[g], ser_MRC_K),
                                       (r_EGC_K[g], amplitude_EGC_K[g], ser_EGC_K),):
            
            distancias = np.sum((recebido[:, None, :] - ganho[:, None, None]
                                * constelacao[None, :, :])**2, axis=2)
            s_estimado = constelacao[np.argmin(distancias, axis=1)]
            curva.append(np.mean(np.any(s_estimado != s_i, axis=1)))

    SER_SC_K.append(ser_SC_K)
    SER_MRC_K.append(ser_MRC_K)
    SER_EGC_K.append(ser_EGC_K)


plot_p_out(0, dB(th), out_SC, out_SC_analiticas, L, snr_media_dB, 'SC')
plot_p_out(0, dB(th), out_MRC, out_MRC_analiticas, L, snr_media_dB, 'MRC')
plot_p_out(0, dB(th), out_EGC, None, L, snr_media_dB, 'EGC')
plot_p_erro(gamma_s, SER_SC, SEP_SC, L, M, 'SC')
plot_p_erro(gamma_s, SER_MRC, SEP_MRC, L, M, 'MRC')
plot_p_erro(gamma_s, SER_EGC, None, L, M, 'EGC')
plot_capacidade(gamma_s, C_SC, C_SC_analitica, L, 'SC')
plot_capacidade(gamma_s, C_MRC, C_MRC_analitica, L, 'MRC')
plot_capacidade(gamma_s, C_EGC, None, L, 'EGC')

#######################################################
#                                                     #
#    2 - COMPARAÇÃO DE DESEMPENHO ENTRE AS TÉCNICAS   #
#                                                     #
#######################################################

# Item 2)
ganho_outage_SC  = ganho_relativo_Rice(out_SC)
ganho_outage_MRC = ganho_relativo_Rice(out_MRC)
ganho_outage_EGC = ganho_relativo_Rice(out_EGC)
ganho_outage = [ganho_outage_SC, ganho_outage_MRC, ganho_outage_EGC]
plot_ganhos(dB(th), ganho_outage, 'Probabilidade de Outage')

ganho_SER_SC  = ganho_relativo_Rice(SER_SC)
ganho_SER_MRC = ganho_relativo_Rice(SER_MRC)
ganho_SER_EGC = ganho_relativo_Rice(SER_EGC)
ganho_SER = [ganho_SER_SC, ganho_SER_MRC, ganho_SER_EGC]
plot_ganhos(dB(gamma_s), ganho_SER, 'SER')

ganho_C_SC  = ganho_relativo_Rice(C_SC)
ganho_C_MRC = ganho_relativo_Rice(C_MRC)
ganho_C_EGC = ganho_relativo_Rice(C_EGC)
ganho_C = [ganho_C_SC, ganho_C_MRC, ganho_C_EGC]
plot_ganhos(dB(gamma_s), ganho_C, 'Capacidade Ergódica')

##############################################################
#                                                            #
#    3 - COMPARAÇÃO DE DESEMPENHO ENTRE OS DESVANECIMENTOS   #
#                                                            #
##############################################################

# Item 3)
plot_p_out_K(dB(th), out_SC_K, out_MRC_K, out_EGC_K, L, snr_media_dB, K_r)
plot_p_erro_K(gamma_s, SER_SC_K, SER_MRC_K, SER_EGC_K, L, M, K_r)
plot_capacidade_K(gamma_s, C_SC_K, C_MRC_K, C_EGC_K, L, K_r)

ganho_outage_K = [ganho_relativo_Rice(curvas)
                  for curvas in (out_SC_K, out_MRC_K, out_EGC_K)]
ganho_SER_K = [ganho_relativo_Rice(curvas)
               for curvas in (SER_SC_K, SER_MRC_K, SER_EGC_K)]
ganho_C_K = [ganho_relativo_Rice(curvas)
             for curvas in (C_SC_K, C_MRC_K, C_EGC_K)]

plot_ganhos_canais(dB(th), ganho_outage, ganho_outage_K,
                   'Probabilidade de Outage', L, K_r, snr_media_dB)
plot_ganhos_canais(dB(gamma_s), ganho_SER, ganho_SER_K, 'SER', L, K_r)
plot_ganhos_canais(dB(gamma_s), ganho_C, ganho_C_K, 'Capacidade Ergódica', L, K_r)

# Desempenhos absolutos com quatro ramos, sem calcular razões entre métricas.
plot_desempenho_canais(dB(th), (out_SC, out_MRC, out_EGC),
                       (out_SC_K, out_MRC_K, out_EGC_K),
                       'Probabilidade de Outage', K_r, ramos=4, snr_media_dB=snr_media_dB)
plot_desempenho_canais(dB(gamma_s), (SER_SC, SER_MRC, SER_EGC),
                       (SER_SC_K, SER_MRC_K, SER_EGC_K), 'SER', K_r, ramos=4, M=M)
plot_desempenho_canais(dB(gamma_s), (C_SC, C_MRC, C_EGC),
                       (C_SC_K, C_MRC_K, C_EGC_K), 'Capacidade Ergódica', K_r, ramos=4)

###############################################################
#                                                             #
#    4 - EFEITO DO DIAGRAMA DA ANTENA RECEPTORA (RAYLEIGH)    #
#                                                             #
###############################################################

from scrpits_auxiliares.antenas import receber_rayleigh_antenas, snr_combinadores

gamma_ant = np.array([lin(snr_media_dB)])
limiar_ant = lin(5)  # 5 dB; mantido igual para os dois diagramas
resultados_ant = {}
angulos_ant = None

for diagrama in ('dipolo', 'parabólica'):
    rng_ant = np.random.default_rng(SEED + 2)
    _, beta_ant, _, Gr_ant, theta_ant = receber_rayleigh_antenas(
        rng_ant, s_i[:, 0] + 1j * s_i[:, 1], gamma_ant, L,
        diagrama=diagrama, E_s=E_s)

    if angulos_ant is None:
        angulos_ant = theta_ant

    snrs_ant = snr_combinadores(beta_ant, gamma_ant)
    resultados_ant[diagrama] = {
        tecnica: (np.mean(snr[:, 0, :] < limiar_ant, axis=1),
                  np.mean(np.log2(1 + snr[:, 0, :]), axis=1))
        for tecnica, snr in snrs_ant.items()}

plot_desempenho_antenas(resultados_ant, L, snr_media_dB)
plot_diagramas_polares(angulos_ant)
