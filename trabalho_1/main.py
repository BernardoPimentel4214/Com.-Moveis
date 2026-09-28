RANDOM_SEED = 4214

####################################
#                                  #
#    1 - AMBIENTE MULTIPERCURSO    #
#                                  #
####################################

from ambiente_multipercurso import (init_geometria,
                                    plot_geometria,
                                    calcula_angulos)

BS, UE, d, v_rx, rng = init_geometria(indoor=False, seed=RANDOM_SEED) # BS, UE e v_rx tem o formato (A_x, A_y, A_z), d é um float.
plot_geometria(BS, UE, d, v_rx)                                  # indoor é uma variana booleana que representaa se a UE está em um ambiente fechado

ang_saida, ang_chegada = calcula_angulos(BS, UE)                 # ang_x = (theta_x, phi_x)

########################################
#                                      #
#    2 - PARÂMETROS DE LARGA ESCALA    #
#                                      #
########################################

from parametros_larga_escala import (init_est_larga_escala,
                                     amostras_larga_escala)

mu_sig_larga_escala, LoS = init_est_larga_escala(f_GHz=3,
                                                 UE=UE,
                                                 BS=BS,
                                                 d=d,
                                                 indoor=False,
                                                 rng=rng)

print(f'LoS: {LoS}')

N = 100                                                         # N: Número de componentes multipercurso

amostras = amostras_larga_escala(est_larga_escala=mu_sig_larga_escala,
                                 N=N,
                                 rng=rng)

print(amostras)

################################################
#                                              #
#    2.1 - ATRASOS & POTÊNCIA MULTIPERCURSO    #
#                                              #
################################################

from parametros_larga_escala import (atrasos_multipercurso,
                                     potencia_multipercurso,
                                     plot_potencia_multipercurso)

atrasos, r_tau, indices = atrasos_multipercurso(sigma_tau=amostras['esp_atraso'],
                                                N=N,
                                                LoS=LoS,
                                                indoor=False,
                                                rng=rng)

potencia, ganho = potencia_multipercurso(tau=atrasos,
                                         sigma_tau=amostras['esp_atraso'],
                                         r_tau=r_tau,
                                         K_R=amostras['fator_Rice'],
                                         LoS=LoS,
                                         N=N,
                                         rng=rng)

plot_potencia_multipercurso(alpha_quad=potencia,
                            tau=atrasos,
                            sigma_tau=amostras['esp_atraso'])

###################################
#                                 #
#    2.2 - DIREÇÕES DE CHEGADA    #
#                                 #
###################################

from parametros_larga_escala import (azimute_chegada,
                                     plot_azimute_chegada,
                                     elev_chegada,
                                     plot_elev_chegada,
                                     dir_chegada,
                                     plot_dir_chegada)

phi = azimute_chegada(alpha_quad=potencia,
                      sigma_phi_AoA=amostras['esp_az_chegada'],
                      phi_chegada=ang_chegada[1],
                      LoS=LoS,
                      N=N,
                      rng=rng)

plot_azimute_chegada(alpha_quad=potencia,
                     phi_prime=phi,
                     sigma_phi_AoA=amostras['esp_az_chegada'])

theta = elev_chegada(alpha_quad=potencia,
                     sigma_theta_AoA=amostras['esp_el_chegada'],
                     theta_chegada=ang_chegada[0],
                     LoS=LoS,
                     N=N,
                     rng=rng)

plot_elev_chegada(alpha_quad=potencia,
                  theta_prime=theta,
                  theta_phi_AoA=amostras['esp_el_chegada'])

r_n = dir_chegada(phi_prime=phi,
                  theta_prime=theta)

plot_dir_chegada(r_n, alpha_quad=potencia)

####################################################
#                                                  #
#    2.3 - DESVIO DOPPLER & FASES MULTIPÉRCURSO    #
#                                                  #
####################################################

from parametros_larga_escala import (desvio_doppler,
                                     plot_doppler,
                                     fases_multipercurso,
                                     plot_fases_multipercurso)

nu_n = desvio_doppler(f_GHz=3,
                      r_n=r_n,
                      v_rx=v_rx)

plot_doppler(alpha_quad=potencia,
             nu_n=nu_n)

phi_n_bar = fases_multipercurso(f_GHz=3,                        # Parte estática
                                nu_n=nu_n[indices],
                                tau=atrasos)

plot_fases_multipercurso(nu_n[indices], phi_n_bar)

##################################
#                                #
#    3 -ESPALHAMENTO TEMPORAL    #
#                                #
##################################

from espalhamento_temporal import (init_pulso,
                                   transmite_pulso,
                                   plot_pulso_recebido)

pulso, espectro_pulso = init_pulso(delta_t=1e-7,
                                   N_amostras=10000) # Equivalente banda base

pulso_transmitido = transmite_pulso(delta_t=1e-7,
                                    alpha_quad=potencia,
                                    phi_n_bar=phi_n_bar,
                                    nu_n=nu_n[indices],
                                    tau=atrasos,
                                    N_amostras=10000)

plot_pulso_recebido(pulso_transmitido,
                       delta_t=1e-7,
                       sigma_tau=amostras['esp_atraso'],
                       N_amostras=10000)

#########################################
#                                       #
#    4 - BANDA & TEMPO DE COERÊNCUIA    #
#                                       #
#########################################

from banda_tempo_coerencia import (autocorrelacao_canal_norm,
                                   banda_de_coerencia,
                                   tempo_de_coerencia)

banda_de_coerencia(alpha_quad=potencia, atrasos=atrasos, nu_n=nu_n[indices], sigma_tau=amostras['esp_atraso'])

tempo_de_coerencia(alpha_quad=potencia, atrasos=atrasos, nu_n=nu_n[indices])