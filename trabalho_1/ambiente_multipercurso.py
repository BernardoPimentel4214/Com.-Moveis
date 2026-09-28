import numpy as np  
import matplotlib.pyplot as plt

'''
    Nese arquivo, é codificada a geometria básica do cenário simulado. O ambiente designado foi 'Urban Microcell' (UMi).
    Primeiramente, deve ser definido:
        ◦ Altura da Base Station (BS) - h_tx
        ◦ Altura do User Terminal (UT) - h_rx
        ◦ Distância BS-UT (2D) - d
        ◦ Velocidade da estação móvel - v_rx

    De acordo com o relatório 3GPP TR 38.901 e 3GPP TR 36.873, h_tx = 10m e h_rx = 3(N_fl - 1) + 1.5, onde n_fl
    é igual a 1 para uma UE outdoor e n_fl ~ U(1, N_fl) com N_fl ~ U(4, 8) para uma UE indoor.
'''


def init_geometria(indoor, seed):
    rng = np.random.default_rng(seed)

    # Espaço cartesiano onde ficam a UE e a BS. A 'intersite distance' é definida como 200m, então essa será a dimensão do espaço.
    l, N = 100, 1000
    x = np.linspace(-l, l, N)
    y = np.linspace(-l, l, N)
    xi, yi = np.meshgrid(x, y)

    # Colocamos a BS sob a origem do plano xy por simplicidade.
    h_tx = 10
    BS = (0, 0, h_tx)

    # a UE será posicionado aleatoriamente. A distância mínima entre a UE e a BS é 10m no relatório:
    d = 0
    while(not(d >= 10 and d <= 100)):
        UE_x = rng.choice(x)            # Verificar depois se isso não pode dar um loop infinito
        UE_y = rng.choice(y)
        d = np.sqrt(UE_x**2 + UE_y**2)

    # Altura da UE:
    h_UE = lambda h: 3*(h - 1) + 1.5
    if (not indoor):
        n_fl = 1
    else:
        N_fl = np.random.uniform(4, 8)
        n_fl = np.random.uniform(1, N_fl)
    h_rx = h_UE(n_fl)
    UE = (UE_x, UE_y, h_rx)

    # a UE se movimenta no plano xy a 3km/h. Sua direção será definida perpendicular ao vetor de raio BS UE no plano xy.
    r = (UE[0] - BS [0], UE[1] - BS[1], 0)
    v_rx = 0.833333 * np.cross(r, (0, 0, 1)) / np.linalg.vector_norm(np.cross(r, (0, 0, 1)))

    return BS, UE, d, v_rx, rng


def plot_geometria(BS, UE, d, v_rx):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(BS[0], BS[1], BS[2], label='BS', marker='^', s=200)
    ax.scatter3D(UE[0], UE[1], UE[2], label='UE', s=144)
    ax.quiver(UE[0], UE[1], UE[2], 3*v_rx[0], 3*v_rx[1], 3*v_rx[2],
              color='k', arrow_length_ratio=0.1, label=r'$v_{rx}$')

    ax.plot([BS[0], BS[0]], [BS[1], BS[1]], [0, BS[2]], color='C0', linewidth=2)
    ax.plot([UE[0], UE[0]], [UE[1], UE[1]], [0, UE[2]], color='C1', linewidth=2)
    ax.plot([BS[0], UE[0]], [BS[1], UE[1]], [0, 0],
             color='k', linestyle='--', label=fr'$d={d:.2f}\;m$')
    ax.set_zlim(0, 20)
    ax.view_init(14, 162)

    ax.legend()

    plt.show()


'''
    Em seguida, devem ser determinados os ângulos de visada direta:
        ◦ Azimute de saída: ϕ_LoS ∈ [−180◦, 180◦]
        ◦ Azimute de chegada: ϕ'_LoS ∈ [−180◦, 180◦]
        ◦ Elevação de saída: θ_LoS ∈ [−90◦, 90◦]
        ◦ Elevação de chegada: θ'_LoS ∈ [−90◦, 90◦]
    
    Esses ângulos são definidos assim como ϕ e θ de coordenadas esféricas usuais, logo podem ser obtidos
    considerando a BS na origem para (ϕ_LoS, θ_LoS) e a UE na origem para (ϕ'_LoS, θ'_LoS). Não será necessária
    a conversão entre sistemas de coordaenadass locais LCS e globais GCS.
'''


def calcula_angulos(BS, UE):
    # Coordenadas da UE com a BS na origem:
    UE_x = UE[0] - BS[0]
    UE_y = UE[1] - BS[1]
    UE_z = UE[2] - BS[2]

    theta_LoS = np.arctan2(np.sqrt(UE_x**2 + UE_y**2), UE_z)
    phi_LoS = np.arctan2(UE_y, UE_x)

    # Coordenadas da BS com UE na origem:
    BS_x = BS[0] - UE[0]
    BS_y = BS[1] - UE[1]
    BS_z = BS[2] - UE[2]

    theta_prime_LoS = np.arctan2(np.sqrt(BS_x**2 + BS_y**2), BS_z)
    phi_prime_LoS = np.arctan2(BS_y, BS_x)

    # Conversão para graus
    theta_LoS = np.rad2deg(theta_LoS)
    theta_prime_LoS = np.rad2deg(theta_prime_LoS)
    phi_LoS = np.rad2deg(phi_LoS)
    phi_prime_LoS = np.rad2deg(phi_prime_LoS)

    return (theta_LoS, phi_LoS), (theta_prime_LoS, phi_prime_LoS)
