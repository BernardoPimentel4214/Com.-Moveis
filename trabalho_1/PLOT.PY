import numpy as np  
import matplotlib.pyplot as plt

'''
    Nese arquivo, é codificada a geometria básica do cenário simulado. O ambiente designado foi 'Urban Microcell'.
    Primeiramente, deve ser definido:
        ◦ Altura da Base Station (BS) - h_tx
        ◦ Altura do User Terminal (UT) - h_rx
        ◦ Distância BS-UT (2D) - d
        ◦ Velocidade da estação móvel - v_rx

    De acordo com o relatório 3GPP TR 38.901 e 3GPP TR 36.873, h_tx = 10m e h_rx = 3(N_fl - 1) + 1.5, onde n_fl
    é igual a 1 para uma UE outdoor e n_fl ~ U(1, N_fl) com N_fl ~ U(4, 8) para uma UE indoor.
'''


def init_geometria(seed, indoor):
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

    # a UE se movimenta no plano xy a 3m/s. Sua direção será definida perpendicular ao vetor de raio BS UE no plano xy.
    r = (UE[0] - BS [0], UE[1] - BS[1], 0)
    v_rx = 3 * np.cross(r, (0, 0, 1)) / np.linalg.vector_norm(np.cross(r, (0, 0, 1)))

    return BS, UE, d, v_rx


def plot_geometria(BS, UE, d, v_rx):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(BS[0], BS[1], BS[2], label='BS', marker='^', s=200)
    ax.scatter3D(UE[0], UE[1], UE[2], label='UE', s=144)
    ax.quiver(UE[0], UE[1], UE[2], 3*v_rx[0], 3*v_rx[1], 3*v_rx[2],
              color='k', arrow_length_ratio=0.4, label=r'$v_{rx}$')

    ax.plot([BS[0], BS[0]], [BS[1], BS[1]], [0, BS[2]], color='C0', linewidth=2)
    ax.plot([UE[0], UE[0]], [UE[1], UE[1]], [0, UE[2]], color='C1', linewidth=2)
    ax.plot([BS[0], UE[0]], [BS[1], UE[1]], [0, 0],
             color='k', linestyle='--', label=fr'$d={d:.2f}\;m$')
    ax.legend()

    ax.set_zlim(0, 20)
    ax.view_init(14, 162)
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

    return (theta_LoS, phi_LoS), (phi_LoS, phi_prime_LoS)


def add_angle_annotations_bs(ax, BS, target_point, theta_deg, phi_deg, radius=4.0):
    x0, y0, z0 = BS
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # 1. AZIMUTH (phi) on Ground Plane (Z = 0) measured from +Y Axis
    # Reference line along +Y direction
    ax.plot([x0, x0], [y0, y0 + radius * 1.5], [0, 0], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Projection line on ground plane (swapped sin/cos for Y-axis reference)
    x_proj = x0 + radius * 1.5 * np.sin(phi)
    y_proj = y0 + radius * 1.5 * np.cos(phi)
    ax.plot([x0, x_proj], [y0, y_proj], [0, 0], color='gray', linestyle=':', linewidth=3)

    # Phi Arc (sweeps from +Y axis)
    phi_steps = np.linspace(0, phi, 30)
    arc_phi_x = x0 + (radius * 0.8) * np.sin(phi_steps)
    arc_phi_y = y0 + (radius * 0.8) * np.cos(phi_steps)
    arc_phi_z = np.zeros_like(phi_steps)
    ax.plot(arc_phi_x, arc_phi_y, arc_phi_z, color='crimson', linewidth=3)

    # Phi Label at midpoint
    mid_phi = phi / 2.0
    ax.text((x0 + radius * np.sin(mid_phi))*2, 2*(y0 + radius * np.cos(mid_phi)), 0,
            r'$\phi_{LoS}$', color='crimson', fontsize=20, fontweight='bold')

    # 2. ELEVATION (theta) at BS Point (Interpolated to LoS vector)
    v = np.array(target_point) - np.array(BS)
    v_hat = v / np.linalg.norm(v)
    z_hat = np.array([0, 0, -1 if BS[2] > target_point[2] else 1])
    
    arc_points = []
    for t in np.linspace(0, 1, 30):
        vec = (1 - t) * z_hat + t * v_hat
        vec_norm = vec / np.linalg.norm(vec)
        arc_points.append(np.array(BS) + radius * vec_norm)
    arc_points = np.array(arc_points)

    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color='darkblue', linewidth=1.8)

    # Theta Label at arc midpoint
    mid_pt = arc_points[15]
    ax.text(mid_pt[0], mid_pt[1], mid_pt[2], r'$180^o-\theta_{LoS}$', color='darkblue', fontsize=11, fontweight='bold')


def add_angle_annotations_ue(ax, UE, target_point, theta_deg, phi_deg, radius=4.0):
    x0, y0, z0 = UE
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # 1. AZIMUTH (phi) on Ground Plane (Z = 0) measured from +Y Axis
    ax.plot([x0, x0], [y0, y0 + radius * 1.5], [0, 0], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    x_proj = x0 + radius * 1.5 * np.sin(phi)
    y_proj = y0 + radius * 1.5 * np.cos(phi)
    ax.plot([x0, x_proj], [y0, y_proj], [0, 0], color='gray', linestyle=':', linewidth=3)

    phi_steps = np.linspace(0, phi, 30)
    arc_phi_x = x0 + (radius * 0.8) * np.sin(phi_steps)
    arc_phi_y = y0 + (radius * 0.8) * np.cos(phi_steps)
    arc_phi_z = np.zeros_like(phi_steps)
    ax.plot(arc_phi_x, arc_phi_y, arc_phi_z, color='crimson', linewidth=1.8)

    mid_phi = phi / 2.0
    ax.text(x0 + radius * np.sin(mid_phi), y0 + radius * np.cos(mid_phi), 0,
            r'$\phi´_{LoS}$', color='crimson', fontsize=20, fontweight='bold')

    # 2. ELEVATION (theta) at UE Point
    v = np.array(target_point) - np.array(UE)
    v_hat = v / np.linalg.norm(v)
    z_hat = np.array([0, 0, 1 if target_point[2] > z0 else -1])

    arc_points = []
    for t in np.linspace(0, 1, 30):
        vec = (1 - t) * z_hat + t * v_hat
        vec_norm = vec / np.linalg.norm(vec)
        arc_points.append(np.array(UE) + radius * vec_norm)
    arc_points = np.array(arc_points)

    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color='darkblue', linewidth=3)

    mid_pt = arc_points[15]
    ax.text(mid_pt[0], mid_pt[1], mid_pt[2], r'$\theta´_{LoS}$', color='darkblue', fontsize=11, fontweight='bold')


def plot_angulos(BS, UE, ang_saida, ang_chegada):
    fig = plt.figure(figsize=(14, 7))
    
    theta_tx, phi_tx = ang_saida
    theta_rx, phi_rx = ang_chegada

    # ------------------ Subplot 1: BS ------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter3D(BS[0], BS[1], BS[2], label='BS', marker='^', s=200)
    ax1.plot([BS[0], BS[0]], [BS[1], BS[1]], [0, BS[2]], color='C0', linewidth=2)
    
    ax1.quiver(BS[0], BS[1], BS[2],
               0.2*(UE[0] - BS[0]), 0.2*(UE[1] - BS[1]), 0.2*(UE[2] - BS[2]),
               color='k', arrow_length_ratio=0.1, label=r'$v_{tx}$')

    add_angle_annotations_bs(ax1, BS, UE, theta_tx, phi_tx, radius=4.0)

    ax1.set_xlim3d(-10, 10)
    ax1.set_ylim3d(-10, 10)
    ax1.set_zlim3d(0, 15)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(18, 80)
    ax1.legend(loc='upper right')

    # ------------------ Subplot 2: UE ------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter3D(UE[0], UE[1], UE[2], label='UE', color='C1', s=200)
    ax2.plot([UE[0], UE[0]], [UE[1], UE[1]], [0, UE[2]], color='C1', linewidth=2)
    ax2.plot([UE[0], UE[0]], [UE[1], UE[1]], [0, 4*UE[2]], color='C1', linewidth=1, linestyle='--')
    
    ax2.quiver(UE[0], UE[1], UE[2],
               0.6*(BS[0] - UE[0]), 0.6*(BS[1] - UE[1]), 0.6*(BS[2] - UE[2]),
               color='k', arrow_length_ratio=0.1, label=r'$v_{rx}$')

    add_angle_annotations_ue(ax2, UE, BS, theta_rx, phi_rx, radius=4.0)

    ax2.set_xlim3d(UE[0] - 20, UE[0] + 20)
    ax2.set_ylim3d(UE[1] - 20, UE[1] + 20)
    ax2.set_zlim3d(0, 15)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(13, 103)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


BS, UE, d, v_rx = init_geometria(4214, False)
plot_geometria(BS, UE, d, v_rx)

ang_saida, ang_chegada = calcula_angulos(BS, UE)
plot_angulos(BS, UE, ang_chegada=ang_chegada, ang_saida=ang_saida)