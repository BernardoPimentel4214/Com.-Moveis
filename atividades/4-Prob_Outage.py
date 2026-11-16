import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import distributions

fig, axs = plt.subplots(2, 3)

for K_r in [0, 5, 20]:   # 21 dB deu NaN

    '''
        Curvas exatas
    '''
    # Função Marcum Q para a P_out com desvanecimento Rayleigh
    def Q(nu, k, x, A=1):
        return distributions.ncx2.cdf(df=nu, nc=2*k, x=x)

    # Conversão dB -> linear
    def linear(x):
        return 10**(x/10)

    # Probabilidades de outage:
    P_out_Rayleigh = lambda th, media: 1 - np.exp(-th/media)
    P_out_Rice     = lambda th, media, K: Q(nu=2, x=2*(K + 1)*th/media, k=K)

    '''
        Curvas Geradas com amostras
    '''
    # Número de amostras
    N = 10_000

    # Distribuição Rayleigh e Rice da Envoltória do Canal com E[β^2] = 1
    mu = 0
    sigma = np.sqrt(1/2)
    beta_Rayleigh = np.sqrt(np.random.normal(0, sigma, N)**2 + np.random.normal(0, sigma, N)**2)

    mu = np.sqrt(K_r/(K_r + 1))
    sigma = np.sqrt(1/(2 * (K_r + 1)))
    beta_Rice = np.sqrt((np.random.normal(mu, sigma, N))**2 + (np.random.normal(0, sigma, N))**2)

    # Check de E[β^2] = 1
    print(rf'$Rayleigh\;\;E[\beta^2] = {np.mean(beta_Rayleigh**2)}$')
    print(rf'$Rice\;\;E[\beta^2] = {np.mean(beta_Rice**2)}$')

    # SNR = SNR_med * β^2
    SNR = np.linspace(10**(-30/10), 10**(30/10), N)
    #SNR = np.logspace((-30/10), 30/10, N)
        
    # Curvas exatas
    curvas_Rayleigh_exatas = []
    curvas_Rice_exatas = []

    curvas_Rayleigh_amostradas = []
    curvas_Rice_amostradas = []

    for media in [-20, 0, 20]:
        curvas_Rayleigh_exatas.append(P_out_Rayleigh(SNR, linear(media)))
        curvas_Rice_exatas.append(P_out_Rice(SNR, linear(media), K_r))

        P_out_Rayleigh_amostrada = []
        SNR_Rayleigh = linear(media)*beta_Rayleigh**2
        P_out_Rice_amostrada = []
        SNR_Rice = linear(media)*beta_Rice**2

        for th in SNR:
            N_s = len(SNR[np.where(SNR_Rayleigh < th)])
            P_out_Rayleigh_amostrada.append(N_s/N)

            N_s = len(SNR[np.where(SNR_Rice < th)])
            P_out_Rice_amostrada.append(N_s/N)

        curvas_Rayleigh_amostradas.append(P_out_Rayleigh_amostrada)
        curvas_Rice_amostradas.append(P_out_Rice_amostrada)

    # Plot
    m = [-20, 0, 20]
    if(K_r == 0):
        for i in range(3):
            axs[0][0].semilogy(10*np.log10(SNR), curvas_Rayleigh_amostradas[i], color ='C'+ str(i) , linewidth=3, linestyle ='-', label=rf'$\hat\gamma_s = {m[i]}$')
            axs[0][0].semilogy(10*np.log10(SNR), curvas_Rice_amostradas[i], color ='C'+ str(i) , linewidth=3, linestyle ='--')
            axs[0][0].set_title(r'$K_r = 0 dB$')

            axs[1][0].semilogy(10*np.log10(SNR), curvas_Rayleigh_exatas[i], color ='C'+ str(i) , linewidth=3, linestyle ='-', label=rf'$\hat\gamma_s = {m[i]}$')
            axs[1][0].semilogy(10*np.log10(SNR), curvas_Rice_exatas[i], color ='C'+ str(i) , linewidth=3, linestyle ='--')
            axs[0][0].semilogy(10*np.log10(SNR), curvas_Rayleigh_exatas[i], color ='k', linewidth=3, linestyle ='-.', alpha=0.3)
            axs[0][0].semilogy(10*np.log10(SNR), curvas_Rice_exatas[i], color ='k' , linewidth=3, linestyle ='-.', alpha=0.3)

    if(K_r == 5):
        for i in range(3):
            axs[0][1].semilogy(10*np.log10(SNR), curvas_Rayleigh_amostradas[i], color ='C'+ str(i) , linewidth=3, linestyle ='-', label=rf'$\hat\gamma_s = {m[i]}$')
            axs[0][1].semilogy(10*np.log10(SNR), curvas_Rice_amostradas[i], color ='C'+ str(i) , linewidth=3, linestyle ='--')
            axs[0][1].set_title(r'$K_r = 5 dB$')

            axs[1][1].semilogy(10*np.log10(SNR), curvas_Rayleigh_exatas[i], color ='C'+ str(i) , linewidth=3, linestyle ='-', label=rf'$\hat\gamma_s = {m[i]}$')
            axs[1][1].semilogy(10*np.log10(SNR), curvas_Rice_exatas[i], color ='C'+ str(i) , linewidth=3, linestyle ='--')
            axs[0][1].semilogy(10*np.log10(SNR), curvas_Rayleigh_exatas[i], color ='k', linewidth=3, linestyle ='-.', alpha=0.3)
            axs[0][1].semilogy(10*np.log10(SNR), curvas_Rice_exatas[i], color ='k' , linewidth=3, linestyle ='-.', alpha=0.3)

    if(K_r == 20):
        for i in range(3):
            axs[0][2].semilogy(10*np.log10(SNR), curvas_Rayleigh_amostradas[i], color ='C'+ str(i) , linewidth=3, linestyle ='-', label=rf'$\hat\gamma_s = {m[i]}$')
            axs[0][2].semilogy(10*np.log10(SNR), curvas_Rice_amostradas[i], color ='C'+ str(i) , linewidth=3, linestyle ='--')
            axs[0][2].set_title(r'$K_r = 20 dB$')

            axs[1][2].semilogy(10*np.log10(SNR), curvas_Rayleigh_exatas[i], color ='C'+ str(i) , linewidth=3, linestyle ='-', label=rf'$\hat\gamma_s = {m[i]}$')
            axs[1][2].semilogy(10*np.log10(SNR), curvas_Rice_exatas[i], color ='C'+ str(i) , linewidth=3, linestyle ='--')
            axs[0][2].semilogy(10*np.log10(SNR), curvas_Rayleigh_exatas[i], color ='k', linewidth=3, linestyle ='-.', alpha=0.3)
            axs[0][2].semilogy(10*np.log10(SNR), curvas_Rice_exatas[i], color ='k' , linewidth=3, linestyle ='-.', alpha=0.3)
        
    for i in range(0, 2):
        for j in range(0, 3):
            axs[i][j].grid(True, which='both', alpha=0.3)
            axs[i][j].set_ylim(1e-5, 2)
            axs[i][j].set_xlabel(r'$\text{Limiar de SNR} - \gamma_th$')
            axs[i][j].set_ylabel('OP', rotation=1)

plt.legend()
plt.show()
