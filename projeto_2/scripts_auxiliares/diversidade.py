import numpy as np

def SC(beta_l, r_l):
    # Índice do ramo com maior envoltória
    idx = np.argmax(beta_l, axis=0)  # (30, 10000)
    r_sc = np.take_along_axis(
           np.array(r_l),
           idx[None, :, :, None],
           axis=0
           ).squeeze(axis=0)
    return r_sc

def MRC(beta_l, phi_l, r_l):
    r_complex = r_l[..., 0] + 1j * r_l[..., 1]
    h = beta_l * np.exp(1j * phi_l)
    r_MRC = np.sum(np.conj(h) * r_complex, axis=0)

    return np.stack((r_MRC.real, r_MRC.imag), axis=-1)

def EGC(phi_l, r_l):
    r_complex = r_l[..., 0] + 1j * r_l[..., 1]
    r_EGC = np.sum(np.exp(-1j * phi_l) * r_complex, axis=0)

    return np.stack((r_EGC.real, r_EGC.imag), axis=-1)