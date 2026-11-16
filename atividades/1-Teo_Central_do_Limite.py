import matplotlib.pyplot as plt
import numpy as np

from numpy.random import uniform


m = int(1e6)
N = [1, 2, 3, 4, 5, 10, 20, 50, 100]
x = np.linspace(-30, 30, 1000)

normal = lambda x, mu, var: np.exp(-(x - mu)**2/(2*var))/np.sqrt(2*np.pi*var)

def TCL(N, m):
    U = np.array([])

    for n in range(N):
        U_n = uniform(-np.pi/2, np.pi/2, m)
        U = np.append(U, np.array([U_n]))

    U = U.reshape(N, m)
    X = np.sum(U, axis=0)

    return X


X = np.array([])
mu_X = np.array([])
var_X = np.array([])

for n in N:
    X_n = TCL(n, m)
    X = np.append(X, X_n)
    mu_X = np.append(mu_X, np.mean(X_n))
    var_X = np.append(var_X, np.var(X_n))
X = X.reshape(len(N), m)

i = 0

def fig1(fig):
    ax = fig.add_subplot(111)
    ax.hist(X[i], 51, density=True, edgecolor='black')
    ax.set_xlabel('Valores')
    ax.set_ylabel('Frequencia')
    plt.title(f'N={N[i]}')

def fig2(fig):
    ax = fig.add_subplot(111)
    ax.hist(X[i], 51, density=True, edgecolor='black', zorder=1)
    ax.plot(x, normal(x, mu_X[i], var_X[i]), linewidth=2, linestyle='--', zorder=100)
    ax.set_xlabel('Valores')
    ax.set_ylabel('Frequencia')
    plt.title(f'N={N[i]}')

switch_figs = {
    0: fig1,
    1: fig1,
    2: fig1,
    3: fig1,
    4: fig1,
    5: fig1,
    6: fig1,
    7: fig2,
    8: fig2
}

def onclick1(fig):
    global i
    print(i)
    fig.clear()
    i += 1
    i %= 9
    switch_figs[i](fig)
    plt.draw()

fig = plt.figure()
switch_figs[0](fig)
fig.canvas.mpl_connect('button_press_event', lambda event: onclick1(fig))

plt.show()