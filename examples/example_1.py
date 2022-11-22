import numpy as np
import pythia
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

def model(X: np.array) -> np.array:
    tau = 1.2
    a = 7
    y = []

    diff = X[:,0] - X[:,1]
    ind1 = np.abs(diff) < tau
    ind2 = (diff >= tau)
    ind3 = (-diff >= tau)

    y = X[:,0]*X[:,1] + X[:,0]*X[:,2]
    if np.sum(ind2) > 0:
        y[ind2] = y[ind2] - (a*(X[ind2,0] - X[ind2,1])**2 - a*tau**2)
    if np.sum(ind3) > 0:
        y[ind3] = y[ind3] + (a*(X[ind3,0] - X[ind3,1])**2 - a*tau**2)
    return y


def model_jac(X: np.array) -> np.array:
    h= 1e-5
    y = []

    # for x1
    Xplus = copy.deepcopy(X)
    Xplus[:,0] += h
    Xminus = copy.deepcopy(X)
    Xminus[:,0] -= h
    y1 = (model(Xplus) - model(Xminus))/h/2

    # for x2
    Xplus = copy.deepcopy(X)
    Xplus[:,1] += h
    Xminus = copy.deepcopy(X)
    Xminus[:,1] -= h
    y2 = (model(Xplus) - model(Xminus))/h/2

    # for x2
    Xplus = copy.deepcopy(X)
    Xplus[:,2] += h
    Xminus = copy.deepcopy(X)
    Xminus[:,2] -= h
    y3 = (model(Xplus) - model(Xminus))/h/2

    y = np.stack((y1, y2, y3), axis=-1)
    return y


def generate_samples(N: int, samples_range) -> np.array:
    """Generates N samples

    :param N: nof samples
    :returns: (N, D)

    """
    # tmp1 = np.random.uniform(0, samples_range/4, size=int(N/2))
    # tmp2 = np.random.uniform(samples_range*3/4, samples_range, size=int(N/2))
    # x1 = np.concatenate([tmp1, tmp2])
    std = .1
    x1 = np.random.normal(1.5, std, size=int(N/5))
    x2 = np.random.normal(3., std, size=int(N/5))
    x3 = np.random.normal(5, std, size=int(N/5))
    x4 = np.random.normal(6.3, std, size=int(N/5))
    x5 = np.random.normal(8.2, std, size=int(N/5))
    # x1 = np.random.uniform(0, samples_range, size=N-2)
    x1 = np.concatenate([np.zeros(int(1)),
                         x1,
                         2*np.ones(1),
                         x2,
                         4.2*np.ones(1),
                         x3,
                         x4,
                         7.1*np.ones(1),
                         x5,
                         9.1*np.ones(1),
                         np.ones(int(1))*samples_range])
    x2 = x1 + np.random.normal(size=(x1.shape[0]))*.6
    x3 = np.random.normal(size=(x1.shape[0]))*20
    return np.stack([x1, x2, x3]).T


def plot_model(model, samples, nof_points, samples_range, savefig):
    x = np.linspace(-.1*samples_range, 1.1*samples_range, nof_points)
    y = np.linspace(-.1*samples_range, 1.1*samples_range, nof_points)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    z = model(np.concatenate([positions, np.zeros((positions.shape[0], 1))], axis=-1))
    zz = np.reshape(z, [x.shape[0], y.shape[0]])
    fig, ax = plt.subplots()
    cs = ax.contourf(xx, yy, zz, levels=400, vmin=-100, vmax=200., cmap=cm.viridis, extend='both')
    ax.plot(samples[:, 0], samples[:, 1], 'ro', label="samples")
    ax.plot(np.linspace(0, 10, 10), np.linspace(0, samples_range, 10), "r--")
    plt.title(r"$f(x_1, x_2, x_3=0)$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.colorbar(cs)
    if savefig:
        tplt.save("./../paper/images/case-2-f-gt.tex")
    plt.show(block=False)


def generate_samples(N: int, samples_range) -> np.array:
    """Generates N samples

    :param N: nof samples
    :returns: (N, D)

    """
    # tmp1 = np.random.uniform(0, samples_range/4, size=int(N/2))
    # tmp2 = np.random.uniform(samples_range*3/4, samples_range, size=int(N/2))
    # x1 = np.concatenate([tmp1, tmp2])
    std = .1
    x1 = np.random.normal(1.5, std, size=int(N/5))
    x2 = np.random.normal(3., std, size=int(N/5))
    x3 = np.random.normal(5, std, size=int(N/5))
    x4 = np.random.normal(6.3, std, size=int(N/5))
    x5 = np.random.normal(8.2, std, size=int(N/5))
    # x1 = np.random.uniform(0, samples_range, size=N-2)
    x1 = np.concatenate([np.zeros(int(1)),
                         x1,
                         2*np.ones(1),
                         x2,
                         4.2*np.ones(1),
                         x3,
                         x4,
                         7.1*np.ones(1),
                         x5,
                         9.1*np.ones(1),
                         np.ones(int(1))*samples_range])
    x2 = x1 + np.random.normal(size=(x1.shape[0]))*.6
    x3 = np.random.normal(size=(x1.shape[0]))*20
    return np.stack([x1, x2, x3]).T


N = 1000
X = generate_samples(N, samples_range)
samples_range = 10
plot_model(model=model, samples=X, nof_points=15, samples_range=samples_range, savefig=False)


# DALE
dale = pythia.DALE(data=X, model=model, model_jac=model_jac)
dale.fit(features=0, alg_params={"method": "fixed", "nof_bins": 10})
dale.plot(s=0)
