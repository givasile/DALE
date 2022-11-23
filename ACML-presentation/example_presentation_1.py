import numpy as np
import pythia
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

np.random.seed(243454211)


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
    x1 = np.array([0, 0.1, 0.33, 0.47, 0.8, 1.3, 1.6, 2.01, 2.07, 3.8, 4.01, 5.7, 5.9, 6.6, 6.8, 7.12, 7.44, 8.2, 8.3, 8.35, 10])
    x2 = x1 + np.random.normal(size=(x1.shape[0]))*.5
    x3 = (np.random.normal(size=x1.shape[0])) * 4 ## np.random.uniform(size=(x1.shape[0]))*3
    x3[7] = 20
    x3[8] = -10
    x3[9] = -10
    x3[10] = 20
    x3[11] = -10
    x3[12] = -10
    return np.stack([x1, x2, x3]).T


def plot_model(model, samples, nof_points, samples_range, limits, savefig):
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

    if limits is not None:
        ax.vlines(limits, ymin=np.min(y), ymax=np.max(y), linestyles="dashed", label="bins")

    plt.title(r"$f(x_1, x_2, x_3=0)$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.colorbar(cs)
    if savefig:
        plt.savefig(savefig)
    plt.show(block=False)


N = 1000
samples_range = 10
X = generate_samples(N, samples_range)
plot_model(model=model, samples=X, nof_points=15, samples_range=samples_range, limits=None, savefig=False)


for nof_bins in [3, 5, 10, 20, 40, 100]:

    # DALE
    dale = pythia.DALE(data=X, model=model, model_jac=model_jac)
    dale.fit(features=0, params={"method": "fixed", "nof_bins": nof_bins})
    dale.plot(s=0, error=False, savefig="./figures/dale_" + str(nof_bins) + "_bins.pdf")
    dale.plot(s=0, error=False, savefig="./figures/dale_" + str(nof_bins) + "_bins.png")

    # ALE
    ale = pythia.ALE(data=X, model=model)
    ale.fit(features=0, params={"nof_bins": nof_bins})
    ale.plot(s=0, error=False, savefig="./figures/ale_" + str(nof_bins) + "_bins.pdf")
    ale.plot(s=0, error=False, savefig="./figures/ale_" + str(nof_bins) + "_bins.png")

    plot_model(model=model, samples=X, nof_points=40, samples_range=samples_range,
               limits=dale.feature_effect["feature_0"]["limits"],
               savefig="./figures/bin_splitting_" + str(nof_bins) + "_bins.png")
    plot_model(model=model, samples=X, nof_points=40, samples_range=samples_range,
               limits=dale.feature_effect["feature_0"]["limits"],
               savefig="./figures/bin_splitting_" + str(nof_bins) + "_bins.pdf")



#
# # DALE
# dale = pythia.DALE(data=X, model=model, model_jac=model_jac)
# dale.fit(features=0, params={"method": "fixed", "nof_bins": 5})
# dale.plot(s=0, error=False)
# plot_model(model=model, samples=X, nof_points=40, samples_range=samples_range,
#            limits=dale.feature_effect["feature_0"]["limits"], savefig=False)
#
#
#
# # ALE
# ale = pythia.ALE(data=X, model=model)
# ale.fit(features=0, params={"nof_bins": 10})
# ale.plot(s=0, error=False)
#
