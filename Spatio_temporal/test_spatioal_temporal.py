import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

fig_path = "/Users/yaolin/HomeOffice/GOOGLE/fig/Sim_Temporal/"

N = 25
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
xv = xx.flatten()
yv = yy.flatten()
# plt.plot(xv, yv, 'k.')
# plt.show()
grid = np.vstack((xv, yv)).T

dist = cdist(grid, grid)
sigma = .1
phi = 4.5 / .5
Sigma_prior = sigma ** 2 * (1 + phi * dist) * np.exp(-phi * dist)
mu_prior = (xv - .5) ** 2 + (yv - .5) ** 2
# plt.scatter(xv, yv, c=mu_prior, s=150, cmap="BrBG")
# plt.colorbar()
# plt.show()
# plt.imshow(Sigma)
# plt.colorbar()
# plt.show()
rho = .9
time_steps = 15
mu_prior_over_time = np.zeros([time_steps, xv.shape[0]])
Sigma_over_time = Sigma_prior
# print((np.linalg.cholesky(Sigma_over_time) @ np.random.randn(xv.shape[0]).reshape(-1, 1)).shape)
# mu_prior_over_time[0, :] = mu_prior



for i in range(1, time_steps):
    plt.figure()
    plt.subplot(121)
    plt.scatter(xv, yv, c=mu_prior_over_time[i-1, :], cmap="BrBG", s=180)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(Sigma_over_time)
    # plt.scatter(xv, yv, c=np.diag(Sigma_over_time))
    plt.colorbar()
    plt.savefig(fig_path + "AR1/P_{:03d}.jpg".format(i))
    plt.close("all")
    print(i)

    # update
    mu
    Sigma_over_time = (1 - rho**2) * Sigma_over_time
    mu_prior_over_time[i, :] = (mu_prior + rho * (mu_prior_over_time[i-1, :] - mu_prior) +
                                np.linalg.cholesky(Sigma_over_time) @ np.random.randn(xv.shape[0]))






