import matplotlib.pyplot as plt
import numpy as np

from usr_func import *

working_directory = os.getcwd()

fig_path = working_directory + "/GOOGLE/fig/Sim_Temporal/"


N = 25
XLIM = (0, 1)
YLIM = (0, 1)
xv = np.linspace(XLIM[0], XLIM[1], N)
yv = np.linspace(YLIM[0], YLIM[1], N)
xx, yy = np.meshgrid(xv, yv)
x = xx.flatten()
y = yy.flatten()
n = x.shape[0]
# plt.plot(xv, yv, 'k.')
# plt.show()
grid = np.vstack((x, y)).T

dist = cdist(grid, grid)
sigma = .1
phi = 4.5 / .5
Sigma_prior = sigma ** 2 * (1 + phi * dist) * np.exp(-phi * dist)
mu_prior = vectorise((x - .5) ** 2 + (y - .5) ** 2)
mu_truth = vectorise(np.linalg.cholesky(Sigma_prior) @ np.random.randn(n))

time_steps = 100
mu_time = np.zeros((time_steps, n))
mu_time[0, :] = mu_prior.flatten()
rho = .99


tau = .078
mu_cond = np.zeros_like(mu_time)
Sigma_cond = np.zeros([time_steps, n, n])
mu_cond[0, :] = mu_prior.flatten()
Sigma_cond[0, :, :] = Sigma_prior

ind_random = np.random.randint(0, n, time_steps)

# for i in range(time_steps):
#     if i == 0:
#         continue
#
#
#     # == plot
#     N_plot = 100
#     xp = np.linspace(XLIM[0], XLIM[1], N_plot)
#     yp = np.linspace(YLIM[0], YLIM[1], N_plot)
#     grid_x, grid_y = np.meshgrid(xp, yp)
#     grid_values = griddata(grid, mu_time[i, :], (grid_x, grid_y), method='cubic')
#
#     plt.figure()
#     plt.scatter(grid_x, grid_y, c=grid_values, cmap="RdBu", vmin=0, vmax=.5)
#     plt.colorbar()
#     plt.xlim(XLIM)
#     plt.ylim(YLIM)
#     plt.title("Prior at time step: " + str(i))
#     plt.savefig(fig_path+"AR1/Prior/P_{:03d}.jpg".format(i))
#     plt.close('all')
#     print(i)


for i in range(time_steps):
    if i == 0:
        pass
    else:
        random_samples = vectorise(np.linalg.cholesky((1 - rho ** 2) * Sigma_prior) @ np.random.randn(n))
        mu_time[i, :] = (mu_prior + rho * (vectorise(mu_time[i - 1, :]) - mu_prior) + random_samples).flatten()

        mu_t_t_1 = mu_prior + rho * (vectorise(mu_cond[i-1, :]) - mu_prior)
        S_t_t_1 = rho**2 * Sigma_cond[i-1, :, :] + (1-rho**2) * Sigma_prior

        f_t = np.zeros([n, 1])
        f_t[ind_random[i]] = True
        # y_t = f_t.T @ mu_truth
        y_t = f_t.T @ vectorise(mu_time[i, :]) + np.linalg.cholesky(np.diagflat(tau**2)) @ vectorise(np.random.randn())

        # print(np.linalg.solve(f_t.T @ S_t_t_1 @ f_t + np.diagflat(tau ** 2), y_t - f_t.T @ mu_t_t_1))

        # print("here")
        mu_cond[i, :] = (mu_t_t_1 + S_t_t_1 @ f_t @ np.linalg.solve(f_t.T@S_t_t_1@f_t+np.diagflat(tau**2),
                                                                    y_t - f_t.T @ mu_t_t_1)).flatten()
        # print("here")
        # print(np.linalg.solve(f_t.T@S_t_t_1@f_t+np.diagflat(tau**2),f_t.T@S_t_t_1))
        Sigma_cond[i, :, :] = S_t_t_1 - S_t_t_1 @ f_t @ np.linalg.solve(f_t.T@S_t_t_1@f_t+np.diagflat(tau**2),
                                                                        f_t.T@S_t_t_1)
        # print("here")

    # == plot
    N_plot = 100
    xp = np.linspace(XLIM[0], XLIM[1], N_plot)
    yp = np.linspace(YLIM[0], YLIM[1], N_plot)
    grid_x, grid_y = np.meshgrid(xp, yp)
    grid_mu = griddata(grid, mu_cond[i, :], (grid_x, grid_y), method='cubic')
    grid_mu_prior = griddata(grid, mu_time[i, :], (grid_x, grid_y), method='cubic')
    # grid_mu_truth = griddata(grid, mu_truth, (grid_x, grid_y), method='cubic')
    grid_variance = griddata(grid, np.diag(Sigma_cond[i, :, :]), (grid_x, grid_y), method='cubic')

    fig = plt.figure(figsize=(40, 10))
    gs = GridSpec(nrows=1, ncols=3)
    # ax = fig.add_subplot(gs[0])
    # im = ax.scatter(grid_x, grid_y, c=grid_mu_truth, cmap="RdBu", vmin=0, vmax=.3)
    # plt.colorbar(im)
    # plt.xlim(XLIM)
    # plt.ylim(YLIM)
    # plt.title("Truth")

    ax = fig.add_subplot(gs[0])
    im = ax.scatter(grid_x, grid_y, c=grid_mu_prior, cmap="RdBu", vmin=0, vmax=.5)
    plt.colorbar(im)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    plt.title("Prior at time step: "+str(i))

    ax = fig.add_subplot(gs[1])
    im = ax.scatter(grid_x, grid_y, c=grid_mu, cmap="RdBu", vmin=0, vmax=.5)
    plt.colorbar(im)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    plt.title("mu_cond at time step: " + str(i))

    ax = fig.add_subplot(gs[2])
    im = ax.scatter(grid_x, grid_y, c=grid_variance, cmap="BrBG", vmin=0, vmax=0.015)
    plt.colorbar(im)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    plt.title("Prediction error at time step: " + str(i))

    plt.savefig(fig_path+"AR1/P_{:03d}.jpg".format(i))
    plt.close('all')
    print(i)








