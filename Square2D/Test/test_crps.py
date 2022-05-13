import matplotlib.pyplot as plt

from usr_func import *
NUM = 25
x = np.linspace(0, 1, NUM)
y = np.linspace(0, 1, NUM)
xx, yy = np.meshgrid(x, y)
xv, yv = map(vectorise, [xx, yy])
grid = np.hstack((xv, yv))
from scipy.spatial.distance import cdist
dist = cdist(grid, grid)
sigma = 1
eta = 4.5 / .5
tau = np.sqrt(.3)
R = tau ** 2
Sigma = sigma ** 2 * (1 + eta * dist) * np.exp(-eta * dist)
plt.imshow(Sigma)
plt.colorbar()
plt.show()



mean = np.zeros(len(grid))
mean = vectorise(mean)
true = mean + np.linalg.cholesky(Sigma) @ vectorise(np.random.randn(len(grid)))
F = getFVector(1, mean.shape[0])
mu_cond, Sigma_cond = update_GP_field(mean, Sigma, F, R, F@true)

CRPS = []
for i in range(len(mean)):
    F = getFVector(i, mean.shape[0])
    # mu_cond, Sigma_cond = update_GP_field(mean, Sigma, F, R, F@true)
    CRPS.append(get_crps_1d(F @ true, mean[i], Sigma[i, i].reshape(-1, 1)))
CRPS = np.array(CRPS)
CRPS_new = get_crps_1d(true, mean, Sigma)

ms = 400
plt.figure(figsize=(30, 8))
plt.subplot(141)
plt.scatter(grid[:, 0], grid[:, 1], c=true, s=ms)
plt.colorbar()
plt.subplot(142)
plt.scatter(grid[:, 0], grid[:, 1], c=CRPS, s=ms)
plt.colorbar()
plt.subplot(143)
plt.scatter(grid[:, 0], grid[:, 1], c=mean, s=ms)
plt.colorbar()
plt.subplot(144)
plt.scatter(grid[:, 0], grid[:, 1], c=CRPS_new, s=ms)
plt.colorbar()
plt.show()

