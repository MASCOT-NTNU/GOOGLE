import numpy as np

path = "../MAFIA/HITLMP/models/"

THRESHOLD = 20
beta1, beta0 = np.load(path + "Google_coef.npy")
threshold = np.load(path + "threshold.npy")
print("threshold is loaded: ", threshold)
print("beta1, beta0 is loaded: ", beta1, beta0)


np.save(path + "threshold.npy", threshold)
np.save(path + "Google_coef.npy", np.array([beta1, beta0]))

beta1, beta0 = np.load(path + "Google_coef.npy")
threshold = np.load(path + "threshold.npy")

print("threshold is updated: ", threshold)
print("beta1, beta0 is updated: ", beta1, beta0)
