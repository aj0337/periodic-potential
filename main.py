import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pandas as pd
from io import TextIOWrapper

filename = 'graphene_test/TG_hr_only_t1.dat'
num_atom, np_data, pd_data = read_data(filename, sparse=True)
phipath, _ = build_phipath_GammaKMGamma(30)
npts = len(phipath)
R_cell = pd_data[[0, 1, 2]].drop_duplicates().reset_index(drop=True)
n_r = len(R_cell)
dim = (n_r, num_atom, num_atom)
HR = build_realspace_H(dim, pd_data, R_cell.to_numpy())
Hk = build_phispace_H(np_data, num_atom, HR, R_cell, phipath)

bands = []
for npt in range(npts):
    eigenergies, _ = eigh(Hk[npt])
    bands.append(eigenergies)
bands = (np.array(bands))
x = np.arange(0, npts, 1)

# xlabel = path
# xtics = [0,199,249,349]
plt.plot(x, bands[:, 0], c='r')
plt.plot(x, bands[:, 1], c='black')
# plt.xticks(xtics,labels=xlabel)
plt.show()
