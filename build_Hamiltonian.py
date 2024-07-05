import numpy as np
from .read_TG_hr_from_WT import pick_cell_R


def is_hermitian(M: np.ndarray):
    return np.isclose(M, M.transpose().conjugate()).all()


def build_realspace_H(dim, data, R_cell):
    H_R = np.zeros(dim)
    for i, r in enumerate(R_cell):
        data_r = pick_cell_R(data, r)
        for _, (a1, a2, c) in data_r[[3, 4, 11]].iterrows():
            i1, i2 = int(a1) - 1, int(a2) - 1
            H_R[i, i1, i2] = c
    return H_R


# FIXME Due to conversion units and factors like 2pi, kdotr is currently wrong and build_kspace_H returns the wrong bandstructure
def build_kspace_H(np_data, num_atom, HR, R_cell, kpath):
    r1 = np_data[:, [5, 6, 7]]
    r2 = np_data[:, [8, 9, 10]]
    Rc = np_data[:, [0, 1, 2]]
    R2 = np.add(Rc, r2)
    R = np.subtract(R2, r1)
    ndegen = 1  #Hardcoded at the moment but it is the value that is printed after number of R points in TG_hr.dat
    nkpts = len(kpath)
    Hk = np.zeros((nkpts, num_atom, num_atom), dtype=complex)
    # normalization = 1/(2*np.pi)
    normalization = 1
    # kpath = k_direct_to_cart(kpath,1.42)
    for nk in range(nkpts):
        k = kpath[nk]
        for iter, (n1, n2, n3, a1,
                   a2) in enumerate(np_data[:, [0, 1, 2, 3, 4]].astype(int)):
            i = R_cell.index[(R_cell[0] == n1) & (R_cell[1] == n2) &
                             (R_cell[2] == n3)].to_list()[0]
            kdotr = k[0] * R[iter, 0] + k[1] * R[iter, 1] + k[2] * R[iter, 2]
            phase = np.exp(2 * np.pi * 1j * kdotr) / ndegen
            Hk[nk, a1 - 1,
               a2 - 1] += normalization * phase * HR[i, a1 - 1, a2 - 1]
        if not is_hermitian(Hk[nk]):
            print(f'HK is not Hermitian at the kpoint {k} with index {nk}')
            continue
        Hk[nk] = 0.5 * (Hk[nk] + np.conj(Hk[nk]).T)  # enforces hermiticity
    return Hk


def build_phispace_H(np_data, num_atom, HR, R_cell, phipath):
    npts = len(phipath)
    Hk = np.zeros((npts, num_atom, num_atom), dtype=complex)
    for npt in range(npts):
        phi1, phi2 = phipath[npt]
        phi3 = 0.0
        for n1, n2, n3, a1, a2 in np_data[:, [0, 1, 2, 3, 4]].astype(int):
            # FIXME find a cleaner way to do this
            i = R_cell.index[(R_cell[0] == n1) & (R_cell[1] == n2) &
                             (R_cell[2] == n3)].to_list()[0]
            phase = np.exp(1j * n1 * phi1) * np.exp(1j * n2 * phi2) * np.exp(
                1j * n3 * phi3)
            Hk[npt, a1 - 1, a2 - 1] += phase * HR[i, a1 - 1, a2 - 1]
        if not is_hermitian(Hk[npt]):
            print(
                f'HK is not Hermitian at the kpoint {phipath[npt]} with index {npt}'
            )
            continue
        Hk[npt] = 0.5 * (Hk[npt] + np.conj(Hk[npt]).T)  # enforces hermiticity
    return Hk
