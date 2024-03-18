import numpy as np


def build_path(p1, p2, npts):
    delta = (p2 - p1) / npts
    points = [p1 + i * delta for i in range(npts - 1)]
    points.append(p2)
    return points


def build_kpath(segment_points, npts_per_segment):
    kpath = []
    for p in range(len(segment_points) - 1):
        p1 = segment_points[p]
        p2 = segment_points[p + 1]
        npts = npts_per_segment[p]
        kpath.extend(build_path(np.array(p1), np.array(p2), npts))
    return np.array(kpath)


def build_grid(xmin, xmax, ymin, ymax, stepsize):
    x_range = np.arange(xmin, xmax, stepsize)
    y_range = np.arange(ymin, ymax, stepsize)
    z = 0
    grid = []
    for x in x_range:
        grid.extend((x, y, z) for y in y_range)
    return np.array(grid)


def get_dist_between(p1, p2):
    return np.linalg.norm(p2 - p1)


def k_direct_to_cart(kpath, dCC):
    # FIXME Ideally, read automatically from POSCAR. Manual entry for now
    lattice_vec = np.array([
        [2.46, 0.00, 0.00],
        [-1.23, 2.13, 0.00],
        [0.00, 0.00, 10.00],
    ])
    A = lattice_vec.T
    B = np.linalg.inv(A)
    # return cartesian coordinates stored row-wise
    return np.matmul(B, kpath.T).T


def build_phipath_GammaKMGamma(Npiover3=10):

    dphi = (np.pi / 3) / Npiover3
    phi1path = []
    phi2path = []
    Gamma = [0.0, 0.0]
    K = [2 * np.pi / 3, -2 * np.pi / 3]
    M = [np.pi, -np.pi]
    M2 = [-np.pi, 0.0]
    GammaK_phi1 = np.arange(Gamma[0], K[0] - 1e-8, dphi)
    GammaK_phi2 = np.arange(Gamma[1], K[1] + 1e-8, -dphi)
    phi1path.extend(GammaK_phi1)
    phi2path.extend(GammaK_phi2)
    KM_phi1 = np.arange(K[0], M[0] - 1e-8, dphi)
    KM_phi2 = np.arange(K[1], M[1] + 1e-8, -dphi)
    phi1path.extend(KM_phi1)
    phi2path.extend(KM_phi2)
    npts = round(np.sqrt(3) / 2 * len(GammaK_phi1))
    dphiaux = (Gamma[0] - M[1]) / npts
    M2Gamma_phi1 = np.arange(M2[0], Gamma[0] - np.sign(dphiaux) * 1e-8,
                             dphiaux)
    M2Gamma_phi2 = [M2[1]] * len(M2Gamma_phi1)
    phi1path.extend(M2Gamma_phi1)
    phi2path.extend(M2Gamma_phi2)
    phi1path.append(Gamma[0])
    phi2path.append(Gamma[1])
    phipath = np.column_stack((phi1path, phi2path))
    pathtickpos = [
        1,
        len(GammaK_phi1) + 1,
        len(GammaK_phi1) + len(KM_phi1) + 1,
        len(phi1path)
    ]
    pathticklabels = ["Γ", "K", "M", "Γ"]
    pathticks = (pathtickpos, pathticklabels)
    return phipath, pathticks


def build_phipath_2D_MKGammaKpM(Npiover3=10):

    dphi = (np.pi / 3) / Npiover3
    phi1path = []
    phi2path = []
    M = [np.pi, -np.pi]
    K = [2 * np.pi / 3, -2 * np.pi / 3]
    Gamma = [0.0, 0.0]
    Kp = [-2 * np.pi / 3, 2 * np.pi / 3]
    M2 = [-np.pi, np.pi]  # equivalent to M
    MK_phi1 = np.arange(M[0], K[0] + dphi, -dphi)
    MK_phi2 = np.arange(M[1], K[1] - dphi, dphi)
    phi1path.extend(MK_phi1)
    phi2path.extend(MK_phi2)
    KGamma_phi1 = np.arange(K[0], Gamma[0] + dphi, -dphi)
    KGamma_phi2 = np.arange(K[1], Gamma[1] - dphi, dphi)
    phi1path.extend(KGamma_phi1)
    phi2path.extend(KGamma_phi2)
    GammaKp_phi1 = np.arange(Gamma[0], Kp[0] + dphi, -dphi)
    GammaKp_phi2 = np.arange(Gamma[1], Kp[1] - dphi, dphi)
    phi1path.extend(GammaKp_phi1)
    phi2path.extend(GammaKp_phi2)
    KpM2_phi1 = np.arange(Kp[0], M2[0] + dphi, -dphi)
    KpM2_phi2 = np.arange(Kp[1], M2[1] - dphi, dphi)
    phi1path.extend(KpM2_phi1)
    phi2path.extend(KpM2_phi2)
    phi1path.append(M2[0])
    phi2path.append(M2[1])
    phipath = np.column_stack((phi1path, phi2path))
    pathtickpos = [
        1,
        len(MK_phi1) + 1,
        len(MK_phi1) + len(KGamma_phi1) + 1,
        len(MK_phi1) + len(KGamma_phi1) + len(GammaKp_phi1) + 1,
        len(phi1path)
    ]
    pathticklabels = ["M", "K", "Γ", "K'", "M"]
    pathticks = (pathtickpos, pathticklabels)
    return phipath, pathticks
