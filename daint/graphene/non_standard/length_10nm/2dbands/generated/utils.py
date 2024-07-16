import numpy as np


def handle_PBC(
    ng: int,
    g: int,
    n: int,
    R: int,
    coords_j: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Adjust atom coordinates w.r.t the periodic boundary
    conditions in the supplied direction.

    Parameters
    ----------
    `ng` : `int`
        number of grid cells in a given direction
    `g` : `int`
        index of grid cell in a given direction
    `n` : `int`
        index of neighboring grid cell in a given direction
    `R` : `int`
        R vector component in a given direction
    `coords_j` : `np.ndarray`
        coordinates of atom j
    `v` : `np.ndarray`
        lattice vector in a given direction

    Returns
    -------
    `tuple[np.ndarray, int]`
        adjusted coordinates of atom j and R vector component
    """
    if g == 0 and g + n < g and (g + n) % ng == ng - 1:
        R -= 1
        coords_j = coords_j - v
    elif g == ng - 1 and g + n > g and (g + n) % ng == 0:
        R += 1
        coords_j = coords_j + v
    return coords_j, R


def get_unit_cell_repetitions(
    length_x: float,
    length_y: float,
    a_unit: float,
    b_unit: float,
) -> tuple:
    """
    Calculate the number of unit cell repetitions in the x and y directions based on given unit cell dimensions.

    Parameters
    ----------
    length_x : float
        Total length in the x-direction in Angstrom.
    length_y : float
        Total length in the y-direction in Angstrom. This value will be used to calculate the number of unit cell repetitions
        in the y-direction based on the given `b_unit`.
    a_unit : float
        Length of one side of the unit cell in the x-direction in Angstrom. Used to calculate the number of unit cell repetitions
        in the x-direction based on the given `length_x`.
    b_unit : float
        Length of the unit cell in the y-direction in Angstrom. Used to calculate the number of unit cell repetitions
        in the y-direction based on the given `length_y`.

    Returns
    -------
    tuple[int, int]
        Number of unit cell repetitions in the x and y directions based on the given lengths and unit sizes.
    """

    repetitions_x = int(length_x // a_unit)
    repetitions_y = int(length_y // b_unit)
    return repetitions_x, repetitions_y

def split_data_by_bands(filename):
    bands = []
    current_band_data = []

    with open(filename, 'r') as file:
        next(file)

        for line in file:
            if line.strip():  # Check if the line is not empty
                data = line.split()
                current_band_data.append([float(val) for val in data])
            else:
                bands.append(current_band_data)
                current_band_data = []

        if current_band_data:
            bands.append(current_band_data)

    return np.array(bands)
