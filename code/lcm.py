"""
# =====================================================================================
# * Weighted Holistic Atom Localization and Entity Shape (WHALES) descriptors *
#   v. 1, May 2018
# -------------------------------------------------------------------------------------
# This file contains all the necessary files to calculate
# atom centred mahalanobis descriptors.
# Starting from the 3D coordinates and the partial
# charges of the molecules, the isolation degree, remoteness
# and their ratio are computed.
# The covariance is centered on each atom and weighted according
# to the partial charges of the selected surrounding
# atoms.
#
# Francesca Grisoni, May 2018, ETH Zurich & University of Milano-Bicocca,
# francesca.grisoni@unimib.it
# please cite as:
#   Francesca Grisoni, Daniel Merk, Viviana Consonni, Jan A. Hiss,
#   Sara Giani Tagliabue, Roberto Todeschini & Gisbert Schneider
#   "Scaffold hopping from natural products to synthetic mimetics
#   by holistic molecular similarity",
#   Nature Communications Chemistry 1, 44, 2018.
# =====================================================================================
"""

# pylint: disable=consider-using-assignment-expr

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

PRECISION = np.float32


def lmahal(
    x: "NDArray[np.floating]", w: "NDArray[np.floating]"
) -> "NDArray[np.floating]":
    """
    main function for calculating the atom-centred mahalanobis distance (ACM),
    used to compute remoteness and isolation degree.

    Args:
        x: molecular 3D coordinate matrix (n_at x 3)
        w: molecular property to consider (n_at x 1)

    Returns:
        res(n_at x 3): atomic descriptors;
        col 0 = Remoteness, col 1 = Isolation degree, col 2 = Isol/Remoteness

    REF: Todeschini, et al. "Locally centred Mahalanobis distance:
    a new distance measure with salient features towards
    outlier detection." Analytica chimica acta 787 (2013): 1-9.

    Francesca Grisoni, 12/2016, v. alpha
    ETH Zurich
    """

    # preliminary
    n, p = x.shape  # matrix dimensions

    if len(w) > 0:  # checks whether at least one atom was included
        dist: "NDArray[np.floating]" = np.zeros(
            (n, n), dtype=PRECISION
        )  # pre allocation (LCM)

        # do covariance centred on each sample
        cov = docov(x, w)

        # calculate distance
        for i in range(n):
            for j in range(n):
                d = domahal(i, j, x, cov)
                dist[i, j] = d / p

        # isolation and remoteness parameters from D
        isol, rem, ir_ratio = is_rem(
            dist, n
        )  # calculates atomic parameters from the distance
        res: NDArray[np.floating] = np.concatenate(
            (rem, isol, ir_ratio), axis=1
        )  # results concatenation
    else:
        res = np.full((1, 3), -999.0, dtype=PRECISION)  # sets missing values

    return res


# -----------------------------------------------------------------------------
def docov(
    x: "NDArray[np.floating]", w: "NDArray[np.floating]"
) -> "Dict[Tuple[int,int], NDArray[np.floating]]":
    """
    Calculates the weighted covariance matrix centered on each atom.
    The original centred covariance (Todeschini et al. 2013)
    is weighted according to the atomic partial charges (normalized absolute values).

    Args:
        x: molecular 3D coordinate matrix (n_at x 3)
        w: molecular property to consider (n_at x 1)

    Returns:
        cov(n_at x n_at): weighted atomic centred covariance

    Francesca Grisoni, 12/2016, v. alpha
    ETH Zurich
    """

    n, p = x.shape  # dimensions
    cov: "Dict[Tuple[int,int], NDArray[np.floating]]" = {}  # pre allocation
    samp_v: "NDArray[np.floating]" = np.zeros((p, p), dtype=PRECISION)  # init

    type_w = 1  # if 1, it normalizes according to the total sum of weights

    # normalizes partial charges
    if type_w == 2:
        den = n - 1
    else:
        den = sum(abs(w))
        if den == 0:
            den = n - 1

    w_abs = abs(w) / den

    for i in range(n):
        for j in range(p):
            for k in range(p):
                cvhere: float = 0.0
                for s in range(n):
                    cvhere += w_abs[s] * (x[s, j] - x[i, j]) * (x[s, k] - x[i, k])
                samp_v[j, k] = cvhere
        cov[i, 1] = samp_v
        samp_v = np.zeros((p, p), dtype=PRECISION)  # re-init

    return cov


# -----------------------------------------------------------------------------


def domahal(
    i: int,
    j: int,
    x: NDArray[np.floating],
    cov: "Dict[Tuple[int,int], NDArray[np.floating]]",
) -> NDArray[np.floating]:
    """
    Calculates the atom centred Mahalanobis distance between
    two atoms i and j when the covariance is centered in j.

    Args:
        i: atoms whose distance has to be computed ...
        j: ... (when the covariance is centred in j)
        x: molecular 3D coordinate matrix (n_at x 3)
        cov: centred covariance

    Returns:
        d: distance between i and j (centered in j)

    Francesca Grisoni, 12/2016, v. alpha
    ETH Zurich
    """

    sv = np.linalg.pinv(cov[j, 1])  # pseudo inverse of covariance
    res = x[i, :] - x[j, :]
    d1 = np.dot(res, sv)  # first part of the matrix product
    d = np.dot(d1, res[np.newaxis, :].T)  # transpose and product # TODO write it better

    return d  # type: ignore[no-any-return]


# -----------------------------------------------------------------------------


def is_rem(
    dist: NDArray[np.floating], n: int
) -> (
    "Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]"
):  # TODO remove n and calculate it here
    """
    Calculates isolation degree and remoteness from a distance matrix and their ratio.

    Args:
        dist: atom-centred Mahalanobis
        n: number of compounds

    Returns:
        A tuple containing:
            isol: isolation degree (column minimum)
            rem: remoteness (row average)
            ir_ratio: ratio between isolation degree and remoteness

    Francesca Grisoni, 12/2016, v. alpha
    ETH Zurich
    """

    for i in range(n):
        dist[i, i] = np.nan

    dist_matrix = np.matrix(dist, dtype=PRECISION)
    isol = np.transpose(
        np.nanmin(dist_matrix, axis=0)
    )  # col minimum (transposed for dimensions)
    rem = np.nanmean(dist_matrix, axis=1, dtype=PRECISION)  # row average
    ir_ratio = isol / rem  # ratio between isol and rem (transpose for dimensions)

    return isol, rem, ir_ratio
