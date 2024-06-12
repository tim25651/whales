"""
# =====================================================================================
# * Weighted Holistic Atom Localization and Entity Shape (WHALES) descriptors *
#   v. 1, May 2018
# -------------------------------------------------------------------------------------
# This file contains all the necessary functions
# to calculate WHALES descriptors for the
# molecules contained in an rdkit supplier.
#
# Francesca Grisoni, May 2018, ETH Zurich & University of Milano-Bicocca,
# francesca.grisoni@unimib.it
# please cite as:
#   Francesca Grisoni, Daniel Merk, Viviana Consonni,
#   Jan A. Hiss, Sara Giani Tagliabue, Roberto Todeschini & Gisbert Schneider
#   "Scaffold hopping from natural products to synthetic
#   mimetics by holistic molecular similarity",
#   Nature Communications Chemistry 1, 44, 2018.
# =====================================================================================
"""

# pylint: disable=consider-using-assignment-expr

from typing import List, Tuple

import lcm
import mol_properties
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Mol  # pylint: disable=unused-import

PRECISION = np.float32


# -----------------------------------------------------------------------------
def whales_from_mol(
    mol: "Mol | None",
    charge_threshold: int = 0,
    do_charge: bool = True,
    property_name: str = "",
) -> "Tuple[np.ndarray, list | None]":
    # check for correct molecule import, throw an error if import/sanitization fail
    mol = import_mol(mol)

    errors = 0

    lab: "List[str] | None" = None
    if not mol:
        x: NDArray[np.floating] = np.full((33,), -999.0, dtype=PRECISION)
        errors += 1
        print("Molecule not loaded.")
    else:
        # coordinates and partial charges (checks for computed charges)
        coords_w = mol_properties.get_coordinates_and_prop(
            mol, property_name, do_charge
        )
        if coords_w:  # no errors in charge
            coords, w = coords_w
            # does descriptors
            x, lab = do_lcd(coords, w, charge_threshold)
        else:
            x = np.full((33,), -999.0, dtype=PRECISION)
            errors += 1
            print("No computed charges.")

    return x, lab


def import_mol(mol: "Mol | None") -> "Mol | None":
    # options for sanitization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE

    # initialization

    if mol is None:
        return None

    # sanitize
    sanit_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
    if sanit_fail:  # type: ignore[truthy-bool]
        raise ValueError(sanit_fail)
        # err = 1

    return mol


# -----------------------------------------------------------------------------
def do_lcd(
    coords: NDArray[np.floating], w: NDArray[np.floating], thr: float
) -> "Tuple[np.ndarray, list]":
    """
    Core function for computing 3D LCD descriptors,
    starting from the coordinates and the partial charges.

    Args:
        coords: molecular 3D coordinate matrix (n_at x 3)
        w: molecular property to consider (n_at x 1)
        thr: threshold to be used to retain atoms (e.g., 0.001)

    Returns:
        A tuple containing:
            x_all: descriptors  for the molecules (1 x p)
            lab_all: descriptors labels (1 x p)
    """

    # calculates lcm with weight scheme 1 (all charges)
    res = lcm.lmahal(coords, w)

    # applies sign
    res = apply_sign(w, res, thr)

    x_all, lab_all = extract_lcm(res)  # MDs and labels

    return x_all, lab_all


# -----------------------------------------------------------------------------
def apply_sign(
    w: NDArray[np.floating], res: NDArray[np.floating], thr: float = 0.0
) -> NDArray[np.floating]:
    """
    applies the sign to negatively charged atoms.

    Args:
        w: partial charge
        res: computed atomic descriptors
        thr: threshold to consider atoms as negatively charged
            (default is 0); other atoms are removed

    Returns:
        computed atomic descriptors with adjusted sign
    """

    # find negative weights and assigns a "-"
    a, _ = np.where(w < 0)
    res[a, :] *= -1

    # removes atoms with abs(w) smaller than the thr
    a, _ = np.where(abs(w) < thr)
    res = np.delete(res, a, 0)

    return res


# -----------------------------------------------------------------------------
def extract_lcm(
    data: NDArray[np.floating],
    start: int = 0,
    end: int = 100,
    step: int = 10,
    lab_string: str = "",
) -> "Tuple[NDArray[np.floating], List[str]]":
    """
    extracts descriptors referred to the whole molecule from numbers referred to atoms,
    e.g., R and I.

    Args:
        data: atomic description (n_atom x p)
        start: minimum percentile (default = minimum value)
        end: maximum percentile (default = maximum value)
        step: step for percentiles generation (default, 10 corresponds to deciles)
        lab_string: additional string to be added to differentiate weighting schemes

    Returns:
        A tuple containing:
            x: molecular description based on percentiles (1 x p1)
            labels: descriptor labels (1 x p1)
    """

    # Calculates percentiles according to the specified settings
    perc = range(start, end + 1, step)

    x = np.percentile(data, list(perc), axis=0)

    # Flattens preserving the ordering
    x = np.concatenate((x[:, 0], x[:, 1], x[:, 2]), axis=0)

    # rounds the descriptors to the third decimal place
    x = np.round(x, 3)

    # produces labels strings
    strings = ["R_", "I_", "IR_"]
    labels: list[str] = []
    for j in strings:
        for i in perc:
            labels.append(j + lab_string + str(int(i / 10)))

    return x, labels
