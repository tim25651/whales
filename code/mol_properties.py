# ======================================================================================================================
# * Weighted Holistic Atom Localization and Entity Shape (WHALES) descriptors *
#   v. 1, May 2018
# ----------------------------------------------------------------------------------------------------------------------
# This file contains all the necessary files to handle molecular properties and coordinates.
#
# Francesca Grisoni, May 2018, ETH Zurich & University of Milano-Bicocca, francesca.grisoni@unimib.it
# please cite as xxxx
# ======================================================================================================================

import numpy as np
from rdkit.Chem import Mol, Atom
from typing import Tuple
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Geometry import Point3D
from numpy.typing import NDArray
PRECISION = np.float32

def get_coordinates_and_prop(mol: Mol, property_name:str='partial_charges', do_charge:bool=True) -> "Tuple[NDArray[np.floating], NDArray[np.floating]] | None":
    """
    Extracts all of the useful chemical information, i.e., the partial charge and the coordinates and formats it
    for atomic centred covariance matrix calculation.
    ====================================================================================================================
    :param
    mol: rdkit molecule
    do_charge: if True, the charges are computed
    do_geom: if True, it calculates MMF 3D coordinates
    :returns
    coords (n_atoms x 3): geometrical matrix (x-y-z coords)
    w (n_atoms x 1): partial charge array
    ====================================================================================================================
    Francesca Grisoni, 05/2018, v. beta
    ETH Zurich
    """

    # molecule preparation
    mol, property_name, err = prepare_mol(mol, property_name, do_charge)

    if err == 0:
        # pre-allocation
        n_at: int = mol.GetNumAtoms()   # type: ignore[call-arg] # num atoms
        coords: NDArray[np.floating] = np.zeros((n_at, 3), dtype=PRECISION)  # init coords
        w: NDArray[np.floating] = np.zeros((n_at, 1), dtype=PRECISION)  # init weights

        # coordinates and property
        for atom in range(n_at):  # loops over atoms, gets 3D coordinate matrix

            # gets atomic positions
            conf = mol.GetConformer()
            pos: Point3D = conf.GetAtomPosition(atom) # type: ignore[arg-type,call-arg]
            coords[atom, ] = [pos.x, pos.y, pos.z]

            # gets atomic properties
            atom_obj: Atom = mol.GetAtomWithIdx(atom) # type: ignore[arg-type,call-arg]
            w[atom] = atom_obj.GetProp(property_name) # type: ignore[arg-type,call-arg]
            
        # checks the weight values computed and throws and error if they are all 0
        if all(v == 0 for v in w):
            return None
        
        return coords, w
    else:
        return None

# ----------------------------------------------------------------------------------------------------------------------


def prepare_mol(mol: Mol, property_name: str, do_charge: bool) -> "Tuple[Mol, str, int]":
    """
    Sets atomic properties if they are specified in the sdf, otherwise computes them. If specified, computes 3D coordinates
    using MMF.  The default number of iterations is 200, but it is progressively increased to 5000 (with a step of 500)
    in case convergence is not reached.
    ====================================================================================================================
    :param
    mol: molecule to be analyzed (from rdkit supplier)
    property_name: name of the property to be used
    do_charge: if True, partial charge is computed
    do_geom: if True, molecular geometry is optimized
    :return:
    mol: molecule with property and 3D coordinates (H depleted)
    property_name: updated on the basis of the settings
    ====================================================================================================================
    Francesca Grisoni, 12/2016, v. alpha
    ETH Zurich
    """
    err = 0

    # partial charges
    if not do_charge:
        if property_name:
            err = check_mol(mol, property_name, do_charge)
            if err == 0:
                # prepares molecule
                # mol = Chem.AddHs(mol)
                mol = Chem.RemoveHs(mol)
                n_at = mol.GetNumAtoms() # type: ignore[call-arg]
                # takes properties
                list_prop = mol.GetPropsAsDict()
                string_values = list_prop[property_name]  # extracts the property according to the set name
                string_values = string_values.split("\n")
                w = np.asarray([float(i) for i in string_values])
            return mol, property_name, err
        else:
            mol = Chem.AddHs(mol)
            n_at = mol.GetNumAtoms() # type: ignore[call-arg]
            w = np.ones((n_at, 1), dtype=PRECISION)/n_at
            w = np.asarray([float(i) for i in w])
            property_name = 'equal_w'
            err = 0
        # extract properties
        for atom in range(n_at):
            atom_obj: Atom = mol.GetAtomWithIdx(atom) # type: ignore[call-arg,arg-type]
            atom_obj.SetDoubleProp(property_name, w[atom])

        mol = Chem.RemoveHs(mol)

    # Gasteiger-Marsili Charges
    elif err == 0:
        AllChem.ComputeGasteigerCharges(mol)
        property_name = '_GasteigerCharge'
        err = check_mol(mol, property_name, do_charge)

    return mol, property_name, err


# ----------------------------------------------------------------------------------------------------------------------
def check_mol(mol: "Mol | None", property_name: str, do_charge: bool) -> int:
    """
    checks if the property is annotated and gives 0 if it is
    """
    if mol is None:
        return 1
    
    n_at: int = mol.GetNumAtoms() # type: ignore[call-arg]
    if not do_charge:
        list_prop = mol.GetPropsAsDict()
        string_values = list_prop[property_name]  # extracts the property according to the set name
        if string_values == '' or string_values == ['']:
            err = 1
        else:
            err = 0
    else:
        err = 0
        atom = 0
        while atom < n_at:
            atom_obj: Atom = mol.GetAtomWithIdx(atom) # type: ignore[call-arg,arg-type]
            value = atom_obj.GetProp(property_name) # type: ignore[call-arg,arg-type]
            # checks for error (-nan, inf, nan)
            if value == '-nan' or value == 'nan' or value == 'inf':
                err = 1
                break

            atom += 1

    # checks for the number of atoms
    if n_at < 4:
        err = 1

    return err

