# Contains all the necessary code to prepare the molecule:
#   - molecule sanitization (check in "import_prepare_mol" to change advanced sanitiization settings")
#   - geometry optimization (if specified by "do_geom = True"), with the specified settings

from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from rdkit import Chem
from rdkit.Chem import Mol, Atom
import numpy as np
import matplotlib.patheffects as PathEffects
from typing import List, Tuple
from numpy.typing import NDArray
from collections import Counter
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections.abc import Sequence
import typing
import rdkit.Chem.Draw
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt
import matplotlib

PRECISION = np.float32

def prepare_mol_from_sdf(filename_in: str, do_geometry:bool=True, do_charge:bool=False, property_name:str='_GasteigerCharge', max_iter:int=1000,
                       mmffvariant:str='MMFF94', seed:int=26, max_attempts:int=100) -> "List[Mol | None]":

    vs_library: "Sequence[Mol | None]" = Chem.SDMolSupplier(filename_in)
    vs_library_prepared: "List[Mol | None]" = []

    cnt = 0
    nmol = len(vs_library)

    for mol in vs_library:
        cnt += 1
        if cnt % 50 == 0:
            print('Molecule: ' + str(cnt))

        mol = prepare_mol(mol, do_geometry, do_charge, property_name, max_iter, mmffvariant, seed, max_attempts)

        if not mol:
            print ('Molecule ' + str(cnt) + ' of ' + str(nmol) + ' not computed.')
        vs_library_prepared.append(mol)
    return vs_library_prepared

def prepare_mol(mol: "Mol | None", do_geometry:bool=True, do_charge:bool=True, property_name:str='_GasteigerCharge', max_iter:int=1000,
                       mmffvariant:str='MMFF94', seed:int=26, max_attempts:int=5) -> "Mol | None":

    # 'mmffVariant : “MMFF94” or “MMFF94s”'
    # seeded coordinate generation, if = -1, no random seed provided
    # removes starting coordinates to ensure reproducibility
    # max attempts, to increase if issues are encountered during optimization

    err = 0

    if do_charge:
        property_name = '_GasteigerCharge'

    # options for sanitization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE

    # sanitization
    if mol is None:
        return None
    else:
        # sanitize
        sanitize_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanitize_fail: # type: ignore[truthy-bool]
            raise ValueError(sanitize_fail)
            # err = 1

        if do_geometry:
            mol, err = opt_geometry(mol, max_iter, mmffvariant, seed, max_attempts)

        # calculates or assigns atom charges based on what annotated in do_charge
        mol = rdmolops.RemoveHs(mol)

        if do_charge:
            mol, name, err = get_charge(mol, property_name, do_charge)

    if err == 1:
        print('Error in molecule pre-treatment')
        return None
    
    return mol



def opt_geometry(mol: Mol, max_iter: int, mmffvariant: str, seed: int, max_attempts: int) -> "Tuple[Mol, int]":

    err = 0
    try:
        mol = rdmolops.AddHs(mol)
        a = AllChem.EmbedMolecule(mol, useRandomCoords=True, useBasicKnowledge=True, randomSeed=seed, clearConfs=True, maxAttempts=max_attempts)
        if a == -1:
            err = 0

        AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iter, mmffVariant=mmffvariant)
    except ValueError:
        err = 1
    except TypeError:
        err = 1

    return mol, err


def get_charge(mol:Mol, property_name: str, do_charge: bool) -> "Tuple[Mol, str, int]":

   
    err = 0

    # partial charges
    if not do_charge:
        err = check_mol(mol, property_name, do_charge)
        if err == 0:
            # prepares molecule
            mol = Chem.RemoveHs(mol)
            n_at: int = mol.GetNumAtoms() # type: ignore[call-arg]
            # takes properties
            list_prop = mol.GetPropsAsDict()
            string_values = list_prop[property_name]  # extracts the property according to the set name
            string_values = string_values.split("\n")
            w = np.asarray([float(i) for i in string_values])
        else:
            mol = Chem.AddHs(mol)
            n_at = mol.GetNumAtoms() # type: ignore[call-arg]
            w = np.ones((n_at, 1)) / n_at
            w = np.asarray([float(i) for i in w])  # same format as previous calculation
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
        err = check_mol(mol, property_name, do_charge)

    return mol, property_name, err


# ----------------------------------------------------------------------------------------------------------------------
def check_mol(mol: Mol, property_name: str, do_charge: bool) -> int:
    """
    checks if the property (as specified by "property_name") is annotated and gives err = 0 if it is
    """
    n_at: int = mol.GetNumAtoms() # type: ignore[call-arg]
    if do_charge is False:
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


# ----------------------------------------------------------------------------------------------------------------------
def do_map(mol: Mol, fig_name:"str | None"=None, lab_atom:bool=False, text:bool=False, MapMin:int=0, MapMax:int=1) -> None:
    

    # settings
    
    scale = -1  # size of dots
    coordscale = 1  # coordinate scaling
    colmap = 'bwr'

    mol, name, err = get_charge(mol, property_name='_GasteigerCharge', do_charge=True)
    if err == 1:
        print('Error in charge calculation')

    n_at: int = mol.GetNumAtoms () # type: ignore[call-arg]   # num atoms
    charge: NDArray[np.floating] = np.zeros((n_at, 1), dtype=PRECISION)  # init weights
    # coordinates and property
    for atom in range (n_at):
        atom_obj: Atom = mol.GetAtomWithIdx(atom) # type: ignore[arg-type,call-arg]
        charge_val: float = atom_obj.GetProp('_GasteigerCharge')  # type: ignore[arg-type,call-arg,assignment]
        charge[atom] = charge_val

    opts = rdkit.Chem.Draw.DrawingOptions()
    opts.clearBackground = True # type: ignore[attr-defined]
    opts.bgColor = (1, 1, 1) # type: ignore[assignment]

    fig = SimilarityMaps.GetSimilarityMapFromWeights( # type: ignore[no-untyped-call]
        mol, charge, coordScale=coordscale, colorMap=colmap,
                                                      colors='w', alpha=0, scale=scale)

    # SimilarityMaps.Draw.MolDrawOptions.clearBackground
    if lab_atom is False:
        for elem in fig.axes[0].get_children ():
            if isinstance(elem, matplotlib.text.Text):
                elem.set_visible (False)

    plt.axis("off")

    if text is True:
        
        for at in range(mol.GetNumAtoms()): # type: ignore[call-arg]
            x = mol._atomPs[at][0]
            y = mol._atomPs[at][1]
            plt.text(x, y, f"{charge[at]:.2f}",
                      path_effects=[PathEffects.withStroke (linewidth=1, foreground="blue")])

    if fig_name is not None:
        fig.savefig(fig_name, bbox_inches='tight')

    plt.show()


def frequent_scaffolds(suppl: Sequence[Mol], output_type:str='supplier') -> "List[Tuple[str, int]] | List[Mol]":
    """
     starting from a supplier file, the function computes the most frequently recurring scaffolds and returns them as a
     supplier file (if output_type='supplier') or as a counter file.
     """

    
    scaff_list: "List[str]" = []

    
    for mol in suppl:
        scaf_smi: str = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) # type: ignore[no-untyped-call]
        scaff_list.append(scaf_smi)

    freq_scaffolds_counter: typing.Counter[str] = Counter()
    for scaff in scaff_list:
        freq_scaffolds_counter[scaff] += 1

    freq_scaffolds = freq_scaffolds_counter.most_common()

    if output_type is 'supplier':
        # converts it back in a supplier file,
        suppl_new: "List[Mol]" = []
        for scaf_smi, count in freq_scaffolds:
            mol = Chem.MolFromSmiles(scaf_smi) # type: ignore[call-overload]
            name = str(round((count/len(suppl))*100,2))+'%'
            mol.SetProp("_Name", name) # assigns the molecule name as the percentage occurrence
            suppl_new.append(mol)

        return suppl_new


    return freq_scaffolds
