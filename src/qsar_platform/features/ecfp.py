from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_ecfp(smiles_list, radius=2, n_bits=2048):
    features = []
    valid_idx = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(n_bits))
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

        features.append(arr)
        valid_idx.append(i)

    return np.array(features), valid_idx
