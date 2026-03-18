from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

DESCRIPTOR_FUNCS = [desc[1] for desc in Descriptors.descList]

def smiles_to_descriptors(smiles_list):
    features = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            features.append([0.0] * len(DESCRIPTOR_FUNCS))
            continue

        values = []
        for func in DESCRIPTOR_FUNCS:
            try:
                values.append(func(mol))
            except:
                values.append(0.0)

        features.append(values)

    return np.array(features)
