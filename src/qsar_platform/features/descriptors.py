from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

DESCRIPTOR_NAMES = [desc[0] for desc in Descriptors.descList]
DESCRIPTOR_FUNCS = [desc[1] for desc in Descriptors.descList]

def smiles_to_descriptors(smiles_list):
    rows = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            rows.append([0.0] * len(DESCRIPTOR_FUNCS))
            continue

        values = []
        for func in DESCRIPTOR_FUNCS:
            try:
                values.append(func(mol))
            except Exception:
                values.append(0.0)

        rows.append(values)

    return pd.DataFrame(rows, columns=DESCRIPTOR_NAMES)
