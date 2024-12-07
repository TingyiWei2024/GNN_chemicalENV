import pandas as pd

# Provide the correct path to your Excel file
file_path = "/workspaces/GNN_chemicalENV/DATA input/CCCC.xlsx"

# Load the Excel file
data = pd.read_excel(file_path)

# Display the first few rows of the data
print(data.head())
print(data.columns)

# get nodes/edges input
from rdkit import Chem

# from dataset
for idx, row in data.iterrows(): #.apply? 大包？
    molecule_smiles = row ['fragment1']
    bde_value = row['bde']
    bdfepred_value = row['bdfe_pred']

print(data.shape)  #check

    

