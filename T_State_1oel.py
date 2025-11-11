import prody
import numpy as np

t_protein = prody.fetchPDB('1oel')
print(t_protein)

# Parse the PDB structure
atoms = prody.parsePDB('1oel')
print(f"Total atoms: {atoms.numAtoms()}")
print(f"Number of residues: {atoms.numResidues()}")
print(f"Number of chains: {atoms.numChains()}")

# Display basic information about the structure
print(f"\nStructure title: {atoms.getTitle()}")

# Get C-alpha atoms for normal mode analysis
calphas = atoms.select('calpha')
print(f"\nC-alpha atoms: {calphas.numAtoms()}")

# Display some structural details
print(f"\nFirst 10 residues:")
for i in range(min(10, calphas.numAtoms())):
    resname = calphas.getResnames()[i]
    resnum = calphas.getResnums()[i]
    chain = calphas.getChids()[i]
    print(f"  {chain}:{resname}{resnum}")