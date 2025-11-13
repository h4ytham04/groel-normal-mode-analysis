from prody import *
import prody
import numpy as np

atoms = prody.parsePDB('1oel')
print(f"Total atoms: {atoms.numAtoms()}")
print(f"Number of residues: {atoms.numResidues()}")

#for nma using only c-alphas
calphas = atoms.select('calpha')

gnm = GNM('Chaperonin')
gnm.buildKirchhoff(calphas)
print(gnm.getKirchhoff())