from prody import *
import prody
import numpy as np
from contact_map import ContactFrequency
import matplotlib.pyplot as plt

r_state = prody.parsePDB('4aas')
rprime_state = prody.parsePDB('1svt')

chains_to_analyze = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

results = []
optimal_chains = [] 

for chain_id in chains_to_analyze:
    
    print()
    print()
    print(f'chain {chain_id}')
    print()
    print()
    
    #select specific chain
    r_chain = r_state.select(f'calpha and chain {chain_id}')
    rprime_chain = rprime_state.select(f'calpha and chain {chain_id}')
    
    #match chains to handle missing residues
    matches = prody.matchChains(r_chain, rprime_chain)
    
    
    r_matched = matches[0][0]
    rprime_matched = matches[0][1]
    
    print(f'matched atoms: {r_matched.numAtoms()}')
    
    #anm on r state
    anm_r = ANM(f'R State Chain {chain_id} ANM analysis')
    anm_r.buildHessian(r_matched, cutoff=7, gamma=1)
    anm_r.calcModes()
    
    print(f'hessian shape: {anm_r.getHessian().shape}')
    
    anm_r_eigvecs = anm_r.getEigvecs()
    anm_r_eigvals = anm_r.getEigvals()
    
    #align structures
    rprime_aligned = rprime_matched.copy()
    calcTransformation(rprime_aligned, r_matched).apply(rprime_aligned)
    
    #r to rprime displacement
    delta = rprime_aligned.getCoords() - r_matched.getCoords()
    delta_vector = delta.flatten()


    prodyrmsd = calcRMSD(r_matched, rprime_aligned)
    print(f'prody rmsd: {prodyrmsd:.3f} angstroms')
    
    #mode overlap
    print()
    print('mode overlap')
    
    overlaps = []
    for i in range(6, 16): #first 6 are zero
        mode = anm_r_eigvecs[:, i]
        
        mode_norm = mode / np.linalg.norm(mode) #.norm does distance from origin
        delta_norm = delta_vector / np.linalg.norm(delta_vector)
        
        overlap = abs(np.dot(mode_norm, delta_norm))
        overlaps.append(overlap)
        
        print(f'non-trivial mode {i-5} (actual mode {i}): {overlap:.4f}')
    
    best_mode = np.argmax(overlaps) + 1 #+1 to convert from index to mode number (starting at 1)
    best_overlap = max(overlaps)
    
    print()
    print(f'mode {best_mode} best predicts r to rprime transition')
    print(f'overlap: {best_overlap:.4f}')
    
    #store data for pseudoprotein
    best_mode_index = best_mode + 5  #convert back to actual array index mode 1 is index 6
    optimal_chains.append({
        'chain_id': chain_id,
        'r_matched': r_matched,
        'best_mode_index': best_mode_index,
        'best_mode_vector': anm_r_eigvecs[:, best_mode_index],
        'rmsd': prodyrmsd
    })
    
    results.append({
        'chain': chain_id,
        'atoms': r_matched.numAtoms(),
        'rmsd': prodyrmsd,
        'best_mode': best_mode,
        'best_overlap': best_overlap
    })

print()
print()
for r in results:
    print(f"results for chain {r['chain']}: atoms={r['atoms']}, rmsd={r['rmsd']:.3f} Angstroms, best mode={r['best_mode']}, best overlap={r['best_overlap']:.4f}")

print()
print("create pseudo-protein from best modes")
# Create pseudo-protein by applying best mode displacements
all_pseudo_coords = []
all_chain_data = []

for chain_data in optimal_chains:
    chain_id = chain_data['chain_id']
    r_matched = chain_data['r_matched']
    best_mode_vector = chain_data['best_mode_vector']
    rmsd = chain_data['rmsd']
    
    # Get original coordinates
    original_coords = r_matched.getCoords()
    
    # Reshape mode vector to Nx3 (x, y, z displacements per atom)
    mode_coords = best_mode_vector.reshape((-1, 3))
    
    # Scale mode to match the actual RMSD of the transition
    scaling_factor = rmsd / np.sqrt(np.mean(np.sum(mode_coords**2, axis=1)))
    scaled_displacement = mode_coords * scaling_factor
    
    # Apply displacement to create pseudo-structure coordinates
    pseudo_coords = original_coords + scaled_displacement
    
    all_pseudo_coords.append(pseudo_coords)
    all_chain_data.append({
        'coords': pseudo_coords,
        'chain_id': chain_id,
        'atom_map': r_matched
    })
    
    print(f"Chain {chain_id}: Applied mode displacement (scaled by {scaling_factor:.2f})")

# Build pseudo-protein structure
combined_pseudo_coords = np.vstack(all_pseudo_coords)

# Use the first chain as template and build from there
first_chain = all_chain_data[0]['atom_map']
n_atoms = first_chain.numAtoms()

# Create lists for all atom properties
all_names = []
all_resnums = []
all_resnames = []
all_chains = []
all_elements = []

for chain_data in all_chain_data:
    atom_map = chain_data['atom_map']
    all_names.extend(atom_map.getNames())
    all_resnums.extend(atom_map.getResnums())
    all_resnames.extend(atom_map.getResnames())
    all_chains.extend([chain_data['chain_id']] * atom_map.numAtoms())
    all_elements.extend(atom_map.getElements())

# Create new AtomGroup
pseudo_protein = prody.AtomGroup('Pseudo-protein from best modes')
pseudo_protein.setCoords(combined_pseudo_coords)
pseudo_protein.setNames(all_names)
pseudo_protein.setResnums(all_resnums)
pseudo_protein.setResnames(all_resnames)
pseudo_protein.setChids(all_chains)
pseudo_protein.setElements(all_elements)

# Save the pseudo-protein
prody.writePDB('pseudo_protein_best_modes_r_to_rprime.pdb', pseudo_protein)
print(f"Pseudo-protein saved to: pseudo_protein_best_modes_r_to_rprime.pdb")
print(f"Total atoms in pseudo-protein: {pseudo_protein.numAtoms()}")

# Compare pseudo-protein to actual Rprime state
print()
print("COMPARING PSEUDO-PROTEIN TO ACTUAL RPRIME STATE")

# Load full Rprime state and match it to pseudo-protein structure
rprime_state_full = prody.parsePDB('1svt')
rprime_calphas = rprime_state_full.select('calpha')

# Match to ensure same residues
matches_pseudo = prody.matchChains(pseudo_protein, rprime_calphas)
if matches_pseudo:
    pseudo_matched = matches_pseudo[0][0]
    rprime_matched_full = matches_pseudo[0][1]
    
    # Align for comparison
    rprime_aligned_full = rprime_matched_full.copy()
    calcTransformation(rprime_aligned_full, pseudo_matched).apply(rprime_aligned_full)
    
    # Calculate RMSD between pseudo-protein and actual Rprime state
    pseudo_to_rprime_rmsd = calcRMSD(pseudo_matched, rprime_aligned_full)
    
    print(f"RMSD between pseudo-protein and actual Rprime state: {pseudo_to_rprime_rmsd:.3f} Angstroms")
    print(f"This shows how well the best modes predict the conformational change!")
else:
    print("Could not match pseudo-protein to Rprime state")