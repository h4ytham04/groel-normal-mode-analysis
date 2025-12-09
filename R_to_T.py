from prody import *
import prody
import numpy as np
from contact_map import ContactFrequency
import matplotlib.pyplot as plt

r_state = prody.parsePDB('4aas')
t_state = prody.parsePDB('1gr5')

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
    t_chain = t_state.select(f'calpha and chain {chain_id}')
    
    #match chains to handle missing residues
    matches = prody.matchChains(r_chain, t_chain)
    
    
    r_matched = matches[0][0]
    t_matched = matches[0][1]
    
    print(f'matched atoms: {r_matched.numAtoms()}')
    
    #anm on r state
    anm_r = ANM(f'R State Chain {chain_id} ANM analysis')
    anm_r.buildHessian(r_matched, cutoff=7, gamma=1)
    anm_r.calcModes()
    
    print(f'hessian shape: {anm_r.getHessian().shape}')
    
    anm_r_eigvecs = anm_r.getEigvecs()
    anm_r_eigvals = anm_r.getEigvals()
    
    #align structures
    t_aligned = t_matched.copy()
    calcTransformation(t_aligned, r_matched).apply(t_aligned)
    
    #r to t displacement
    delta = t_aligned.getCoords() - r_matched.getCoords()
    delta_vector = delta.flatten()


    prodyrmsd = calcRMSD(r_matched, t_aligned)
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
    print(f'mode {best_mode} best predicts r to t transition')
    print(f'overlap: {best_overlap:.4f}')
    
    # Store data for pseudo-protein construction
    best_mode_index = best_mode + 5  # Convert back to actual array index (mode 1 -> index 6)
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
print("="*60)
print("CREATING PSEUDO-PROTEIN FROM BEST MODES")
print("="*60)

# Create pseudo-protein by applying best mode displacements
all_pseudo_coords = []
all_original_atoms = []

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
    # This makes the displacement magnitude similar to the real conformational change
    scaling_factor = rmsd / np.sqrt(np.mean(np.sum(mode_coords**2, axis=1)))
    scaled_displacement = mode_coords * scaling_factor
    
    # Apply displacement to create pseudo-structure coordinates
    pseudo_coords = original_coords + scaled_displacement
    
    all_pseudo_coords.append(pseudo_coords)
    all_original_atoms.append(r_matched)
    
    print(f"Chain {chain_id}: Applied mode displacement (scaled by {scaling_factor:.2f})")

# Combine all chains into a single pseudo-protein
combined_pseudo_coords = np.vstack(all_pseudo_coords)

# Create a new AtomGroup for the pseudo-protein
# First, combine all original atom groups to get the structure template
pseudo_protein = all_original_atoms[0].copy()
for atom_group in all_original_atoms[1:]:
    pseudo_protein += atom_group

# Set the new coordinates
pseudo_protein.setCoords(combined_pseudo_coords)

# Save the pseudo-protein
prody.writePDB('pseudo_protein_best_modes.pdb', pseudo_protein)
print(f"\nPseudo-protein saved to: pseudo_protein_best_modes.pdb")
print(f"Total atoms in pseudo-protein: {pseudo_protein.numAtoms()}")

# Compare pseudo-protein to actual T state
print()
print("="*60)
print("COMPARING PSEUDO-PROTEIN TO ACTUAL T STATE")
print("="*60)

# Load full T state and match it to pseudo-protein structure
t_state_full = prody.parsePDB('1gr5')
t_calphas = t_state_full.select('calpha')

# Match to ensure same residues
matches_pseudo = prody.matchChains(pseudo_protein, t_calphas)
if matches_pseudo:
    pseudo_matched = matches_pseudo[0][0]
    t_matched_full = matches_pseudo[0][1]
    
    # Align for comparison
    t_aligned_full = t_matched_full.copy()
    calcTransformation(t_aligned_full, pseudo_matched).apply(t_aligned_full)
    
    # Calculate RMSD between pseudo-protein and actual T state
    pseudo_to_t_rmsd = calcRMSD(pseudo_matched, t_aligned_full)
    
    print(f"RMSD between pseudo-protein and actual T state: {pseudo_to_t_rmsd:.3f} Angstroms")
    print(f"This shows how well the best modes predict the conformational change!")
else:
    print("Could not match pseudo-protein to T state")

print("chungus chungus chungus")
