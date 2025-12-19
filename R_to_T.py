from prody import *
import prody
import numpy as np
import matplotlib.pyplot as plt
import warnings
import csv

# Suppress ProDy warnings
warnings.filterwarnings('ignore')
prody.confProDy(verbosity='none')

r_state = prody.parsePDB('4aas')
t_state = prody.parsePDB('1gr5')

chains_to_analyze = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

results = []
optimal_chains = []
all_chain_overlaps = []

for chain_id in chains_to_analyze:
    
    print(f'\n=== CHAIN {chain_id} ===')
    
    r_chain = r_state.select(f'calpha and chain {chain_id}')
    t_chain = t_state.select(f'calpha and chain {chain_id}')
    
    matches = prody.matchChains(r_chain, t_chain)
    r_matched = matches[0][0]
    t_matched = matches[0][1]
    
    print(f'Matched atoms: {r_matched.numAtoms()}')
    
    anm_r = ANM(f'R State Chain {chain_id}')
    anm_r.buildHessian(r_matched, cutoff=7, gamma=1)
    anm_r.calcModes()
    
    anm_r_eigvecs = anm_r.getEigvecs()
    
    t_aligned = t_matched.copy()
    calcTransformation(t_aligned, r_matched).apply(t_aligned)
    
    delta = t_aligned.getCoords() - r_matched.getCoords()
    delta_vector = delta.flatten()

    prodyrmsd = calcRMSD(r_matched, t_aligned)
    print(f'RMSD: {prodyrmsd:.3f} Angstroms')
    
    print('\nMode overlaps:')
    overlaps = []
    for i in range(6, 16):
        mode = anm_r_eigvecs[:, i]
        mode_norm = mode / np.linalg.norm(mode)
        delta_norm = delta_vector / np.linalg.norm(delta_vector)
        overlap = abs(np.dot(mode_norm, delta_norm))
        overlaps.append(overlap)
        print(f'  Mode {i-5:2d}: {overlap:.4f}')
    
    best_mode = np.argmax(overlaps) + 1
    best_overlap = max(overlaps)
    
    all_chain_overlaps.append(overlaps)
    
    print(f'\n Best mode: {best_mode} (overlap: {best_overlap:.4f})')
    
    best_mode_index = best_mode + 5
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

# Save results to CSV
with open('r_to_t_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['chain', 'atoms', 'rmsd', 'best_mode', 'best_overlap'])
    writer.writeheader()
    writer.writerows(results)

# Summary statistics
ring1_rmsds = [r['rmsd'] for r in results[:7]]
ring2_rmsds = [r['rmsd'] for r in results[7:]]
ring1_overlaps = [r['best_overlap'] for r in results[:7]]
ring2_overlaps = [r['best_overlap'] for r in results[7:]]

print('\n' + '='*60)
print('SUMMARY STATISTICS')
print('='*60)
print(f'Ring 1 (A-G): RMSD = {np.mean(ring1_rmsds):.3f} ± {np.std(ring1_rmsds):.3f} Angstroms')
print(f'Ring 2 (H-N): RMSD = {np.mean(ring2_rmsds):.3f} ± {np.std(ring2_rmsds):.3f} Angstroms')
print(f'Ring 1 overlap = {np.mean(ring1_overlaps):.4f} ± {np.std(ring1_overlaps):.4f}')
print(f'Ring 2 overlap = {np.mean(ring2_overlaps):.4f} ± {np.std(ring2_overlaps):.4f}')

print('\n' + '='*60)
print('CREATING PSEUDO-PROTEIN')
print('='*60)

all_pseudo_coords = []
all_chain_data = []

for chain_data in optimal_chains:
    chain_id = chain_data['chain_id']
    r_matched = chain_data['r_matched']
    best_mode_vector = chain_data['best_mode_vector']
    rmsd = chain_data['rmsd']
    
    original_coords = r_matched.getCoords()
    mode_coords = best_mode_vector.reshape((-1, 3))
    
    scaling_factor = rmsd / np.sqrt(np.mean(np.sum(mode_coords**2, axis=1)))
    scaled_displacement = mode_coords * scaling_factor
    pseudo_coords = original_coords + scaled_displacement
    
    all_pseudo_coords.append(pseudo_coords)
    all_chain_data.append({
        'coords': pseudo_coords,
        'chain_id': chain_id,
        'atom_map': r_matched
    })
    
    print(f'Chain {chain_id}: scaling factor = {scaling_factor:.2f}')

combined_pseudo_coords = np.vstack(all_pseudo_coords)

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

pseudo_protein = prody.AtomGroup('Pseudo-protein R→T')
pseudo_protein.setCoords(combined_pseudo_coords)
pseudo_protein.setNames(all_names)
pseudo_protein.setResnums(all_resnums)
pseudo_protein.setResnames(all_resnames)
pseudo_protein.setChids(all_chains)
pseudo_protein.setElements(all_elements)

prody.writePDB('pseudo_protein_best_modes.pdb', pseudo_protein)
print(f'\n✓ Saved: pseudo_protein_best_modes.pdb ({pseudo_protein.numAtoms()} atoms)')

print('\n' + '='*60)
print('VALIDATION: PSEUDO-PROTEIN vs ACTUAL T STATE')
print('='*60)

t_state_full = prody.parsePDB('1gr5').select('calpha')

# Match chains by ID for global alignment
pseudo_coords_by_chain = {}
t_coords_by_chain = {}

for chain_id in chains_to_analyze:
    pseudo_chain = pseudo_protein.select(f'chain {chain_id}')
    t_chain = t_state_full.select(f'chain {chain_id}')
    
    if pseudo_chain and t_chain:
        min_atoms = min(pseudo_chain.numAtoms(), t_chain.numAtoms())
        pseudo_coords_by_chain[chain_id] = pseudo_chain.getCoords()[:min_atoms]
        t_coords_by_chain[chain_id] = t_chain.getCoords()[:min_atoms]

all_pseudo = np.vstack(list(pseudo_coords_by_chain.values()))
all_t = np.vstack(list(t_coords_by_chain.values()))

print(f'Matched {len(all_pseudo)} atoms across all chains')

# Single global alignment
transformation = calcTransformation(all_t, all_pseudo)
all_t_aligned = transformation.apply(all_t)

# Final RMSD
final_rmsd = np.sqrt(np.mean(np.sum((all_pseudo - all_t_aligned)**2, axis=1)))
print(f'\n✓ Global RMSD: {final_rmsd:.3f} Angstroms ({len(all_pseudo)} atoms)')

# Generate heatmap
print('\n' + '='*60)
print('GENERATING HEATMAP')
print('='*60)

overlap_matrix = np.array(all_chain_overlaps)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(overlap_matrix, cmap='hot', aspect='auto', interpolation='nearest')

ax.set_xticks(np.arange(10))
ax.set_xticklabels(np.arange(1, 11))
ax.set_yticks(np.arange(14))
ax.set_yticklabels(chains_to_analyze)

ax.set_xlabel('Mode Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Chain ID', fontsize=12, fontweight='bold')
ax.set_title('Mode Overlap: R → T Transition\n(GroEL Apo State)', 
             fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Overlap Coefficient', rotation=270, labelpad=20, fontsize=11)

ax.axhline(y=6.5, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
ax.text(10.3, 3, 'Ring 1\n(cis)', va='center', fontsize=10, color='cyan', fontweight='bold')
ax.text(10.3, 10, 'Ring 2\n(trans)', va='center', fontsize=10, color='cyan', fontweight='bold')

for i in range(14):
    for j in range(10):
        text = ax.text(j, i, f'{overlap_matrix[i, j]:.3f}',
                      ha="center", va="center", color="white", fontsize=8)

plt.tight_layout()
plt.savefig('mode_overlap_heatmap.png', dpi=300, bbox_inches='tight')
print('✓ Heatmap saved: mode_overlap_heatmap.png')

print('\n' + '='*60)
print('ANALYSIS COMPLETE')
print('='*60)
