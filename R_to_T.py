from prody import *
import prody
import numpy as np
from contact_map import ContactFrequency
import matplotlib.pyplot as plt

r_state = prody.parsePDB('4aas')
t_state = prody.parsePDB('1gr5')

chains_to_analyze = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

results = []

optimal_anm_protein = [] #coords of the chains of the optimal ANM mode for the entire protein

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
    calcTransformation(t_aligned, r_matched).apply(t_aligned) #align t to r
    
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


