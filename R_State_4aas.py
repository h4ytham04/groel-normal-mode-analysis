import numpy as np
import prody

r_state = prody.parsePDB('4aas')
rprime_state = prody.parsePDB('1svt')


#chains in each state

#num of chains
print("r state num of chains")
chains = set(r_state.getChids())
print(len(chains))

#14 vs 21 chains


print("r prime state num of chains")
chains2 = set(rprime_state.getChids())
print(len(chains2))


chains_to_analyze = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

for chain_id in chains_to_analyze:

    print()
    print()
    print(f'chain {chain_id}')
    print()
    print()

    # Select C-alpha atoms for both structures
    r_calphas_all = r_state.select(f'calpha and chain {chain_id}')
    rprime_calphas_all = rprime_state.select(f'calpha and chain {chain_id}')
    
    # Match residues - only use residues present in BOTH structures
    r_resnums = set(r_calphas_all.getResnums())
    rprime_resnums = set(rprime_calphas_all.getResnums())
    common_resnums = sorted(r_resnums.intersection(rprime_resnums))
    
    # Create selection string for common residues
    resnum_str = ' '.join(map(str, common_resnums))
    r_calphas = r_state.select(f'calpha and chain {chain_id} and resnum {resnum_str}')
    rprime_calphas = rprime_state.select(f'calpha and chain {chain_id} and resnum {resnum_str}')

    gnm_r = prody.GNM('R State')
    gnm_r.buildKirchhoff(r_calphas, cutoff=7, gamma=1)
    gnm_r.calcModes(n_modes=20)

    print("R state gnm")
    print("Kirchhoff shape:", gnm_r.getKirchhoff().shape)
    gnm_r_eigvals = gnm_r.getEigvals().round(3)
    gnm_r_eigvecs = gnm_r.getEigvecs().round(3)
    print("R State eigenvalues:", gnm_r_eigvals)

    #align
    rprime_aligned = rprime_calphas.copy()
    prody.calcTransformation(rprime_aligned, r_calphas).apply(rprime_aligned)

    #r to r prime displacement
    delta = rprime_aligned.getCoords() - r_calphas.getCoords()
    delta_magnitude = np.sqrt(np.sum(delta**2, axis=1))

    print("mode overlap")
    delta_norm = delta_magnitude / np.linalg.norm(delta_magnitude)

    for i in range(1, 11):
        mode = gnm_r_eigvecs[:, i]
        mode_norm = mode / np.linalg.norm(mode) #np.linalg.norm(mode) calculates distance from origin
        overlap = abs(np.dot(mode_norm, delta_norm))
        print(f"Mode {i}: overlap = {overlap:.4f}")

    print(f"RMSD: {np.sqrt(np.mean(delta_magnitude**2)):.3f} Angstroms")

    #best mode
    best_mode = np.argmax([abs(np.dot(gnm_r_eigvecs[:, i]/np.linalg.norm(gnm_r_eigvecs[:, i]), delta_norm)) for i in range(1, 11)]) + 1
    print(f"mode {best_mode} best predicts the R to R' transition")
    print()

