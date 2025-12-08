'''

for chain_id in chains_to_analyze:

    print()
    print()
    print(f'chain {chain_id}')
    print()
    print()

    t_calphas_all = t_state.select(f'calpha and chain {chain_id}')
    r_calphas_all = r_state.select(f'calpha and chain {chain_id}')
    
    t_resnums = set(t_calphas_all.getResnums())
    r_resnums = set(r_calphas_all.getResnums())
    common_resnums = sorted(t_resnums.intersection(r_resnums))

    resnum_str = ' '.join(map(str, common_resnums))
    t_calphas = t_state.select(f'calpha and chain {chain_id} and resnum {resnum_str}')
    r_calphas = r_state.select(f'calpha and chain {chain_id} and resnum {resnum_str}')

    gnm_t = GNM('T State')
    gnm_t.buildKirchhoff(t_calphas, cutoff=7, gamma=1)
    gnm_t.calcModes(n_modes=10)

    print("T state gnm")
    print("Kirchhoff shape:", gnm_t.getKirchhoff().shape)
    gnm_t_eigvals = gnm_t.getEigvals().round(3)
    gnm_t_eigvecs = gnm_t.getEigvecs().round(3)
    print("T State eigenvalues:", gnm_t_eigvals)

    #align
    r_aligned = r_calphas.copy()
    calcTransformation(r_aligned, t_calphas).apply(r_aligned)

    #t to r  displacement
    delta = r_aligned.getCoords() - t_calphas.getCoords()
    delta_magnitude = np.sqrt(np.sum(delta**2, axis=1))


    print("mode overlap")
    delta_norm = delta_magnitude / np.linalg.norm(delta_magnitude)

    for i in range(1, 11):
        mode = gnm_t_eigvecs[:, i]
        mode_norm = mode / np.linalg.norm(mode) #np.linalg.norm(mode) calculates distance from origin
        overlap = abs(np.dot(mode_norm, delta_norm))
        print(f"Mode {i}: overlap = {overlap:.4f}")

    print(f"RMSD: {np.sqrt(np.mean(delta_magnitude**2)):.3f} Angstroms")

    #best mode
    best_mode = np.argmax([abs(np.dot(gnm_t_eigvecs[:, i]/np.linalg.norm(gnm_t_eigvecs[:, i]), delta_norm)) for i in range(1, 11)]) + 1
    print(f"mode {best_mode} best predicts the T to R transition")
    print() 


'''