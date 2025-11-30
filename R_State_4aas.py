import numpy
import prody

r_state = prody.parsePDB('4aas')
rprime_state = prody.parsePDB('1svt')

#chain A
r_calphas = r_state.select('calpha and chain A')
rprime_calphas = rprime_state.select('calpha and chain A')

#matchChains instead
matches = prody.matchChains(r_calphas, rprime_calphas)

if matches:
    match = matches[0]
    r_matched = match[0]
    rprime_matched = match[1]
    
    print(f"After matching - R: {r_matched.numAtoms()}, R': {rprime_matched.numAtoms()}")
    
    rprime_aligned = rprime_matched.copy()
    prody.calcTransformation(rprime_aligned, r_matched).apply(rprime_aligned)
else:
    raise ValueError("Can't match chains")

gnm_r = prody.GNM('R State')
gnm_r.buildKirchhoff(r_matched, cutoff=7, gamma=1)
gnm_r.calcModes(n_modes=20)

print("R state gnm")
print("Kirchhoff shape:", gnm_r.getKirchhoff().shape)
gnm_r_eigvals = gnm_r.getEigvals().round(3)
gnm_r_eigvecs = gnm_r.getEigvecs().round(3)
print("R State eigenvalues:", gnm_r_eigvals)

#displacement
delta = rprime_aligned.getCoords() - r_matched.getCoords()
delta_magnitude = numpy.sqrt(numpy.sum(delta**2, axis=1))

print("mode overlap")
delta_norm = delta_magnitude / numpy.linalg.norm(delta_magnitude)

for i in range(1, 11):
    mode = gnm_r_eigvecs[:, i]
    mode_norm = mode / numpy.linalg.norm(mode)
    overlap = abs(numpy.dot(mode_norm, delta_norm))
    print(f"Mode {i}: overlap = {overlap:.4f}")

print(f"RMSD: {numpy.sqrt(numpy.mean(delta_magnitude**2)):.3f} Angstroms")

#best mode
best_mode = numpy.argmax([abs(numpy.dot(gnm_r_eigvecs[:, i]/numpy.linalg.norm(gnm_r_eigvecs[:, i]), delta_norm)) for i in range(1, 11)]) + 1
print(f"Mode {best_mode} best predicts the R to R' transition")

#chain B
print("chain b")
r_calphas_b = r_state.select('calpha and chain B')
rprime_calphas_b = rprime_state.select('calpha and chain B')

matches_b = prody.matchChains(r_calphas_b, rprime_calphas_b)
if matches_b:
    match_b = matches_b[0]
    r_matched_b = match_b[0]
    rprime_matched_b = match_b[1]
    
    rprime_aligned_b = rprime_matched_b.copy()
    prody.calcTransformation(rprime_aligned_b, r_matched_b).apply(rprime_aligned_b)
    
    delta_b = rprime_aligned_b.getCoords() - r_matched_b.getCoords()
    delta_magnitude_b = numpy.sqrt(numpy.sum(delta_b**2, axis=1))
    print(f"RMSD chain B: {numpy.sqrt(numpy.mean(delta_magnitude_b**2)):.3f} Angstroms")
    
    delta_norm_b = delta_magnitude_b / numpy.linalg.norm(delta_magnitude_b)
    for i in range(1, 11):
        mode = gnm_r_eigvecs[:, i]
        mode_norm = mode / numpy.linalg.norm(mode)
        overlap = abs(numpy.dot(mode_norm, delta_norm_b))
        print(f"Mode {i} chain B: overlap = {overlap:.4f}")
    
    best_mode_b = numpy.argmax([abs(numpy.dot(gnm_r_eigvecs[:, i]/numpy.linalg.norm(gnm_r_eigvecs[:, i]), delta_norm_b)) for i in range(1, 11)]) + 1
    print(f"Mode {best_mode_b} best predicts the R to R' transition for chain B")

