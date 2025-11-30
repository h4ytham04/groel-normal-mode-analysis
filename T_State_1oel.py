from prody import *
import prody
import numpy as np


t_state = prody.parsePDB('1oel')
r_state = prody.parsePDB('1svt')

#only do chain A for simplicity
t_calphas = t_state.select('calpha and chain A')
r_calphas = r_state.select('calpha and chain A')


gnm_t = GNM('T State')
gnm_t.buildKirchhoff(t_calphas, cutoff=7, gamma=1)
gnm_t.calcModes(n_modes=20)

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

#chain b
t_calphas_b = t_state.select('calpha and chain B')
r_calphas_b = r_state.select('calpha and chain B')
r_aligned_b = r_calphas_b.copy()
calcTransformation(r_aligned_b, t_calphas_b).apply(r_aligned_b)
#t to r  displacement
delta_b = r_aligned_b.getCoords() - t_calphas_b.getCoords()
delta_magnitude_b = np.sqrt(np.sum(delta_b**2, axis=1))
print(f"RMSD chain B: {np.sqrt(np.mean(delta_magnitude_b**2)):.3f} Angstroms")
#best mode chain B
delta_norm_b = delta_magnitude_b / np.linalg.norm(delta_magnitude_b)
for i in range(1, 11):
    mode = gnm_t_eigvecs[:, i]
    mode_norm = mode / np.linalg.norm(mode) #np.linalg.norm(mode) calculates distance from origin
    overlap = abs(np.dot(mode_norm, delta_norm_b))
    print(f"Mode {i} chain B: overlap = {overlap:.4f}")

best_mode_b = np.argmax([abs(np.dot(gnm_t_eigvecs[:, i]/np.linalg.norm(gnm_t_eigvecs[:, i]), delta_norm_b)) for i in range(1, 11)]) + 1
print(f"mode {best_mode_b} best predicts the T to R transition for chain B")