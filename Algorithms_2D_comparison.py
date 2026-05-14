"""
2D Fourier Holography Algorithm Comparison
-----------------------------------------
This script benchmarks and visualizes multiple 2D phase-only hologram generation algorithms for computer-generated holography (CGH).

It loads a target image, applies various FFT-based phase retrieval algorithms (RS, GS, WGS, AWGS, BWC variants), and compares their reconstructed intensities and convergence (RMSE).

Usage:
	python Algorithms_2D_comparison.py

Dependencies:
	numpy, opencv-python, matplotlib, Algorithms_2D.py (in same folder)
"""
#%% Imports
import numpy as np
from Algorithms_2D import *
import cv2
import matplotlib.pyplot as plt

#%% Test algorithm on monkey
N, P = 600, 400  # Target is NxN, Signal Region is PxP
img = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
img = cv2.resize(img, (P, P))   
target = np.zeros((N, N), dtype=np.float32)
start = (N - P) // 2
target[start:start+P, start:start+P] = img

#%% --------- Badly sampled hologram results, can be skipped ----------
# hologram_GS, rmse_GS= GS(target,num_iterations=50)
# reco_GS = np.abs(SLM_to_FourierPlane(hologram_GS))**2
# plt.figure();plt.imshow(reco_GS, cmap='gray');plt.title("Reconstructed Intensity GS");plt.colorbar()
# hologram_WGS, rmse_WGS= WGS(target, num_iterations=10)
# reco_WGS = np.abs(SLM_to_FourierPlane(hologram_WGS))**2
# plt.figure();plt.imshow(reco_WGS, cmap='gray');plt.title("Reconstructed Intensity WGS");plt.colorbar()
# hologram_AWGS, rmse_AWGS= AWGS(target,num_iterations=50)
# reco_AWGS = np.abs(SLM_to_FourierPlane(hologram_AWGS))**2
# plt.figure();plt.imshow(reco_AWGS, cmap='gray');plt.title("Reconstructed Intensity AWGS");plt.colorbar()

#%% ---------- Resampled algorithm Testing ----------
hologram_RS = RS(target)
reco_rs = np.abs(SLM_to_FourierPlane(hologram_RS))**2
hologram_GS , rmse_gs= GS_resampled(target, num_iterations=30)
reco_GS = np.abs(SLM_to_FourierPlane(hologram_GS))**2
hologram_WGS , rmse_wgs= WGS_resampled(target, num_iterations=30)
reco_WGS = np.abs(SLM_to_FourierPlane(hologram_WGS))**2

hologram_AWGS , rmse_awgs= AWGS_resampled(target, num_iterations=30)
reco_AWGS = np.abs(SLM_to_FourierPlane(hologram_AWGS))**2

hologram_BWC_rand , rmse_bwc_rand= BWC_random(target, Q=4/6, num_iterations=30)
reco_BWC_rand = np.abs(SLM_to_FourierPlane(hologram_BWC_rand))**2

hologram_BWC_rand_nz , rmse_bwc_rand_nz= BWC_random_nz(target, num_iterations=30)
reco_BWC_rand_nz = np.abs(SLM_to_FourierPlane(hologram_BWC_rand_nz))**2

hologram_BWC_lens , rmse_bwc_lens= BWC_lens(target, Q=4/6, num_iterations=30)
reco_BWC_lens = np.abs(SLM_to_FourierPlane(hologram_BWC_lens))**2

#%% 
#Plot results
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()

axes[0].imshow(target.repeat(2, axis=0).repeat(2, axis=1), cmap='gray')
axes[0].set_title("Target Intensity")
axes[0].axis('off')
fig.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)

axes[1].imshow(reco_rs, cmap='gray')
axes[1].set_title("RS")
axes[1].axis('off')

axes[2].imshow(reco_GS, cmap='gray')
axes[2].set_title("GS_resampled")
axes[2].axis('off')

axes[3].imshow(reco_WGS, cmap='gray')
axes[3].set_title("WGS_resampled")
axes[3].axis('off')

axes[4].imshow(reco_AWGS, cmap='gray')
axes[4].set_title("AWGS_resampled")
axes[4].axis('off')

axes[5].imshow(reco_BWC_rand, cmap='gray')
axes[5].set_title("BWC_random")
axes[5].axis('off')

axes[6].imshow(reco_BWC_rand_nz, cmap='gray')
axes[6].set_title("BWC_random_nz")
axes[6].axis('off')

axes[7].imshow(reco_BWC_lens, cmap='gray')
axes[7].set_title("BWC_lens")
axes[7].axis('off')

plt.tight_layout()
plt.show()

#%% Figura 1: Immagini target e intensità ricostruite con RMSE
plt.figure()
plt.plot(rmse_gs, label="GS")
plt.plot(rmse_wgs, label="WGS")
plt.plot(rmse_awgs, label="AWGS")
plt.plot(rmse_bwc_rand, label="BWC_random")
plt.plot(rmse_bwc_rand_nz, label="BWC_random_nz")
plt.plot(rmse_bwc_lens, label="BWC_lens")
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.legend()
plt.grid()
plt.show()
