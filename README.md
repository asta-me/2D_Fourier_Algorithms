# 2D Fourier Algorithms for Computer-Generated Holography

This repository provides a comprehensive suite of Python algorithms for generating 2D phase-only holograms (kinoforms) in the Fourier regime, using Fast Fourier Transform (FFT) methods. It includes both classic and advanced iterative phase retrieval techniques, as well as tools for benchmarking and visualizing their performance.

## Features

- **Multiple Algorithms:**
	- Random Superposition (RS)
	- Gerchberg–Saxton (GS) and Weighted GS (WGS)
	- Amplitude-Weighted GS (AWGS)
	- Resampled (upsampled) versions of GS, WGS, AWGS
	- Bandwidth-Constrained (BWC) algorithms with random, nonzero, and lens-based initialization

- **Comparison & Visualization Tools:**
	- Automated benchmarking and visualization of algorithm performance
	- Root Mean Square Error (RMSE) tracking for iterative methods
	- Region of Interest (ROI) selection for detailed analysis
	- Example plotting of reconstructed intensities and convergence curves

- **Sampling Correction:**
	- Both direct and resampled (2x upsampling) implementations to address Fourier plane sampling issues

## Dependencies

- numpy
- opencv-python
- matplotlib
- tqdm

## Usage

1. **Prepare a target intensity image** (e.g., PNG, grayscale).
2. **Run the main script** to generate and compare holograms:
	 ```
	 python Algorithms_2D_comparison.py
	 ```
	 - The script loads a target image, applies all implemented algorithms, simulates their reconstructions, and displays results.
	 - Plots are generated for both reconstructed intensities and RMSE convergence.
3. **Explore the code** in `Algorithms_2D.py` for detailed algorithm implementations and documentation (with extensive docstrings and references).

## File Overview

- **Algorithms_2D.py**  
	Core implementations of all phase retrieval and hologram generation algorithms, with detailed docstrings and references. Includes:
	- Direct and resampled (upsampled) algorithms
	- Bandwidth-constrained and lens-initialized variants
	- Utility functions for normalization and Fourier propagation

- **Algorithms_2D_comparison.py**  
	Example workflow for loading a target image, running all algorithms, visualizing reconstructions, and comparing RMSE. Now includes a clear header and improved documentation.

- **README.md**  
	Project overview and instructions.

## References

- Chen, Z., et al. "Bandwidth-constrained Gerchberg–Saxton algorithm for computer-generated holography." Optics Express 29.19 (2021): 30870-30883.
- Yang Wu, et al. "Adaptive weighted Gerchberg-Saxton algorithm for generation of phase-only hologram with artifacts suppression," Opt. Express 29, 1412-1427 (2021)

---
