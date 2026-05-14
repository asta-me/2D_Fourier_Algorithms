"""
================================================================================
2D Algorithms for Fourier Transform (FFT) Computer-Generated Holography
================================================================================

This script provides a collection of Python functions for generating 2D phase-only
holograms (kinoforms) from a target intensity image using Fast Fourier Transform
(FFT) based methods. It includes implementations of several popular iterative
and non-iterative phase retrieval algorithms.

The primary goal of these algorithms is to calculate a phase pattern for a
spatial light modulator (SLM) that, when illuminated, reconstructs the desired
intensity distribution in the far-field (Fourier plane).

Implemented Algorithms
----------------------
The script contains two main categories of algorithms:

1.  **Simplified Implementations (Without Corrected Sampling):**
    These are direct implementations that operate on arrays of the same size as
    the target. Very commonly used, they do not enforce the Nyquist sampling condition required for
    accurate sampling in the Fourier plane, which can result in the optimisation of an 
    aliased, non-physical reconstruction.
    They are here implemented for hystorical reasons, and for future
    users not to come tell me that my GS is "too slow".
    - `RS`: Random Superposition (a simple, non-iterative method).
    - `GS`: The classic iterative Gerchberg-Saxton algorithm.
    - `WGS`: Weighted Gerchberg-Saxton, which introduces a weighting factor
             to improve convergence.
    - `AWGS`: Amplitude-Weighted Gerchberg-Saxton, which uses an exponential
             weight for amplitude correction.

2.  **Resampled Implementations (With Corrected Sampling):**
    NxN targets will be resampled to 2Nx2N and 
    produce a NxN phase only hologram.
    - `GS_resampled`: Gerchberg-Saxton with 2x upsampling.
    - `WGS_resampled`: Weighted GS with 2x upsampling.
    - `AWGS_resampled`: Amplitude-Weighted GS with 2x upsampling.
    - `BWC_random` & `BWC_lens`: Bandwidth-Constrained (BWC) algorithm. This
      method enforces the target amplitude within a defined "signal region"
      while allowing freedom in the surrounding "noise region." Implementations
      with both random and quadratic ("lens") phase initializations are provided.

Dependencies
------------
- NumPy
- OpenCV-Python
- Matplotlib
- TQDM

Usage
-----
The main execution block (`if __name__ == '__main__':`) demonstrates a complete
workflow:
1. Load and prepare a target intensity image.
2. Generate holograms using each of the implemented algorithms.
3. Simulate the optical reconstruction from each hologram via (padded)FFT.
4. Display the final reconstructed images and plot the Root Mean Square
   Error (RMSE) to compare the convergence of the iterative methods.

Note
----
The `WGS` and `WGS_resampled` functions are unstable in their
current implementation and may not produce valid results.

References
---------------------------
- Chen, Z., et al. "Bandwidth-constrained Gerchberg–Saxton algorithm for
  computer-generated holography." Optics Express 29.19 (2021): 30870-30883.

- Yang Wu, et al. "Adaptive weighted Gerchberg-Saxton algorithm for generation 
  of phase-only hologram with artifacts suppression," Opt. Express 29, 1412-1427 (2021)
  """

#%% Imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2,ifft2,fftshift,ifftshift
from tqdm import tqdm

def norm(matrix):
    min=np.min(matrix);max=np.max(matrix);
    return((matrix-min)/(max-min))

def RS(target):
    """
    Random Superposition (RS) algorithm for phase-only hologram generation.

    This algorithm generates a random initial phase, combines it with the
    normalized target amplitude, and computes the Fourier transform to
    obtain a phase-only hologram. 

    Note:
        This implementation does **not** enforce correct Fourier sampling
        conditions, so the reconstruction may be suboptimal.

    Args:
        target (np.ndarray): 
            2D array representing the desired target intensity pattern.
            Values are automatically normalized to [0, 1].

    Returns:
        np.ndarray: 
            2D array containing the phase-only hologram (values in [-π, π]).
    """
    # Normalize input to [0, 1]
    target = norm(target)

    # Generate random phase uniformly distributed in [-π, π]
    random_phase = np.random.uniform(-np.pi, np.pi, target.shape)

    # Construct complex field with target amplitude and random phase
    field = np.sqrt(target) * np.exp(1j * random_phase)

    # Compute hologram: FFT of field -> extract phase
    hologram_phase = np.angle(
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    )

    return hologram_phase

def GS(target, num_iterations=10, compute_rmse=True):
    """
    Gerchberg–Saxton (GS) algorithm for phase-only hologram generation.

    The GS algorithm iteratively enforces amplitude constraints in the
    target plane and hologram (SLM) plane to recover a phase pattern
    that reproduces the desired target intensity upon propagation.

    Args:
        target (np.ndarray):
            2D target intensity pattern. Automatically normalized to [0, 1].
        num_iterations (int):
            Number of GS iterations (default: 10).
        compute_rmse (bool):
            If True, compute and return the RMSE between the reconstructed 
            intensity and the target at each iteration.

    Returns:
        tuple:
            - phase (np.ndarray): Final phase-only hologram (values in [-π, π]).
            - rmse_values (list[float]): RMSE at each iteration (empty if compute_rmse=False).
    """
    # Normalize target to [0, 1]
    target = norm(target)

    # Total energy in target plane
    total_energy = np.sum(target)

    # Fixed amplitude in hologram plane (energy conservation)
    hologram_amplitude = 1.0 / np.sqrt(total_energy)

    # Initialization: random phase in [-π, π]
    random_phase = np.random.uniform(-np.pi, np.pi, target.shape)
    target_amplitude = np.sqrt(target)
    field = target_amplitude * np.exp(1j * random_phase)

    rmse_values = []

    # Main iteration loop
    for _ in tqdm(range(num_iterations), desc="GS Iterations", unit="iter"):
        # Forward propagation: object plane -> hologram plane
        hologram_field = fftshift(fft2(ifftshift(field)))

        # Enforce amplitude constraint in hologram plane
        hologram_field = hologram_amplitude * np.exp(1j * np.angle(hologram_field))

        # Back propagation: hologram plane -> object plane
        field = fftshift(ifft2(ifftshift(hologram_field)))

        # Compute normalized intensity in object plane
        intensity = np.abs(field) ** 2
        intensity = norm(intensity)

        # Enforce amplitude constraint in object plane
        field = target_amplitude * np.exp(1j * np.angle(field))

        # Optionally compute RMSE
        if compute_rmse:
            rmse = np.sqrt(np.mean((intensity - target) ** 2))
            rmse_values.append(rmse)

    return np.angle(hologram_field), rmse_values

def WGS(target, num_iterations=10, alpha=0.2, eps=1e-5, weight_clip=1e2, compute_rmse=True):
    """
    Weighted Gerchberg–Saxton (WGS) algorithm for phase-only hologram generation.

    The WGS algorithm introduces a weighting factor that progressively
    corrects the amplitude mismatch between the reconstructed field
    and the target, improving convergence compared to standard GS.

    Args:
        target (np.ndarray):
            2D target intensity pattern. Automatically normalized to [0, 1].
        num_iterations (int):
            Number of iterations (default: 10).
        alpha (float):
            Weighting exponent controlling the update strength (default: 1).
        eps (float):
            Small constant added to avoid division by zero (default: 0).
        weight_clip (float): Maximum weight value (default: 1e2).
        compute_rmse (bool):
            If True, compute and return the RMSE between the reconstructed 
            intensity and the target at each iteration.

    Returns:
        tuple:
            - phase (np.ndarray): Final phase-only hologram (values in [-π, π]).
            - rmse_values (list[float]): RMSE at each iteration (empty if compute_rmse=False).
    """
    # Normalize target to [0, 1]
    target = norm(target)

    # Total energy in target plane
    total_energy = np.sum(target)

    # Fixed amplitude in hologram plane (energy conservation)
    hologram_amplitude = 1.0 / np.sqrt(total_energy)

    # Initialization: random phase in [-π, π]
    random_phase = np.random.uniform(-np.pi, np.pi, target.shape)
    target_amplitude = np.sqrt(target)
    field = target_amplitude * np.exp(1j * random_phase)

    # Initialize weights (all ones)
    weights = np.ones(target.shape)
    rmse_values = []

    # Main iteration loop
    for _ in tqdm(range(num_iterations), desc="WGS Iterations", unit="iter"):
        # Forward propagation: object plane -> hologram plane
        hologram_field = fftshift(fft2(ifftshift(field)))

        # Enforce amplitude constraint in hologram plane
        hologram_field = hologram_amplitude * np.exp(1j * np.angle(hologram_field))

        # Back propagation: hologram plane -> object plane
        field = fftshift(ifft2(ifftshift(hologram_field)))

        # Compute normalized intensity in object plane
        intensity = np.abs(field) ** 2
        intensity = norm(intensity)

        # Update weights (progressively correct mismatch)
        weights *= (target / (intensity + eps)) ** alpha
        weights = np.clip(weights, 0, weight_clip)  # Prevent extreme weights
        
        # Apply updated amplitude constraint
        field = weights * target_amplitude * np.exp(1j * np.angle(field))

        # Optionally compute RMSE
        if compute_rmse:
            rmse = np.sqrt(np.mean((intensity - target) ** 2))
            rmse_values.append(rmse)

    return np.angle(hologram_field), rmse_values

def AWGS(target, num_iterations=10, compute_rmse=True):
    """
    Amplitude-Weighted Gerchberg–Saxton (AWGS) algorithm.

    AWGS is a variant of WGS where the amplitude correction is applied
    using an exponential weight:
    
        weights = exp(target - intensity)
    
    This tends to provide a smoother correction and can improve 
    convergence in some cases.

    Args:
        target (np.ndarray):
            2D target intensity pattern. Automatically normalized to [0, 1].
        num_iterations (int):
            Number of iterations (default: 10).
        compute_rmse (bool):
            If True, compute and return the RMSE between the reconstructed 
            intensity and the target at each iteration.

    Returns:
        tuple:
            - phase (np.ndarray): Final phase-only hologram (values in [-π, π]).
            - rmse_values (list[float]): RMSE at each iteration (empty if compute_rmse=False).
    """
    # Normalize target to [0, 1]
    target = norm(target)

    # Total energy in target plane
    total_energy = np.sum(target)

    # Fixed amplitude in hologram plane (energy conservation)
    hologram_amplitude = 1.0 / np.sqrt(total_energy)

    # Initialization: random phase in [-π, π]
    random_phase = np.random.uniform(-np.pi, np.pi, target.shape)
    target_amplitude = np.sqrt(target)
    field = target_amplitude * np.exp(1j * random_phase)

    # Initialize weights (all ones)
    weights = np.ones(target.shape)
    rmse_values = []

    # Main iteration loop
    for _ in tqdm(range(num_iterations), desc="AWGS Iterations", unit="iter"):
        # Forward propagation: object plane -> hologram plane
        hologram_field = fftshift(fft2(ifftshift(field)))

        # Enforce amplitude constraint in hologram plane
        hologram_field = hologram_amplitude * np.exp(1j * np.angle(hologram_field))

        # Back propagation: hologram plane -> object plane
        field = fftshift(ifft2(ifftshift(hologram_field)))

        # Compute normalized intensity in object plane
        intensity = np.abs(field) ** 2
        intensity = norm(intensity)

        # Update weights using exponential correction
        weights = np.exp(target - intensity)

        # Apply updated amplitude constraint
        field = weights * target_amplitude * np.exp(1j * np.angle(field))

        # Optionally compute RMSE
        if compute_rmse:
            rmse = np.sqrt(np.mean((intensity - target) ** 2))
            rmse_values.append(rmse)

    return np.angle(hologram_field), rmse_values

def GS_resampled(target, num_iterations=10, resampling="sinc", compute_rmse=True):
    """
    Gerchberg–Saxton (GS) algorithm with resampling.

    The target intensity is upsampled (by a factor of 2) to improve
    the reconstruction quality, then the GS algorithm is applied.

    Args:
        target (np.ndarray):
            2D target intensity pattern. Automatically normalized to [0, 1].
        num_iterations (int):
            Number of GS iterations (default: 10).
        resampling (str):
            Method for upsampling the target: "nearest" (default) or "sinc".
        compute_rmse (bool):
            If True, compute RMSE at each iteration.

    Returns:
        tuple:
            - phase (np.ndarray): Cropped phase-only hologram (values in [-π, π]).
            - intensity (np.ndarray): Final reconstructed intensity (normalized).
            - rmse_values (list[float]): RMSE per iteration (empty if compute_rmse=False).
    """
    target = norm(target)
    n = target.shape[0]

    # --- Resampling ---
    if resampling == "sinc":
        freq = fftshift(fft2(target))
        freq = np.pad(freq, ((n//2, n//2), (n//2, n//2)), mode="constant")
        target_resampled = np.abs(ifft2(ifftshift(freq))) * 4
    elif resampling == "nearest":
        target_resampled = target.repeat(2, axis=0).repeat(2, axis=1)
    else:
        raise ValueError("Resampling method must be 'nearest' or 'sinc'.")
    
    target_resampled = norm(target_resampled)

    # Initialization
    target_amplitude = np.sqrt(target_resampled)
    random_phase = np.random.uniform(-np.pi, np.pi, target_resampled.shape)
    field = target_amplitude * np.exp(1j * random_phase)

    # Hologram amplitude (energy conservation)
    hologram_amplitude = np.pad(np.ones((n, n)), ((n//2, n//2), (n//2, n//2)), mode="constant")
    hologram_amplitude *= np.sum(target_resampled) / np.sum(hologram_amplitude)
    hologram_amplitude = np.sqrt(hologram_amplitude)

    rmse_values = []

    for _ in tqdm(range(num_iterations), desc="GS Resampled Iterations", unit="iter"):
        hologram_field = fftshift(ifft2(ifftshift(field)))
        phase = np.angle(hologram_field)
        hologram_field = hologram_amplitude * np.exp(1j * phase)
        field = fftshift(fft2(fftshift(hologram_field)))
        intensity = norm(np.abs(field) ** 2)
        field = target_amplitude * np.exp(1j * np.angle(field))

        if compute_rmse:
            rmse_values.append(np.sqrt(np.mean((intensity - target_resampled) ** 2)))

    # Crop back to original size
    center = n
    crop = n // 2
    phase_cropped = phase[center - crop:center + crop, center - crop:center + crop]

    return phase_cropped, rmse_values

def WGS_resampled(target, num_iterations=10, alpha=1, eps=1e-5, weight_clip=1e2, resampling="sinc", compute_rmse=True):
    """
    Weighted Gerchberg–Saxton (WGS) with resampling.

    Similar to GS_resampled but introduces a weighting factor to
    iteratively correct amplitude mismatch.

    Args:
        target (np.ndarray): Target intensity pattern, normalized internally.
        num_iterations (int): Number of iterations.
        alpha (float): Weighting exponent.
        eps (float): Small constant to avoid division by zero.
        weight_clip (float): Maximum weight value.
        resampling (str): "nearest" (default) or "sinc".
        compute_rmse (bool): Compute RMSE if True.

    Returns:
        tuple:
            - phase (np.ndarray): Cropped phase-only hologram.
            - intensity (np.ndarray): Final reconstructed intensity.
            - rmse_values (list[float]): RMSE per iteration.
    """
    target = norm(target)
    n = target.shape[0]

    # --- Resampling ---
    if resampling == "sinc":
        freq = fftshift(fft2(target))
        freq = np.pad(freq, ((n//2, n//2), (n//2, n//2)), mode="constant")
        target_resampled = np.abs(ifft2(ifftshift(freq))) * 4
    elif resampling == "nearest":
        target_resampled = target.repeat(2, axis=0).repeat(2, axis=1)
    else:
        raise ValueError("Resampling method must be 'nearest' or 'sinc'.")
    
    target_resampled = norm(target_resampled)

    target_amplitude = np.sqrt(target_resampled)
    random_phase = np.random.uniform(-np.pi, np.pi, target_resampled.shape)
    field = target_amplitude * np.exp(1j * random_phase)

    hologram_amplitude = np.pad(np.ones((n, n)), ((n//2, n//2), (n//2, n//2)), mode="constant")
    hologram_amplitude *= np.sum(target_resampled) / np.sum(hologram_amplitude)
    hologram_amplitude = np.sqrt(hologram_amplitude)

    weights = np.ones_like(target_resampled)
    rmse_values = []

    for _ in tqdm(range(num_iterations), desc="WGS Resampled Iterations", unit="iter"):
        hologram_field = fftshift(ifft2(ifftshift(field)))
        phase = np.angle(hologram_field)
        hologram_field = hologram_amplitude * np.exp(1j * phase)
        field = fftshift(fft2(fftshift(hologram_field)))
        intensity = norm(np.abs(field) ** 2)

        weights *= (target_resampled / (intensity + eps)) ** alpha
        weights= np.clip(weights, 0, weight_clip)  # Prevent extreme weights
        print(np.max(weights))
        print(weights)
        field = weights * target_amplitude * np.exp(1j * np.angle(field))

        if compute_rmse:
            rmse_values.append(np.sqrt(np.mean((intensity - target_resampled) ** 2)))

    center = n
    crop = n // 2
    phase_cropped = phase[center - crop:center + crop, center - crop:center + crop]

    return phase_cropped, rmse_values

def AWGS_resampled(target, num_iterations=10, resampling="sinc", compute_rmse=True):
    """
    Amplitude-Weighted GS (AWGS) with resampling.

    Variant of GS_resampled where the amplitude mismatch correction
    is applied using exponential weights:

        weights = exp(target_resampled - intensity)

    Args:
        target (np.ndarray): Target intensity pattern, normalized internally.
        num_iterations (int): Number of iterations.
        resampling (str): "nearest" (default) or "sinc".
        compute_rmse (bool): Compute RMSE if True.

    Returns:
        tuple:
            - phase (np.ndarray): Cropped phase-only hologram.
            - intensity (np.ndarray): Final reconstructed intensity.
            - rmse_values (list[float]): RMSE per iteration.
    """
    target = norm(target)
    n = target.shape[0]

    # --- Resampling ---
    if resampling == "sinc":
        freq = fftshift(fft2(target))
        freq = np.pad(freq, ((n//2, n//2), (n//2, n//2)), mode="constant")
        target_resampled = np.abs(ifft2(ifftshift(freq))) * 4
    elif resampling == "nearest":
        target_resampled = target.repeat(2, axis=0).repeat(2, axis=1)
    else:
        raise ValueError("Resampling method must be 'nearest' or 'sinc'.")
    
    target_resampled = norm(target_resampled)

    target_amplitude = np.sqrt(target_resampled)
    random_phase = np.random.uniform(-np.pi, np.pi, target_resampled.shape)
    field = target_amplitude * np.exp(1j * random_phase)

    hologram_amplitude = np.pad(np.ones((n, n)), ((n//2, n//2), (n//2, n//2)), mode="constant")
    hologram_amplitude *= np.sum(target_resampled) / np.sum(hologram_amplitude)
    hologram_amplitude = np.sqrt(hologram_amplitude)

    weights = np.ones_like(target_resampled)
    rmse_values = []

    for _ in tqdm(range(num_iterations), desc="AWGS Resampled Iterations", unit="iter"):
        hologram_field = fftshift(ifft2(ifftshift(field)))
        phase = np.angle(hologram_field)
        hologram_field = hologram_amplitude * np.exp(1j * phase)
        field = fftshift(fft2(fftshift(hologram_field)))
        intensity = norm(np.abs(field) ** 2)

        weights = np.exp(target_resampled - intensity)
        field = weights * target_amplitude * np.exp(1j * np.angle(field))

        if compute_rmse:
            rmse_values.append(np.sqrt(np.mean((intensity - target_resampled) ** 2)))

    center = n
    crop = n // 2
    phase_cropped = phase[center - crop:center + crop, center - crop:center + crop]

    return phase_cropped, rmse_values


    # Normalize target to 0-255
    target = norm(target)
    n = target.shape[0]
    
    # Resampling target using sinc interpolation
    xx = fftshift(fft2(target))
    xx = np.pad(xx, ((n//2, n//2), (n//2, n//2)), mode='constant')
    target_resampled = np.abs(ifft2(ifftshift(xx)) * 4)
    target_resampled = norm(target_resampled)
    
    # Signal and Noise Region Definition
    Q_size = int(Q * 2*n)       # Calcola la dimensione del quadrato centrale
    SR = np.zeros((2*n, 2*n)) # Creiamo una matrice di tutti 0
    start_idx = (2*n - Q_size) // 2     # Calcoliamo gli indici per centrare il quadrato di 1
    end_idx = start_idx + Q_size
    SR[start_idx:end_idx, start_idx:end_idx] = 1 # Impostiamo a 1 il quadrato centrale
    NR = 1 - SR # NR è il complemento di SR
    
    # Initialization Step
    A0 = np.sqrt(target_resampled)
    x = np.linspace(-1, 1, 2*n ) * n
    X, Y = np.meshgrid(x, x)
    rho = np.hypot(X, Y)
    fi_0 = np.pi * (rho**2) / (2 * Q * target_resampled.shape[0])
    fi_0 = np.mod(fi_0, 2*np.pi) - np.pi
    A = A0 * np.exp(1j * fi_0)
    
    # Fixed amplitude in hologram plane
    H_ampl = np.pad(np.ones((n, n)), ((n//2, n//2), (n//2, n//2)), mode='constant')
    H_ampl *= np.sum(target_resampled) / np.sum(H_ampl)
    H_ampl = np.sqrt(H_ampl)
    
    RMSE = np.zeros(F)
    
    # Main Iteration Loop
    for f in tqdm(range(F), desc="BWC lens Progress"):
        H = fftshift(ifft2(ifftshift(A)))
        kino = np.angle(H)
        H = H_ampl * np.exp(1j * kino)
        A = fftshift(fft2(fftshift(H)))
        A_int = np.abs(A)**2
        # A_int *= np.sum(target_resampled) / np.sum(A_int)
        A_int=norm(A_int)
        Phase_out = np.angle(A)
        A_Ampl = (SR * A0) + (np.sqrt(A_int) * NR)
        A = A_Ampl * np.exp(1j * Phase_out)
        RMSE[f] = np.sum((A_int - target_resampled)**2) / (np.sum(target_resampled)**2)
    
    # Cropping the kinoform to the central region
    center = n
    crop_size = n//2
    kinoform = kino[center-crop_size:center+crop_size, center-crop_size:center+crop_size]
    return kinoform, A_int, RMSE 

def BWC_random(target, Q, num_iterations=10, resampling="sinc", compute_rmse=True):
    """
    Bandwidth-Constrained (BWC) Gerchberg–Saxton with random initialization.

    This algorithm applies a bandwidth constraint by splitting the field into
    a central "signal region" (SR) and its complement "noise region" (NR).
    The SR enforces the target amplitude, while NR is allowed to adapt.

    Args:
        target (np.ndarray):
            2D target intensity pattern. Automatically normalized to [0, 1].
        Q (float):
            Fraction of the total size used for the square signal region
            (e.g. Q=0.5 means the central square covers 50% of the width).
        num_iterations (int):
            Number of iterations (default: 10).
        resampling (str):
            "nearest" (default) or "sinc" for upsampling the target.
        compute_rmse (bool):
            Compute RMSE at each iteration if True.

    Returns:
        tuple:
            - phase (np.ndarray): Cropped phase-only hologram ([-π, π]).
            - intensity (np.ndarray): Final reconstructed intensity (normalized).
            - rmse_values (list[float]): RMSE per iteration.
    """
    target = norm(target)
    n = target.shape[0]

    # --- Resampling ---
    if resampling == "sinc":
        freq = fftshift(fft2(target))
        freq = np.pad(freq, ((n//2, n//2), (n//2, n//2)), mode="constant")
        target_resampled = np.abs(ifft2(ifftshift(freq))) * 4
    elif resampling == "nearest":
        target_resampled = target.repeat(2, axis=0).repeat(2, axis=1)
    else:
        raise ValueError("Resampling method must be 'nearest' or 'sinc'.")
    
    target_resampled = norm(target_resampled)

    # Define Signal Region (SR) and Noise Region (NR)
    Q_size = int(Q * 2 * n)
    SR = np.zeros((2 * n, 2 * n))
    start = (2 * n - Q_size) // 2
    SR[start:start + Q_size, start:start + Q_size] = 1
    NR = 1 - SR

    target_amplitude = np.sqrt(target_resampled)
    random_phase = np.random.uniform(-np.pi, np.pi, target_resampled.shape)
    field = target_amplitude * np.exp(1j * random_phase)

    hologram_amplitude = np.pad(np.ones((n, n)), ((n//2, n//2), (n//2, n//2)), mode="constant")
    hologram_amplitude *= np.sum(target_resampled) / np.sum(hologram_amplitude)
    hologram_amplitude = np.sqrt(hologram_amplitude)

    rmse_values = []

    for _ in tqdm(range(num_iterations), desc="BWC Random Iterations", unit="iter"):
        hologram_field = fftshift(ifft2(ifftshift(field)))
        phase = np.angle(hologram_field)
        hologram_field = hologram_amplitude * np.exp(1j * phase)
        field = fftshift(fft2(fftshift(hologram_field)))

        intensity = norm(np.abs(field) ** 2)
        updated_amplitude = (SR * target_amplitude) + (np.sqrt(intensity) * NR)
        field = updated_amplitude * np.exp(1j * np.angle(field))

        if compute_rmse:
            rmse_values.append(np.sqrt(np.mean((intensity - target_resampled) ** 2)))

    center = n
    crop = n // 2
    phase_cropped = phase[center - crop:center + crop, center - crop:center + crop]

    return phase_cropped, rmse_values

def BWC_random_nz(target, num_iterations=10, resampling="sinc", compute_rmse=True):
    """
    Bandwidth-Constrained GS with random initialization (non-centered SR).

    Like BWC_random, but the Signal Region (SR) is defined by the nonzero
    support of the target instead of a fixed central square. This is
    useful for decentered or irregularly shaped targets.

    Args:
        target (np.ndarray): Target intensity pattern, normalized internally.
        num_iterations (int): Number of iterations (default: 10).
        resampling (str): "nearest" (default) or "sinc".
        compute_rmse (bool): Compute RMSE if True.

    Returns:
        tuple:
            - phase (np.ndarray): Cropped phase-only hologram.
            - intensity (np.ndarray): Final reconstructed intensity.
            - rmse_values (list[float]): RMSE per iteration.
    """
    target = norm(target)
    n = target.shape[0]

    # --- Resampling ---
    if resampling == "sinc":
        freq = fftshift(fft2(target))
        freq = np.pad(freq, ((n//2, n//2), (n//2, n//2)), mode="constant")
        target_resampled = np.abs(ifft2(ifftshift(freq))) * 4
    elif resampling == "nearest":
        target_resampled = target.repeat(2, axis=0).repeat(2, axis=1)
    else:
        raise ValueError("Resampling method must be 'nearest' or 'sinc'.")

    target_resampled = norm(target_resampled)

    # Signal Region from target support
    SR = (target_resampled > 0).astype(float)
    NR = 1 - SR

    target_amplitude = np.sqrt(target_resampled)
    random_phase = np.random.uniform(-np.pi, np.pi, target_resampled.shape)
    field = target_amplitude * np.exp(1j * random_phase)

    hologram_amplitude = np.pad(np.ones((n, n)), ((n//2, n//2), (n//2, n//2)), mode="constant")
    hologram_amplitude *= np.sum(target_resampled) / np.sum(hologram_amplitude)
    hologram_amplitude = np.sqrt(hologram_amplitude)

    rmse_values = []

    for _ in tqdm(range(num_iterations), desc="BWC Random-nz Iterations", unit="iter"):
        hologram_field = fftshift(ifft2(ifftshift(field)))
        phase = np.angle(hologram_field)
        hologram_field = hologram_amplitude * np.exp(1j * phase)
        field = fftshift(fft2(fftshift(hologram_field)))

        intensity = norm(np.abs(field) ** 2)
        updated_amplitude = (SR * target_amplitude) + (np.sqrt(intensity) * NR)
        field = updated_amplitude * np.exp(1j * np.angle(field))

        if compute_rmse:
            rmse_values.append(np.sqrt(np.mean((intensity - target_resampled) ** 2)))

    center = n
    crop = n // 2
    phase_cropped = phase[center - crop:center + crop, center - crop:center + crop]

    return phase_cropped, rmse_values

def BWC_lens(target, Q, num_iterations=10, resampling="sinc", compute_rmse=True):
    """
    Bandwidth-Constrained GS with lens-based initialization.

    Similar to BWC_random, but initializes the phase with a quadratic
    "lens" term to accelerate convergence.

    Args:
        target (np.ndarray):
            Target intensity pattern. Automatically normalized to [0, 1].
        Q (float):
            Fraction of the total size used for the central square (signal region).
        num_iterations (int):
            Number of iterations (default: 10).
        resampling (str):
            "sinc" (default) or "nearest".
        compute_rmse (bool):
            Compute RMSE if True.

    Returns:
        tuple:
            - phase (np.ndarray): Cropped phase-only hologram.
            - intensity (np.ndarray): Final reconstructed intensity.
            - rmse_values (list[float]): RMSE per iteration.
    """
    target = norm(target)
    n = target.shape[0]

    # --- Resampling ---
    if resampling == "sinc":
        freq = fftshift(fft2(target))
        freq = np.pad(freq, ((n//2, n//2), (n//2, n//2)), mode="constant")
        target_resampled = np.abs(ifft2(ifftshift(freq))) * 4
    elif resampling == "nearest":
        target_resampled = target.repeat(2, axis=0).repeat(2, axis=1)
    else:
        raise ValueError("Resampling method must be 'nearest' or 'sinc'.")
    
    target_resampled = norm(target_resampled)

    # Signal/Noise Region
    Q_size = int(Q * 2 * n)
    SR = np.zeros((2 * n, 2 * n))
    start = (2 * n - Q_size) // 2
    SR[start:start + Q_size, start:start + Q_size] = 1
    NR = 1 - SR

    # Initialization with quadratic lens phase
    target_amplitude = np.sqrt(target_resampled)
    x = np.linspace(-1, 1, 2 * n) * n
    X, Y = np.meshgrid(x, x)
    rho = np.hypot(X, Y)
    lens_phase = np.mod(np.pi * (rho ** 2) / (2 * Q * n), 2 * np.pi) - np.pi
    field = target_amplitude * np.exp(1j * lens_phase)

    hologram_amplitude = np.pad(np.ones((n, n)), ((n//2, n//2), (n//2, n//2)), mode="constant")
    hologram_amplitude *= np.sum(target_resampled) / np.sum(hologram_amplitude)
    hologram_amplitude = np.sqrt(hologram_amplitude)

    rmse_values = []

    for _ in tqdm(range(num_iterations), desc="BWC Lens Iterations", unit="iter"):
        hologram_field = fftshift(ifft2(ifftshift(field)))
        phase = np.angle(hologram_field)
        hologram_field = hologram_amplitude * np.exp(1j * phase)
        field = fftshift(fft2(fftshift(hologram_field)))

        intensity = norm(np.abs(field) ** 2)
        updated_amplitude = (SR * target_amplitude) + (np.sqrt(intensity) * NR)
        field = updated_amplitude * np.exp(1j * np.angle(field))

        if compute_rmse:
            rmse_values.append(np.sqrt(np.mean((intensity - target_resampled) ** 2)))

    center = n
    crop = n // 2
    phase_cropped = phase[center - crop:center + crop, center - crop:center + crop]

    return phase_cropped, rmse_values

def SLM_to_FourierPlane(phase):
    N, M = phase.shape

    # SLM field with zero padding
    # phase
    SLM_phase = np.pad(phase, pad_width=((N//2, N//2), (M//2, M//2)), mode='constant', constant_values=0)
    # amplitude
    SLM_ampl = np.zeros((2*N, 2*M))
    SLM_ampl[N//2:N//2 + N, M//2:M//2 + M] = 1
    # Slm field
    SLM_field = SLM_ampl * np.exp(1j*SLM_phase)
    
    #Fourier Field
    F_field = fftshift(fft2(ifftshift(SLM_field)))
    
    return F_field

#%%
# if __name__ == '__main__':
    
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
