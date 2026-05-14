import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Algorithms_2D import GS_resampled, AWGS_resampled, BWC_random_nz, norm, BWC_random, BWC_lens
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
from tqdm import tqdm

#%% === PARAMETERS ===
F = 50
N_holograms = 100
Q = 400 / 1200
MODE = "compute"  # Options: "compute" or "load"
MODE = "load"  # Options: "compute" or "load"

output_folder = "Algorithm_comparison_data"
os.makedirs(output_folder, exist_ok=True)

target_path = r"C:\Users\astam\Desktop\Repositories\holography_astarita\Cambridge\Test_imgs\rectangles_gradient_4x4_400x400_centered.png"
box_size = 100

#%% === LOAD TARGET ===
target= cv2.imread(target_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
target_resampled = target.repeat(2, axis=0).repeat(2, axis=1)
target_resampled = norm(target_resampled)

#%% === SELECT ROI ON TARGET ONLY ONCE ===
print("Select a 4x4 region of interest on the resampled TARGET. Press Enter to confirm.")
selection = {}

def onselect(eclick, erelease):
    x0, y0 = int(eclick.xdata), int(eclick.ydata)
    x1, y1 = int(erelease.xdata), int(erelease.ydata)
    selection['x0'], selection['x1'] = sorted([x0, x1])
    selection['y0'], selection['y1'] = sorted([y0, y1])

def onkeypress(event):
    if event.key == 'enter':
        plt.close()

fig, ax = plt.subplots()
ax.imshow(target_resampled, cmap='gray')
ax.set_title("Select a 4x4 block region on the resampled TARGET and press Enter")
selector = RectangleSelector(ax, onselect, useblit=True, interactive=True,
                             spancoords='pixels', button=[1], drag_from_anywhere=True)
fig.canvas.mpl_connect('key_press_event', onkeypress)
plt.show(block=True)

#%% === COMPUTE 16 ROI COORDINATES ===
x0, x1 = selection['x0'], selection['x1']
y0, y1 = selection['y0'], selection['y1']
coords = []
for j in range(4):
    y = y0 + j * (y1 - y0) / 3
    for i in range(4):
        x = x0 + i * (x1 - x0) / 3
        coords.append((int(x), int(y)))

#%% === FUNCTION TO COMPUTE OR LOAD AVERAGE RECONSTRUCTION ===
def generate_average_intensity(algorithm_func, label):
    save_dir = os.path.join(output_folder, label)
    os.makedirs(save_dir, exist_ok=True)

    if MODE == "load":
        print(f"Loading saved data for {label}...")
        files = sorted([f for f in os.listdir(save_dir) if f.endswith(".npy")])
        sum_intensity = None
        for f in files:
            I = np.load(os.path.join(save_dir, f))
            if sum_intensity is None:
                sum_intensity = I
            else:
                sum_intensity += I
        return sum_intensity / len(files)

    else:
        print(f"Computing and saving data for {label}...")
        sum_intensity = None
        for i in tqdm(range(N_holograms), desc=f"{label} - Generating"):
            if "BWC" in label:
                kino, I, _ = algorithm_func(target, Q=Q, F=F)
            else:
                kino, I, _ = algorithm_func(target, F=F)

            np.save(os.path.join(save_dir, f"{i:04d}_hologram.npy"), I)

            if sum_intensity is None:
                sum_intensity = I
            else:
                sum_intensity += I
        return sum_intensity / N_holograms

#%% === COMPUTE OR LOAD AVERAGED RECONSTRUCTIONS ===
I_avg_GS = generate_average_intensity(GS_resampled, "GS")
I_avg_AWGS = generate_average_intensity(AWGS_resampled, "AWGS")
I_avg_BWC = generate_average_intensity(BWC_random, "BWC_random")
I_avg_BWC_nz = generate_average_intensity(BWC_random_nz, "BWC_random_nz")
I_avg_BWC_lens = generate_average_intensity(BWC_lens, "BWC_lens")

#%% === ANALYSIS FUNCTION ===
def full_analysis(image, coords, label):
    H, W = image.shape
    means = []
    contrasts = []
    cmap = plt.colormaps.get_cmap('plasma')
    colors = [cmap(i / 15) for i in range(16)]
    x_vals = np.arange(1, 17)

    # === COMPUTE MEAN AND CONTRAST PER ROI ===
    for (x, y) in coords:
        x1_roi = max(0, x - box_size // 2)
        x2_roi = min(W, x + box_size // 2)
        y1_roi = max(0, y - box_size // 2)
        y2_roi = min(H, y + box_size // 2)
        roi = image[y1_roi:y2_roi, x1_roi:x2_roi]
        mean_val = np.mean(roi)
        std_val = np.std(roi)
        contrast = std_val / (mean_val + 1e-8)
        means.append(mean_val)
        contrasts.append(contrast)

    means = np.array(means)
    contrasts = np.array(contrasts)

    coeffs_lin = np.polyfit(x_vals, means, 1)
    fit_lin = np.polyval(coeffs_lin, x_vals)

    def compute_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred)**2)
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        return mse, r2

    mse_lin, r2_lin = compute_metrics(means, fit_lin)
    print(f"[{label}] MSE = {mse_lin:.6f}, R² = {r2_lin:.4f}")
    contrast_avg = np.mean(contrasts)

    # === COMBINED PLOT ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].imshow(image, cmap='inferno')
    axs[0].set_title(f"{label} – ROI Selection")
    for i, (x, y) in enumerate(coords):
        x1_roi, y1_roi = x - box_size // 2, y - box_size // 2
        rect = patches.Rectangle((x1_roi, y1_roi), box_size, box_size,
                                 linewidth=2, edgecolor=colors[i], facecolor='none')
        axs[0].add_patch(rect)
        axs[0].text(x1_roi + 3, y1_roi + 12, f'{i+1}', color=colors[i], fontsize=9, weight='bold')
    axs[0].axis('off')

    axs[1].scatter(x_vals, means, c=colors, s=80, edgecolors='k', label=f'{label}')
    axs[1].plot(x_vals, fit_lin, 'r--', label=f'Linear Fit (R²={r2_lin:.3f})')
    axs[1].set_xlabel("ROI Number")
    axs[1].set_ylabel("Average Value")
    axs[1].set_title("Linearity")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].scatter(x_vals, contrasts, c=colors, s=80, edgecolors='k', label='ROI Contrast')
    axs[2].axhline(contrast_avg, color='gray', linestyle='--', label=f'Avg Contrast: {contrast_avg:.3f}')
    axs[2].set_xlabel("ROI Number")
    axs[2].set_ylabel("Contrast (σ / μ)")
    axs[2].set_title("Contrast")
    axs[2].legend()
    axs[2].grid(True)

    plt.suptitle(label, fontsize=14)
    plt.tight_layout()
    plt.show()

#%% === PERFORM ANALYSIS ===
full_analysis(target_resampled, coords, "TARGET")
full_analysis(I_avg_GS, coords, "GS")
full_analysis(I_avg_AWGS, coords, "AWGS")
full_analysis(I_avg_BWC, coords, "BWC")
full_analysis(I_avg_BWC_nz, coords, "BWC nz")
full_analysis(I_avg_BWC_lens, coords, "BWC lens")

