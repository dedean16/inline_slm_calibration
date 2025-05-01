"""
Demonstrate the inline calibration method on synthetic data.
"""
# External (3rd party)
import torch
import numpy as np
import matplotlib.pyplot as plt

# Internal
from calibration_functions import learn_field, signal_model, fit_bleaching
from plot_utilities import plot_results_ground_truth
from directories import data_folder


# === Settings === #
# Paths/globs to measurement data files
inline_glob = data_folder.glob("inline/inline-slm-calibration_t*.npz")                  # Our inline method
ref_glob = data_folder.glob("tg_fringe/tg-fringe-slm-calibration-r*_noraw.npz")         # Reference TG fringe

plt.rcParams.update({'font.size': 14})
settings = {
    "do_live_plot": True,
    "do_weights_plot": True,
    "do_bleach_plot": True,
    "do_signal_fit_plot": True,
    "do_end_plot": True,
    "plot_per_its": 500,
    "nonlinearity": 2.0,
    "learning_rate": 0.3,
    "iterations": 10000,
    "cmap": 'viridis',
}

# Number of synthetic runs
num_runs = 3


# === Synthetic ground truth === #
# Gray values
n0 = 128
n_step = 8
gv0 = np.arange(0, n0)
gv1 = np.arange(0, n0, n_step)
numel = gv0.size * gv1.size

# Interference and Fluorescence
phase_gt = 3.8 * np.pi * torch.linspace(0.0, 1.0, n0) ** 2
amp_gt = torch.linspace(1.0, 1.0, n0)
E_gt = amp_gt * torch.exp(1j*phase_gt)

a_gt = 0.5
b_gt = 1.5
S_bg_gt = 1.0
nonlin_gt = 2.0

# Photobleaching
decay_gt = 5e-5
factor_gt = 1.0

# Noise
read_noise_level = 0.5
shot_noise_level = 0.5
n_samples = 100

# === Create synthetic measurement === #
# Note: The bleaching curve is not created with the same model as what is fit. For low bleaching, this is close enough.
m_gt_nobleach = signal_model(gray_values0=gv0, gray_values1=gv1, E=E_gt, a=a_gt, b=b_gt, S_bg=S_bg_gt,
                    nonlinearity=nonlin_gt, decay=0, factor=factor_gt,
                    signal_integral=torch.ones(numel))
_, _, signal_integral = fit_bleaching(gv0, gv1, m_gt_nobleach, weights=torch.ones_like(m_gt_nobleach))     # Approximate bleaching

m_gt = signal_model(gray_values0=gv0, gray_values1=gv1, E=E_gt, a=a_gt, b=b_gt, S_bg=S_bg_gt,
                    nonlinearity=nonlin_gt, decay=decay_gt, factor=factor_gt, signal_integral=signal_integral)


# Initialize arrays
inline_gray_all = [None] * num_runs
inline_phase_all = [None] * num_runs
inline_amp_all = [None] * num_runs
inline_amp_norm_all = [None] * num_runs
nonlin_all = [None] * num_runs

for r in range(num_runs):
    # Add noise
    shape = [*m_gt.shape, n_samples]
    m_synth_raw = m_gt.unsqueeze(-1) + read_noise_level * torch.randn(*shape) \
                  + shot_noise_level * torch.randn(*shape) * m_gt.unsqueeze(-1).sqrt()
    measurements_synth = m_synth_raw.mean(dim=-1)
    stds_synth = m_synth_raw.std(dim=-1)

    # Learn phase response
    nonlin, a, b, S_bg, phase, amplitude, amplitude_norm, losses = learn_field(
        gray_values0=gv0, gray_values1=gv1, measurements=measurements_synth, stds=stds_synth, **settings)

    print(f"a={a:.4f}, b={b:.4f}, S_bg={S_bg:.4f}, nonlin = {nonlin:.4f} ({settings['nonlinearity']})")

    # Store results in array
    inline_gray_all[r] = gv0
    inline_amp_all[r] = amplitude
    inline_amp_norm_all[r] = amplitude_norm
    inline_phase_all[r] = phase
    nonlin_all[r] = nonlin

# Summarize results with median and std
inline_gray = inline_gray_all[0]
inline_amplitude_norm = np.median(inline_amp_norm_all, axis=0)
inline_amplitude_norm_std = np.std(inline_amp_norm_all, axis=0)
inline_amplitude_norm_std_per_measurement = np.std(inline_amp_norm_all, axis=1)
inline_phase = np.median(inline_phase_all, axis=0)
inline_phase -= inline_phase.mean()
inline_phase_std = np.std(inline_phase_all, axis=0)

ref_phase = (phase_gt - phase_gt.mean()).numpy()

print(f'avg σ_φ={np.mean(inline_phase_std):.3g} rad')

if settings['do_end_plot']:
    plot_results_ground_truth(
        inline_gray, inline_phase, inline_phase_std, inline_amplitude_norm, inline_amplitude_norm_std,
        gv0, ref_phase, np.zeros_like(ref_phase), amp_gt, np.zeros_like(amp_gt), reflabel='Ground Truth')

