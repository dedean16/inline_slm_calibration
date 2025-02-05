import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def colormap_adjust_piecewise(cmap_name: str, x_points=None, y_points=None):
    if x_points is None:
        x_points = (0.0, 0.1, 0.2, 0.4, 1.0)
    if y_points is None:
        y_points = (0.0, 0.5, 0.75, 0.85, 1.0)

    x = np.linspace(0.0, 1.0, 256)
    y = np.interp(x, x_points, y_points)

    # Get the colors from the colormap using the transformed values
    cmap = plt.cm.get_cmap(cmap_name)
    new_colors = cmap(y)

    # Create a new colormap with the transformed colors
    return ListedColormap(new_colors, name=f'{cmap_name}_piecewise{x_points},{y_points}')


def plot_results_ground_truth(gray_values, phase, phase_std, amplitude_norm, amplitude_norm_std,
                              ref_gray_values, ref_phase, ref_phase_std, ref_amplitude_norm, ref_amplitude_norm_std):
    lightC0 = '#a8d3f0'
    lightC1 = '#ffc899'

    # Plot calibration curves of phase and amplitude
    plt.figure(figsize=(9, 5))
    plt.subplots_adjust(left=0.1, right=0.95, hspace=0.35, wspace=0.35, top=0.92, bottom=0.12)

    plt.subplot(1, 2, 1)
    plt.errorbar(ref_gray_values, ref_phase, yerr=ref_phase_std, linestyle='dashed', color='C0', ecolor=lightC0, label='Reference')
    plt.errorbar(gray_values, phase, yerr=phase_std, color='C1', ecolor=lightC1, label='Inline (ours)')
    plt.xlabel('Gray value')
    plt.ylabel('Phase (rad)')
    plt.title('a. Phase response')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.errorbar(ref_gray_values, ref_amplitude_norm, yerr=ref_amplitude_norm_std, linestyle='dashed', fmt='', ecolor=lightC0, label='Reference')
    plt.plot(ref_gray_values, ref_amplitude_norm, linestyle='dashed', color='C0', zorder=3)
    plt.errorbar(gray_values, amplitude_norm, yerr=amplitude_norm_std, color='C1', ecolor=lightC1, label='Inline (ours)')
    plt.xlabel('Gray value')
    plt.ylabel('Normalized amplitude')
    plt.ylim((0, 1.1 * np.maximum(amplitude_norm.max(), ref_amplitude_norm.max())))
    plt.title('b. Normalized amplitude response')
    plt.legend()

    # Phase difference
    plt.figure()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
    plt.plot(np.diff(ref_phase))
    plt.title('Phase response slope')
    plt.xlabel('Gray level')
    plt.ylabel('$d\\phi/dg$')

    # Plot field response in complex plane
    plt.figure()
    amplitude_norm = amplitude_norm / amplitude_norm.mean()
    E_norm = amplitude_norm * np.exp(1.0j * phase)
    ref_amplitude_norm = ref_amplitude_norm / ref_amplitude_norm.mean()
    E_ref_norm = ref_amplitude_norm * np.exp(1.0j * ref_phase)
    plt.plot(E_ref_norm.real, E_ref_norm.imag, label="Reference")
    plt.plot(E_norm.real, E_norm.imag, label="Our method")
    plt.legend()

    plt.show()


def plot_field_response(field_response_per_gv):
    plt.plot(field_response_per_gv.detach().abs(), label='$A$')
    plt.plot(field_response_per_gv.detach().angle(), label='$\\phi$')
    plt.xlabel("Gray value")
    plt.ylabel("Phase (rad) | Relative amplitude")
    plt.legend()
    plt.title("Predicted field response")


def plot_feedback_fit(feedback_measurements, feedback, gray_values0, gray_values1):
    plt.subplot(1, 3, 2)
    extent = (gray_values1.min()-0.5, gray_values1.max()+0.5, gray_values0.max()+0.5, gray_values0.min()-0.5)
    vmin = torch.minimum(feedback_measurements.min(), feedback.min())
    vmax = torch.maximum(feedback_measurements.max(), feedback.max())
    cmap = colormap_adjust_piecewise('viridis')
    plt.imshow(feedback_measurements.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title("Measured feedback")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(feedback.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title("Predicted feedback")
    plt.colorbar()


def plot_result_feedback_fit(feedback_measurements, feedback, gray_values0, gray_values1, weights):
    extent = (gray_values1.min()-0.5, gray_values1.max()+0.5, gray_values0.max()+0.5, gray_values0.min()-0.5)
    vmin = torch.minimum(feedback_measurements.min(), feedback.min())
    vmax = torch.maximum(feedback_measurements.max(), feedback.max())

    plt.subplot(1, 3, 1)
    cmap = colormap_adjust_piecewise('viridis')
    plt.imshow(feedback_measurements.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title("a. Measured signal")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(feedback.detach(), extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title("b. Fit signal")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()

    weighted_residual = ((feedback_measurements- feedback) * weights).detach()
    plt.subplot(1, 3, 3)
    plt.imshow(weighted_residual, extent=extent, interpolation="nearest", cmap='magma')
    plt.title("c. Weighted residual")
    plt.xlabel('$g_B$')
    plt.ylabel('$g_A$')
    plt.colorbar()
