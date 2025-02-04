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


def plot_results_ground_truth(gray_values, phase, amplitude,
                              gray_values_ref, phase_ref, phase_ref_err, amplitude_ref, amplitude_ref_err):
    lightC0 = '#a8d3f0'

    # Plot calibration curves of phase and amplitude
    plt.figure(figsize=(9, 5))
    plt.subplots_adjust(left=0.1, right=0.95, hspace=0.35, wspace=0.35, top=0.92, bottom=0.12)

    plt.subplot(1, 2, 1)
    phase_ref -= phase_ref[0]
    phase_diff = np.interp(gray_values_ref, gray_values, phase) - phase_ref
    phase -= phase_diff.mean()

    plt.errorbar(gray_values_ref, phase_ref, yerr=phase_ref_err, color='C0', ecolor=lightC0, label='Reference')
    plt.plot(gray_values, phase, '+', color='C1', label='Inline (ours)')
    plt.xlabel('Gray value')
    plt.ylabel('Phase')
    plt.title('a. Phase response')
    plt.legend()

    plt.subplot(1, 2, 2)
    rel_amplitude_ref = amplitude_ref / amplitude_ref.mean()
    rel_amplitude_ref_err = amplitude_ref_err / amplitude_ref.mean()
    rel_amplitude = amplitude / amplitude.mean()

    plt.errorbar(gray_values_ref, rel_amplitude_ref, yerr=rel_amplitude_ref_err, color='C0', ecolor=lightC0, label='Reference')
    plt.plot(gray_values, rel_amplitude, '+', color='C1', label='Inline (ours)')
    plt.xlabel('Gray value')
    plt.ylabel('Normalized amplitude')
    plt.ylim((0, 1.1 * np.maximum(rel_amplitude.max(), rel_amplitude_ref.max())))
    plt.title('b. Normalized amplitude response')
    plt.legend()

    # Phase difference
    plt.figure()
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
    plt.plot(np.diff(phase_ref))
    plt.title('Phase response slope')
    plt.xlabel('Gray level')
    plt.ylabel('$d\\phi/dg$')

    # Plot field response in complex plane
    plt.figure()
    amplitude_norm = amplitude / amplitude.mean()
    E_norm = amplitude_norm * np.exp(1.0j * phase)
    amplitude_ref_norm = amplitude_ref / amplitude_ref.mean()
    E_ref_norm = amplitude_ref_norm * np.exp(1.0j * phase_ref)
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
