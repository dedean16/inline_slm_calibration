"""
Utilities for importing calibrations.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch


def import_reference_calibrations(ref_glob, do_plot=False, do_remove_bias=False):
    """
    Import a reference calibration from npz files.

    Args:
        ref_glob: Glob that defines the paths to the npz files. Each file must contain the keys "gray_values" and
            "field".
        do_plot: Plot extracted calibration curves.
        do_remove_bias: Remove bias in the field by centering the minima and maxima of the real and imaginary parts.

    Returns:
        gray values, median phase, phase std, median amplitude, amplitude std. The std indicates the repeatability.
    """
    # Initialize
    ref_files = list(ref_glob)
    ref_gray_all = [None] * len(ref_files)
    ref_phase_all = [None] * len(ref_files)
    ref_amp_all = [None] * len(ref_files)

    # Check
    if len(ref_files) < 2:
        raise ValueError(f'The ref_glob yielded {len(ref_files)} files. At least 2 files are required.')

    # Extract from files
    for n_f, filepath in enumerate(ref_files):
        npz_data = np.load(filepath)
        ref_gray_all[n_f] = npz_data["gray_values"]
        ref_amp_all[n_f] = np.abs(npz_data["field"])
        ref_phase = np.unwrap(np.angle(npz_data["field"]))                      # Unwrap
        ref_phase -= ref_phase.mean()                                           # Use common offset
        ref_phase_all[n_f] = ref_phase * np.sign(ref_phase[-1] - ref_phase[0])  # Make dφ/dg mostly positive

    # Summarize as 1 curve with error bars
    # Note: take median as it is robust against outliers. Use std to represent repeatability
    ref_gray = ref_gray_all[0]
    ref_amplitude = np.median(ref_amp_all, axis=0)
    ref_amplitude_norm = ref_amplitude / ref_amplitude.mean()
    ref_amplitude_norm_std = np.std(ref_amp_all, axis=0)
    ref_phase = np.median(ref_phase_all, axis=0)
    ref_phase -= ref_phase.mean()
    ref_phase_std = np.std(ref_phase_all, axis=0)

    # Remove bias by centering extremes around zero
    Er = ref_amplitude * np.exp(1j * ref_phase)
    Er_bias = 0.5 * (Er.real.max() + Er.real.min()) + 0.5j * (Er.imag.max() + Er.imag.min())  # Center E around 0
    Er_bias_prcnt = 100 * abs(Er_bias) / np.mean(abs(Er))
    print(f'Bias in reference field = {Er_bias_prcnt:.1f}%')

    if do_remove_bias:
        Er_centered = Er - Er_bias
        ref_amplitude = abs(Er_centered)
        ref_phase = np.unwrap(np.angle(Er_centered))
        ref_phase -= ref_phase.mean()

    if do_plot:
        plt.figure()
        plt.plot(np.asarray(ref_phase_all).T)
        plt.xlabel('Gray value')
        plt.ylabel('Phase')
        plt.title('TG fringe calibration, unwrapped\ncompensated for sign and offset')
        plt.pause(0.01)

        plt.figure()
        plt.plot(np.asarray(ref_amp_all).T)
        plt.xlabel('Gray value')
        plt.ylabel('Amp')
        plt.title('TG fringe calibration, unwrapped\ncompensated for sign and offset')
        plt.show()

    return ref_gray, ref_phase, ref_phase_std, ref_amplitude_norm, ref_amplitude_norm_std


def import_inline_calibration(inline_file, do_plot=False):
    """
    Import an inline calibration.

    Args:
        inline_file: Path to npz file containing the keys:
            'frames': 4D array containing the raw signal measurements. First two dims may be used to store the frames.
            'gray_values1', 'gray_values2': correspond to gray values of group A and B.
            'dark_frame': contains frame taken with laser blocked.
        do_plot: Plot dark_frame noise statistics.

    Returns:
        gv0: Gray values for group A.
        gv1: Gray values for group B.
        measurements: 2D array containing the signal per gray value pair.
    """
    npz_data = np.load(inline_file)
    measurements = npz_data["frames"].mean(axis=(0, 1, 2)) - npz_data['dark_frame'].mean()
    stds = npz_data["frames"].std(axis=(0, 1, 2))
    gv0 = npz_data['gray_values1'][0]
    gv1 = npz_data['gray_values2'][0]

    # Fit noise model
    def noise_model(x, a, b, c):
        return a * x ** 2 + b * x + c

    alpha = torch.tensor(0.5, requires_grad=True)
    beta = torch.tensor(10.0, requires_grad=True)
    var = torch.tensor(npz_data['dark_frame'].std() ** 2, requires_grad=True)
    optimizer = torch.optim.Adam([alpha, beta, var], lr=0.1)
    measurements = torch.tensor(measurements, dtype=torch.float32)
    stds = torch.tensor(stds, dtype=torch.float32)
    for _ in range(1000):
        optimizer.zero_grad()
        loss = torch.mean((stds ** 2 - noise_model(measurements, alpha, beta, var)) ** 2)
        loss.backward()
        optimizer.step()

    alpha = alpha.detach()
    beta = beta.detach()
    var = var.detach()
    plt.loglog(measurements, stds ** 2, 'o')
    plt.loglog(measurements, noise_model(measurements, alpha, beta, var), 'o')
    weights = noise_model(measurements, 0, beta, var).pow(-0.5)
    weights = weights / weights.mean()

    plt.figure()
    plt.plot(measurements, weights)
    plt.show()
    plt.pause(0.01)

    if do_plot:
        plt.figure()
        plt.hist(npz_data['dark_frame'].flatten(), bins=range(-100, 100))
        plt.title('Dark frame noise distribution')
        plt.xlabel('Signal')
        plt.ylabel('Counts')
        plt.pause(0.01)

    return gv0, gv1, measurements, weights
