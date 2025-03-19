# External 3rd party
import torch
from torch import Tensor as tt
import numpy as np
from numpy import ndarray as nd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Internal
from plot_utilities import plot_field_response, plot_feedback_fit, plot_result_feedback_fit, colormap_adjust_piecewise
from helper_functions import fit_quadratic, ensure_tensor


# Fit noise model
def noise_model(x, a, b, c):
    """
    The noise model is used to determine the influence of different noise sources.
    variance = ax² + bx + c

    Args:
        x: Measured signal
        a: Image variation and technical noise
        b: Shot noise σ²∝⟨x⟩
        c: Read-out noise

    Returns:
        Estimation of the variance
    """
    return a * x ** 2 + b * x + c


def compute_weights(measurements, stds, do_weight_plot=False):
    """
    Compute weights

    Compute weights by fitting a noise model. Compute weights based on shot-noise and read-out noise, such that
    w = 1 / σ². Technical noise is neglected.

    Args:
        measurements: Average signal per measurement
        stds: Standard deviation per measurement
        do_weight_plot: Plot fitting results

    Returns:
        weights: Weights based on the variance determined with noise analysis.
    """
    a_n, b_n, c_n = fit_quadratic(measurements, stds**2)
    weights = 1 / noise_model(measurements, 0, b_n, c_n)

    if do_weight_plot:
        m_sorted = measurements.flatten().sort().values
        fit_img_var = noise_model(m_sorted, a_n, b_n, c_n)
        fit_noise_var = noise_model(m_sorted, 0, b_n, c_n)

        # Plot noise model fit - log log axes
        plt.figure()
        plt.subplots_adjust(bottom=0.12)
        plt.loglog(measurements.flatten(), stds.flatten() ** 2, '+', color='C0', label='Measurement', markersize=10)
        plt.loglog(m_sorted, fit_img_var, '--k', label='Fit')
        plt.loglog(m_sorted, fit_noise_var, ':r', label='Noise')
        plt.xlabel('Mean of image')
        plt.ylabel('Variance of image')
        plt.title('Noise analysis')
        plt.legend()

        # Plot noise model fit - linear axes
        plt.figure()
        plt.subplots_adjust(bottom=0.12)
        plt.plot(measurements.flatten(), stds.flatten() ** 2, '+', color='C0', label='Measurement')
        plt.plot(m_sorted, fit_img_var, '--k', label='Fit')
        plt.plot(m_sorted, fit_noise_var, ':r', label='Noise')
        plt.xlabel('Mean of image')
        plt.ylabel('Variance of image')
        plt.title('Noise analysis')
        plt.legend()

        # Plot weights vs mean signal
        plt.figure()
        plt.plot(measurements, weights, '.k')
        plt.xlabel('Mean signal')
        plt.ylabel('Weights')
        plt.title('Weights')

        # Plot weights vs gray values
        plt.figure()
        extent = (0, 1, 1, 0)
        plt.imshow(weights, extent=extent, interpolation='nearest')
        plt.title('Weights')

        print(f'\nClose plots to continue... or turn off weight plots (set do_weight_plot to False)')
        plt.show()

    return weights


def fit_bleaching(gray_value0, gray_value1, measurements: tt, weights: tt | float, nonlinearity: tt | float,
                  do_bleach_plot=False, iterations=2000):
    """
    Fit photobleaching

    Args:
        TODO

    Returns:
        TODO
    """
    S_m = measurements.clone().t().contiguous().view(-1) # flatten("F")
    w = ensure_tensor(weights).view(-1)

    # Negative signal is impossible and will cause error in gradient descent (in compute_S_beta)
    S_m[(S_m / S_m[0]) < 0] = 0

    # locate elements for which gv0 == gv1. These are measured twice and should be equal except for noise and photobleaching.
    gv0 = torch.tensor(gray_value0).view(1, -1)
    gv1 = torch.tensor(gray_value1).view(-1, 1)
    i_all = torch.arange(0, S_m.numel())
    mask = (gv0 == gv1).view(-1)  # Mask of maxima due to constructive interference: where g_A == g_B

    learning_rate = 0.001

    def compute_S_beta(S_m, S_0, beta_N_ratio):
        """S_β(t) = ∫ (S(t) / S₀)^(β/N) dt"""
        return ((S_m / S_0) ** beta_N_ratio).cumsum(dim=0) / S_m.numel()

    # Initial values
    beta_N_ratio = torch.tensor(1.0, requires_grad=True)
    S_0 = S_m[mask][0].clone().requires_grad_(True)
    alpha = torch.tensor(1.0, requires_grad=True)

    # First estimate of bleaching rate, by finding exponential through first and last point
    S_beta_last_init = compute_S_beta(S_m, S_0.detach(), beta_N_ratio.detach())[mask][-1]
    rate_init = (-S_beta_last_init / torch.log(S_m[mask][-1] / S_0.detach()))
    lamd = rate_init.clone().requires_grad_(True)  # Overall bleaching rate lambda

    params = [
        {"params": [S_0, alpha, lamd], "lr": learning_rate},
        {"params": [beta_N_ratio], "lr": 10*learning_rate},
    ]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)

    if do_bleach_plot:
        plt.figure(figsize=(15, 5))

    torch.autograd.set_detect_anomaly(True)

    for it in range(iterations):
        S_beta = compute_S_beta(S_m, S_0, beta_N_ratio)
        S_beta_atmax = S_beta[mask]
        efficiency_fit_atmax = photobleaching_model(S_0, S_beta_atmax, lamd, beta_N_ratio, alpha)
        loss_pb = (w[mask] * ((efficiency_fit_atmax - S_m[mask])**2)).mean()

        if ((it % 2 == 0 and it < 80) or (it % 25 == 0 and it >=80) or it == iterations-1) and do_bleach_plot:
            efficiency_fit = photobleaching_model(S_0, S_beta, lamd, beta_N_ratio, alpha)
            plt.clf()
            plt.plot(i_all, S_m, label='All measurements')
            plt.plot(i_all[mask], S_m[mask], 'or', label='$g_A=g_B$')
            plt.plot(i_all, efficiency_fit.detach(), '--k', label='Bleach fit')
            plt.title(f'Photobleaching, it: {it}\n$\\alpha$={alpha:.4g}, $\\beta/N$={beta_N_ratio:.4g}, $\\lambda$={lamd:.3g}')
            plt.ylabel('Signal')
            plt.xlabel('Time (measurement index)')
            plt.legend()

            plt.pause(0.05)

        loss_pb.backward()
        optimizer.step()
        optimizer.zero_grad()

    efficiency_fit = photobleaching_model(S_0, S_beta, lamd, beta_N_ratio, alpha).detach()

    if do_bleach_plot:
        plt.figure(figsize=(6, 4))
        plt.subplots_adjust(bottom=0.15)
        plt.plot(i_all, S_m, label='All measurements')
        plt.plot(i_all[mask], S_m[mask], 'or', label='$g_A=g_B$')
        plt.plot(i_all, efficiency_fit, '--k', label='Bleach fit')
        plt.title(f'Photobleaching')
        plt.ylabel('Signal')
        plt.xlabel('Time (measurement index)')
        plt.legend()
        plt.show()

    # efficiency = torch.tensor(efficiency_fit.numpy().reshape(measurements.shape, order='F'))
    sh = measurements.shape
    efficiency = efficiency_fit.reshape((sh[1], sh[0])).t().contiguous()

    if do_bleach_plot:
        plt.figure()
        extent = (gray_value1.min()-0.5, gray_value1.max()+0.5, gray_value0.max()+0.5, gray_value0.min()-0.5)
        cmap = colormap_adjust_piecewise('viridis')
        plt.imshow(efficiency, extent=extent, interpolation='nearest', cmap=cmap)
        plt.title('Efficiency due to Photobleaching')
        plt.colorbar()
        plt.show()

    return efficiency


def photobleaching_model(S_0: tt, S_beta: tt, lamd: tt, beta_N_ratio: tt, alpha: tt) -> tt:
    """
    Photobleaching model

    The signal decays exponentially with the previously received energy:

    We define I(t) such that (I(t))^N == 1 when g_A(t) == g_B(t).

    Args:

    Returns:
        The estimated signal efficiency η(t)
    """
    N_beta_ratio = 1 / beta_N_ratio
    gamma = alpha * beta_N_ratio
    # taylor_factor = 1 / (1 + (gamma - 1) * (N_beta_ratio * S_beta * (2*lamd + S_beta)) / (2*lamd**2))
    taylor_factor = 1.0
    return S_0 * torch.exp(-N_beta_ratio * S_beta / lamd) / taylor_factor



def signal_model(gray_values0, gray_values1, E: tt, a: tt, b: tt, S_bg: tt, nonlinearity: tt, efficiency) -> tt:
    """
    Compute feedback signal from two interfering fields.

    signal = αᵢ⋅|a⋅E(g_A) + b⋅E(g_B)|^(2N) + S_bg

    The illuminated pixel groups produce the corresponding fields a⋅E(g_A) and b⋅E(g_B) at the detector,
    where g_A, g_B are the gray values displayed on group A and B respectively, E(g) is the complex field response
    to the gray value g, and a, b are gray-value-invariant complex factors to account for the total intensity per group
    and the corresponding paths through the optical setup. S_bg denotes background signal and N is the nonlinear order.

    Args:
        gray_values0: Contains gray values of group A, corresponding to dim 0 of feedback.
        gray_values1: Contains gray values of group B, corresponding to dim 1 of feedback.
        E: Contains field response per gray value.
        a: Complex pre-factor for group A field.
        b: Complex pre-factor for group B field.
        S_bg: Background signal.
        nonlinearity: Nonlinearity coefficient.
    """
    E0 = E[gray_values0].view(-1, 1)
    E1 = E[gray_values1].view(1, -1)
    I_excite = (a * E0 + b * E1).abs().pow(2)
    return I_excite.pow(nonlinearity) * efficiency + S_bg


def learn_field(
        gray_values0: np.array,
        gray_values1: np.array,
        measurements: np.array,
        stds: np.array,
        nonlinearity: float = 1.0,
        iterations: int = 50,
        do_live_plot: bool = False,
        do_weights_plot: bool = False,
        do_bleach_plot: bool = False,
        do_signal_fit_plot: bool = False,
        do_end_plot: bool = False,
        do_import_plot: bool = False,
        plot_per_its: int = 10,
        cmap=colormap_adjust_piecewise('viridis'),
        learning_rate: float = 0.1,
) -> tuple[float, float, float, float, nd, nd, nd]:
    """
    Learn the field response from dual gray value measurements.

    Signal is normalized by std to make parameters (such as optimizer step size) independent of raw signal magnitude.
    For model details, please see the signal_model function docstring.

    Args:
        gray_values0: Contains gray values of group A, corresponding to dim 0 of feedback.
        gray_values1: Contains gray values of group B, corresponding to dim 1 of feedback.
        measurements: Average signal measurements as 2D array. Indices correspond to gray_values0 and gray_values1.
        stds: Standard deviations per signal measurement.
        nonlinearity: Expected nonlinearity coefficient. 1 = linear, 2 = 2PEF, 3 = 3PEF, etc., 0 = detector is broken :)
        iterations: Number of learning iterations.
        do_live_plot: If True, plot during learning.
        do_weights_plot: If True, plot noise analysis results.
        do_bleach_plot: If True, plot photobleaching analysis results.
        do_signal_fit_plot: If True, plot fit of the signal model.
        do_end_plot: Unused. Accepted for convenient dictionary argument.
        do_import_plot: Unused. Accepted for convenient dictionary argument.
        plot_per_its: Plot per this many learning iterations.
        cmap: Colormap for plotting signal measurements and model. Non-linear colormap may be useful for highly bleached
            measurements.
        learning_rate: Learning rate of the optimizer.

    Returns:
       nonlinearity, a, b, S_bg, phase, amplitude, normalized amplitude
    """
    # Initialize
    measurements = torch.tensor(measurements, dtype=torch.float32)
    measurements = measurements / measurements.std()                                # Normalize by std

    weights = compute_weights(measurements, stds, do_weights_plot)

    # Fit signal decay due to photobleaching
    print('Start learning photobleaching...')
    efficiency = fit_bleaching(gray_values0, gray_values1, measurements, weights, nonlinearity, do_bleach_plot)

    # Initial guess
    E = torch.exp(2j * np.pi * torch.rand(gray_values0.size))                       # Field response
    E.requires_grad_(True)
    a = torch.tensor(1.0, requires_grad=True, dtype=torch.complex64)           # Group A complex pre-factor
    b = torch.tensor(1.0, requires_grad=True, dtype=torch.complex64)           # Group B complex pre-factor
    S_bg = torch.tensor(0.0, requires_grad=True)
    N = torch.tensor(nonlinearity, dtype=torch.float32, requires_grad=True)

    # Initialize parameters and optimizer
    print('\nStart learning field...')
    params = [
        {"lr": learning_rate, "params": [E, a, b, S_bg]},
        {"lr": learning_rate * 0.1, "params": [N]},
    ]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)
    progress_bar = tqdm(total=iterations)

    # Gradient descent loop
    for it in range(iterations):
        predicted_signal = signal_model(
            gray_values0, gray_values1, E, a, b, S_bg, N, efficiency)
        loss = (weights * (measurements - predicted_signal).pow(2)).mean()

        # Gradient descent step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Plot
        if do_live_plot and (it % plot_per_its == 0 or it == 0 or it == iterations - 1):
            if it == 0:
                plt.figure(figsize=(13, 4))
            else:
                plt.clf()
            plt.subplot(1, 3, 1)
            plot_field_response(E)
            plot_feedback_fit(measurements, predicted_signal, gray_values0, gray_values1, cmap=cmap)
            plt.title(f"feedback loss: {loss:.3g}\na: {a:.3g}, b: {b:.3g}, S_bg: {S_bg:.3g}")
            plt.pause(0.01)

        progress_bar.update()

    # Post-process
    amplitude = E.detach().abs()
    amplitude_norm = amplitude / amplitude.mean()
    phase = np.unwrap(np.angle(E.detach()))
    phase *= np.sign(phase[-1] - phase[0])
    phase -= phase.mean()

    if do_signal_fit_plot:
        plt.figure(figsize=(17, 4.3))
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15)
        plot_result_feedback_fit(measurements, predicted_signal, gray_values0, gray_values1, weights, cmap=cmap)
        plt.pause(0.1)

    if do_live_plot:
        print(f'\nClose plots to continue... or turn off live plots (set do_live_plot to False)')
        plt.show()

    return N.item(), a.item(), b.item(), S_bg.item(), phase, amplitude, amplitude_norm

