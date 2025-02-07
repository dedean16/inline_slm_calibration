# External 3rd party
import torch
from torch import Tensor as tt
import numpy as np
from numpy import ndarray as nd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Internal
from plot_utilities import plot_field_response, plot_feedback_fit, plot_result_feedback_fit
from helper_functions import fit_quadratic


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
    """
    return a * x ** 2 + b * x + c


def compute_weights(measurements, stds, do_plot=False):
    """
    Compute weights

    Compute weights by fitting a noise model. Compute weights based on shot-noise and read-out noise. Technical noise
    is neglected.

    Args:
        measurements: Average signal per measurement
        stds: Standard deviation per measurement
        do_plot: Plot fitting results

    Returns:
        weights: Weights based on the noise analysis.
    """
    a_n, b_n, c_n = fit_quadratic(measurements, stds**2)
    w = 1 / noise_model(measurements, 0, b_n, c_n) ** 2
    weights = w / w.mean()

    if do_plot:
        # Plot noise model fit
        plt.loglog(measurements.flatten(), stds.flatten() ** 2, '+', color='tab:blue', label='Measurement')
        plt.loglog(measurements.flatten(), noise_model(measurements, a_n, b_n, c_n).flatten(), '.k',
                   label='Least Squares Fit')
        plt.xlabel('Mean signal')
        plt.ylabel('Signal variance')
        plt.title('Noise model')
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
        plt.show()

    return weights


def fit_bleaching(gray_value0, gray_value1, measurements: np.ndarray, weights, do_plot=False):
    """
    Fit photobleaching

    Args:
        TODO

    Returns:
        TODO
    """
    m = torch.tensor(measurements).t().contiguous().view(-1) # flatten("F")
    w = torch.tensor(weights).t().contiguous().view(-1)
    m = m / m.abs().mean()

    # locate elements for which gv0 == gv1. These are measured twice and should be equal except for noise and photobleaching.
    gv0 = np.asarray(gray_value0)
    sym_selection = [np.nonzero(gv0 == gv1)[0][0].item() for gv1 in gray_value1]

    learning_rate = 0.1

    # Initial values
    factor = torch.tensor(0.1 * (m.max() - m.min()), dtype=torch.float32, requires_grad=True)
    decay = torch.tensor(0.1 / len(m), dtype=torch.float32, requires_grad=True)
    received_energy = np.cumsum(torch.maximum(m,torch.tensor(0.0)))

    def take_diag(M):
        return M.reshape((measurements.shape[1], measurements.shape[0]))[:, sym_selection].diagonal()

    params = [
        {"params": [factor], "lr": learning_rate},
        {"params": [decay], "lr": 10*learning_rate / len(m)}
    ]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)

    if do_plot:
        plt.figure(figsize=(15, 5))

    for it in range(500):
        m_fit = photobleaching_model(factor, decay, received_energy)
        m_compensated = m / m_fit
        loss = ((take_diag(m) - take_diag(m_fit)) * weights).pow(2).mean()

        measurements_compensated = m_compensated.detach().numpy().reshape(measurements.shape, order='F')

        if it % 10 == 0 and do_plot:
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow(measurements_compensated, aspect='auto', interpolation='nearest')
            plt.title(f'Scale={factor.detach():.3f}, decay={decay.detach():.3g}')

            plt.subplot(1, 3, 2)
            plt.plot(take_diag(m).detach())
            plt.plot(take_diag(m_fit).detach())
            plt.ylim((0, 10))
            plt.title(f'Fit diagonal entries (gv0==gv1)')

            plt.subplot(1, 3, 3)
            plt.plot(take_diag(m_compensated).detach())
            plt.ylim((0, 10))
            plt.title(f'Compensated diagonal entries (gv0==gv1), {it}')
            plt.pause(0.01)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    ff = measurements_compensated[sym_selection, :]

    if do_plot:
        plt.figure()
        plt.imshow(ff)
        plt.title('Selected measurements\nwith gray values swapped')

        plt.figure()
        plt.plot(m)
        plt.title('Measurements in sequence')
        plt.ylabel('Signal')
        plt.xlabel('Index')
        plt.pause(0.01)

    return decay, factor, received_energy


def photobleaching_model(factor, decay, received_energy):
    """
    Photobleaching model

    The signal decays exponentially with the previously received energy: αᵢ=α₀⋅exp(-D⋅∑S).

    Args:
        factor: α₀
        decay: D
        received_energy: ∑S
    """
    return factor * torch.exp(-decay * received_energy)


def signal_model(gray_values0, gray_values1, E: tt, a: tt, b: tt, S_bg: tt, nonlinearity: tt, decay: tt,
                 factor, received_energy) -> tt:
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
    bleach_factor = photobleaching_model(factor, decay, received_energy).reshape((len(gray_values1), len(gray_values0))).T
    return I_excite.pow(nonlinearity) * bleach_factor + S_bg


def learn_field(
        gray_values0: np.array,
        gray_values1: np.array,
        measurements: np.array,
        stds: np.array,
        nonlinearity: float = 1.0,
        iterations: int = 50,
        do_plot: bool = False,
        do_weights_plot: bool = False,
        do_end_plot: bool = False,
        plot_per_its: int = 10,
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
        do_plot: If True, plot during learning.
        do_weights_plot: If True, plot noise analysis results.
        do_end_plot: If True, plot after learning.
        plot_per_its: Plot per this many learning iterations.
        learning_rate: Learning rate of the optimizer.

    Returns:
       nonlinearity, a, b, S_bg, phase, amplitude, normalized amplitude
    """
    # Initialize
    measurements = torch.tensor(measurements, dtype=torch.float32)
    measurements = measurements / measurements.std()                                # Normalize by std

    weights_np = compute_weights(measurements, stds, do_weights_plot)
    weights = torch.tensor(weights_np, dtype=torch.float32)

    # Fit signal decay due to photobleaching
    decay, factor, received_energy = fit_bleaching(gray_values0, gray_values1, measurements, weights, do_plot)

    # Initial guess
    E = torch.exp(2j * np.pi * torch.rand(256))                                     # Field response
    E.requires_grad_(True)
    a = torch.tensor(1.0, requires_grad=True, dtype=torch.complex64)           # Group A complex pre-factor
    b = torch.tensor(1.0, requires_grad=True, dtype=torch.complex64)           # Group B complex pre-factor
    S_bg = torch.tensor(0.0, requires_grad=True)
    nonlinearity = torch.tensor(nonlinearity, dtype=torch.float32, requires_grad=True)

    # Initialize parameters and optimizer
    params = [
        {"lr": learning_rate, "params": [E, a, b, S_bg]},
        {"lr": learning_rate * 0.1, "params": [nonlinearity]},
    ]
    optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)
    progress_bar = tqdm(total=iterations)

    # Gradient descent loop
    for it in range(iterations):
        predicted_signal = signal_model(
            gray_values0, gray_values1, E, a, b, S_bg, nonlinearity, decay, factor, received_energy)
        loss = ((measurements - predicted_signal) * weights).pow(2).mean()

        # Gradient descent step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Plot
        if do_plot and (it % plot_per_its == 0 or it == 0 or it == iterations - 1):
            if it == 0:
                plt.figure(figsize=(13, 4))
            else:
                plt.clf()
            plt.subplot(1, 3, 1)
            plot_field_response(E)
            plot_feedback_fit(measurements, predicted_signal, gray_values0, gray_values1)
            plt.title(f"feedback loss: {loss:.3g}\na: {a:.3g}, b: {b:.3g}, S_bg: {S_bg:.3g}")
            plt.pause(0.01)

        progress_bar.update()

    # Post-process
    amplitude = E.detach().abs()
    amplitude_norm = amplitude / amplitude.mean()
    phase = np.unwrap(np.angle(E.detach()))
    phase *= np.sign(phase[-1] - phase[0])
    phase -= phase.mean()

    if do_plot and do_end_plot:
        plt.figure(figsize=(14, 4.3))
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15)
        plot_result_feedback_fit(measurements, predicted_signal, gray_values0, gray_values1, weights)

    return nonlinearity.item(), a.item(), b.item(), S_bg.item(), phase, amplitude, amplitude_norm

