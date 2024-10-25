import numpy as np
import matplotlib.pyplot as plt
import h5py

from helper_functions import get_dict_from_hdf5
from calibration_functions import learn_field
from directories import data_folder
from online_slm_calibration.plot_utilities import plot_results_ground_truth

# === Settings === #
settings = {
    "do_plot": True,
    "plot_per_its": 30,
    "nonlinearity": 2,
    "learning_rate": 0.3,
    "iterations": 1800,
    "smooth_loss_factor": 0,
}

# Import feedback measurements and reference phase response
filepath_ref = data_folder.joinpath("slm_reference_phase_response.mat")  # Reference phase response
filepath_measurements = data_folder.joinpath("slm_calibration_signal_feedback.mat")
with h5py.File(filepath_measurements, "r") as f:
    feedback_dict = get_dict_from_hdf5(f)
    measurements = feedback_dict["feedback"]
    gv0 = feedback_dict["gv_row"].astype(int).ravel() % 256
    gv1 = feedback_dict["gv_col"].astype(int).ravel() % 256

with h5py.File(filepath_ref) as f:
    ref_dict = get_dict_from_hdf5(f)
    ref_gray = ref_dict["gray_values"][0]
    ref_phase = ref_dict["phase_mean"][0]
    ref_phase_err = ref_dict["phase_std"][0]

# Compensate for photo-bleaching
m = measurements.flatten(order="F")
trend = np.polynomial.Polynomial.fit(range(len(m)), m, 2)
m = m - trend(range(len(m)))
measurements = m.reshape(measurements.shape, order="F")

# feedback_meas[90,:] = 0.5 * (feedback_meas[89,:]+feedback_meas[91,:])
# feedback_meas[82,:] = 0.5 * (feedback_meas[81,:]+feedback_meas[83,:])
# extent = (gv1.min()-0.5, gv1.max()+0.5, gv0.min()-0.5, gv0.max()+0.5)
# plt.imshow(feedback_meas, extent=extent)
# plt.show()
#
ff = measurements[gv1, :]
plt.figure()
plt.imshow(ff)
plt.show()

plt.plot(m)
plt.show()


# Learn phase response
lr, nl, phase, amplitude = learn_field(
    gray_values0=gv0, gray_values1=gv1, measurements=measurements, **settings
)

print(f"lr = {lr} (1.0), nl = {nl} ({settings['nonlinearity']})")

plot_results_ground_truth(phase, amplitude, ref_phase)

plt.figure()
E = amplitude * np.exp(1.0j * phase)
E_ref = amplitude.mean() * np.exp(1.0j * ref_phase)
plt.plot(E.real, E.imag)
plt.plot(E_ref.real, E_ref.imag)
plt.show()

