Inline calibration of spatial light modulators in nonlinear microscopy
======================================================================

Python implementation of inline calibration of the field response of a spatial light modulator in a nonlinear
microscope. Read the pre-print article on https://arxiv.org/abs/2505.22482. Download the corresponding measurement data from https://data.4tu.nl/datasets/e06926cc-5f6e-4fc0-8170-16d71d5b1c1e.

Installation
------------
This repository uses Poetry for installing its dependencies.
Please see https://python-poetry.org/.

When Poetry is installed, dependencies can be installed by running poetry inside the repository's directory:
``poetry install``

Running the scripts
-------------------
The inline calibration measurements are performed by running: ``inline_slm_calibration/calibrate_inline_experiment.py``.
This script mostly defines the hardware settings and then performs the measurements in several locations in the sample
(beads are selected automatically). The inline measurements (phase stepping part A and B, and retrieving the feedback
signal) are then carried out by the function ``inline_calibrate`` in
``inline_slm_calibration/experiment_helper_functions.py``.

The measured data can be analyzed by running ``inline_slm_calibration/inline_calibrate_phase_response.py``,
which should plot the calibration curves. Plot settings and optimizer settings can be found near the top of the script.

Before running any script, please ensure the variables defined in ``inline_slm_calibration/directories.py``
point to valid directories. Additionally, for the scripts that import/export data, please ensure that (sub)directories
and file paths in the settings (near the top of each script) are valid. By default, the a live plot of the fitting
process is shown. For the best performance, it is recommended to set ``settings["do_plot"]`` to ``False``.

A demonstration on synthetic data can be run with ``inline_slm_calibration/inline_calibrate_phase_response_synth.py``.
