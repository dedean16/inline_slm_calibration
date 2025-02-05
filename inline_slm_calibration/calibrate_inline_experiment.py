"""
Calibration measurements for the SLM using our new inline method with non-linear fitting analysis.

Note: When newly running this script, make sure the defined file and folder paths are valid, and update if required.
Some folders from directories.py are used.
"""
# Built-in
import time
from pathlib import Path

# External (3rd party)
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from tqdm import tqdm
from zaber_motion import Units
from zaber_motion.ascii import Connection

# External (ours)
from openwfs.devices import ScanningMicroscope, Gain, SLM, Axis
from openwfs.devices.galvo_scanner import InputChannel
from openwfs.utilities import Transform

# Internal
from filters import DigitalNotchFilter
from experiment_helper_classes import RandomSLMShutter, OffsetRemover
from experiment_helper_functions import (autodelay_scanner, converge_parking_spot, park_beam, get_com_by_vid_pid,
                                         inline_calibrate)
from inline_slm_calibration.helper_functions import gitinfo
from inline_slm_calibration.directories import data_folder


# ========== Settings ========== #

# Save filepath and filename prefix
save_path = Path(data_folder)
filename_prefix = 'inline-slm-calibration_'

do_quick_test = False

if not do_quick_test:
    # Sample stage settings
    stage_settings = {
        'settle_time': 60 * u.s,
        'step_size': 150 * u.um,
        'num_steps_axis1': 3,
        'num_steps_axis2': 3,
    }

    # SLM gray values to calibrate
    gray_values1 = range(256)
    gray_values2 = range(0, 256, 16)

else:
    # Sample stage settings
    stage_settings = {
        'settle_time': 1 * u.s,
        'step_size': 100 * u.um,
        'num_steps_axis1': 2,
        'num_steps_axis2': 1,
    }

    # SLM gray values to calibrate
    gray_values1 = range(0, 256, 8)
    gray_values2 = range(0, 256, 64)

# PMT Amplifier
signal_gain = 0.6 * u.V

# SLM
slm_props = {
    'wavelength': 804 * u.nm,
    'f_obj': 12.5 * u.mm,
    'NA_obj': 0.8,
    'slm_to_pupil_magnification': 2,
    'pixel_pitch': 9.2 * u.um,
    'height_pix': 1152,
}

# SLM computed properties
slm_props['height'] = slm_props['pixel_pitch'] * slm_props['height_pix']
slm_props['pupil_radius_on_slm'] = slm_props['f_obj'] * slm_props['NA_obj'] / slm_props['slm_to_pupil_magnification']
slm_props['scaling_factor'] = 2 * slm_props['pupil_radius_on_slm'] / slm_props['height']

# Notch filter
notch_kwargs = {
    'frequency': 320 * u.kHz,
    'bandwidth': 15 * u.kHz,
    'dtype_out': 'int16',
}

# Laser scanner
scanner_props = {
    'sample_rate': 0.8 * u.MHz,
    'resolution': 1024,
    'zoom': 15,
    'initial_delay': 100.0 * u.us,
    'scale': 428 * u.um / u.V,
    'v_min': -1.0 * u.V,
    'v_max': 1.0 * u.V,
    'maximum_acceleration': 5.0e4 * u.V/u.s**2,
}

input_channel_kwargs = {
    'channel': 'Dev4/ai16',
    'v_min': -1.0 * u.V,
    'v_max': 1.0 * u.V,
}

park_kwargs = {
    'do_plot': False,                # For debugging
    'median_filter_size': (3, 3),
    'target_width': 32,
    'max_iterations': 15,
    'park_to_one_pixel': False,
}


# ====== Prepare hardware ====== #
print('Start hardware initialization...')

x_axis = Axis(channel='Dev4/ao3',
              v_min=scanner_props['v_min'],
              v_max=scanner_props['v_max'],
              maximum_acceleration=scanner_props['maximum_acceleration'],
              scale=scanner_props['scale'])

y_axis = Axis(channel='Dev4/ao2',
              v_min=scanner_props['v_min'],
              v_max=scanner_props['v_max'],
              maximum_acceleration=scanner_props['maximum_acceleration'],
              scale=scanner_props['scale'])

pmt_input_channel = InputChannel(**input_channel_kwargs)

# Define laser scanner, with offset
scanner_with_offset = ScanningMicroscope(
    bidirectional=True,
    sample_rate=scanner_props['sample_rate'],
    y_axis=y_axis,
    x_axis=x_axis,
    input=pmt_input_channel,
    delay=scanner_props['initial_delay'],
    preprocessor=DigitalNotchFilter(**notch_kwargs),
    reference_zoom=2.0,
    resolution=scanner_props['resolution'])  # Define notch filter to remove background ripple
scanner_with_offset.zoom = scanner_props['zoom']

# Define Processor that fetches data from scanner and removes offset
reader = OffsetRemover(source=scanner_with_offset, offset=2 ** 15, dtype_out='float64')

# SLM
slm_shape = (slm_props['height_pix'], slm_props['height_pix'])
slm_transform = Transform(((slm_props['scaling_factor'], 0.0), (0.0, slm_props['scaling_factor'])))
slm = SLM(2, transform=slm_transform)
shutter = RandomSLMShutter(slm)
print('Found SLM')

# Define NI-DAQ Gain channel and settings
gain_amp = Gain(
    port_ao="Dev4/ao0",
    port_ai="Dev4/ai0",
    port_do="Dev4/port0/line0",
    reset=False,
    gain=0.00 * u.V,
)
gain_amp.gain = signal_gain
print('Connected to Gain NI-DAQ')


# ====== Preparation measurements ====== #
print('Reading dark frame...')
dark_frame = reader.read()

shutter.open = False
input(f'Please unblock laser and press enter')

# Determine and update scanner delays
print('Determining bidirectional delay of scanner...')
scanner_props['delay'] = autodelay_scanner(shutter=shutter, image_reader=reader, scanner=reader.source)
print(f"Scanner delay: {scanner_props['delay']}")

# Zaber stage
comport = get_com_by_vid_pid(vid=0x2939, pid=0x495b)                # Get COM-port of Zaber X-MCB2


# ========= Main measurements ========= #
with (Connection.open_serial_port(comport) as connection):            # Open connection with Zaber stage
    # Zaber stage initialization
    connection.enable_alerts()
    device_list = connection.detect_devices()
    device = device_list[0]
    print(f"Connected to {device.name} (serial number: {device.serial_number}) at {comport}")
    axis1 = device.get_axis(1)
    axis2 = device.get_axis(2)
    axis1_start_position = axis1.get_position()
    axis2_start_position = axis2.get_position()

    # Repeat experiment on different locations. Move with Zaber stage.
    total_steps = stage_settings['num_steps_axis1'] * stage_settings['num_steps_axis2'] * len(gray_values1) * len(gray_values2)
    progress_bar = tqdm(colour='blue', total=total_steps, ncols=90)

    for a1 in range(stage_settings['num_steps_axis1']):             # Loop over stage axis 1
        for a2 in range(stage_settings['num_steps_axis2']):         # Loop over stage axis 2
            progress_bar_suffix = f'axes: {a1 + 1}/{stage_settings["num_steps_axis1"]},' \
                + f' {a2+1}/{stage_settings["num_steps_axis2"]}'

            print('\nStart converging to parking spot')
            park_location, park_imgs = converge_parking_spot(shutter=shutter, image_reader=reader,
                                                             scanner=reader.source, **park_kwargs)
            print(f'Beam parking spot at {park_location}')
            park_beam(scanner_with_offset, park_location)

            # Flat wavefront signal, before running the algorithm
            shutter.open = True
            slm.set_phases(0)

            frames = inline_calibrate(feedback=reader, slm=slm,
                                      gray_values1=gray_values1, gray_values2=gray_values2,
                                      progress_bar=progress_bar, progress_bar_suffix=progress_bar_suffix)

            shutter.open = False
            scanner_with_offset.reset_roi()

            # Save results
            # TODO: switch to HDF5 and/or json
            print('\nSave...')
            park_result = {
                'location': park_location,
                'imgs': park_imgs,
            }

            np.savez(
                save_path.joinpath(f'{filename_prefix}t{round(time.time())}'),
                gitinfo=[gitinfo()],
                time=time.time(),
                stage_settings=[stage_settings],
                signal_gain=[{'gain': signal_gain}],
                slm_props=[slm_props],
                notch_kwargs=[notch_kwargs],
                scanner_props=[scanner_props],
                input_channel_kwargs=[input_channel_kwargs],
                park_kwargs=[park_kwargs],
                park_result=[park_result],
                dark_frame=[dark_frame],
                gray_values1=[gray_values1],
                gray_values2=[gray_values2],
                frames=[frames],
            )
            # TODO: add calibration raw measurement

            progress_bar.update()

            print('\nMove stage')

            if a2+1 < stage_settings['num_steps_axis2']:
                # Move stage axis and let it settle
                axis2.move_relative(stage_settings['step_size'].to_value(u.um), Units.LENGTH_MICROMETRES)
                time.sleep(stage_settings['settle_time'].to_value(u.s))
            else:
                break                                   # Skip last stage move and sleep

        axis2.move_absolute(axis2_start_position)       # Return stage axis to starting position
        if a1+1 < stage_settings['num_steps_axis1']:
            # Move stage axis and let it settle
            axis1.move_relative(stage_settings['step_size'].to_value(u.um), Units.LENGTH_MICROMETRES)
            time.sleep(stage_settings['settle_time'].to_value(u.s))
        else:
            break                                       # Skip last sleep and stage movement

    axis1.move_absolute(axis1_start_position)           # Return stage axis to starting position


print('--- Done! ---')
input('Please block laser and press enter')
shutter.open = True

scanner_with_offset.close()
