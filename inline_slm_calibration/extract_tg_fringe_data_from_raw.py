"""
The raw data for the Twymann-Green interference measurements is huge. This script reads and recreates the files without
the raw frames.
"""
import glob
import numpy as np
import os

# Define the glob pattern for the input files
pattern = r"C:\LocalData\tg_fringe\tg-fringe-slm-calibration-r*.npz"

# Get the list of files matching the pattern
file_list = glob.glob(pattern)

# Process each file in the list
for input_file in file_list:
    print(f"Processing {input_file}")

    # Load the npz file and extract 'field' and 'gray_values'
    with np.load(input_file) as data:
        field = data['field']
        gray_values = data['gray_values']

    # Create the output file name by appending '_noraw' before the extension
    base_name, ext = os.path.splitext(input_file)
    output_file = base_name + '_noraw.npz'

    # Save the extracted arrays to the new npz file
    np.savez(output_file, field=field, gray_values=gray_values)
    print(f"Saved {output_file}")
