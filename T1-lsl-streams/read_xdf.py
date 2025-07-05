# %%
'''
Converts XDF files to Raw objects. Returns the Raw object and event-related
objects for Epochs generation.

Parameters:
    file : string
        Name of the XDF file
    device : string
        Name of the recording device
    with-stim : Boolean
        Option to add a STIM channel to the Raw object.

Returns:
    raw : mne.Raw of shape (channels, samples)
        Continous EEG data.
    event_arr : np.array of shape (num_events, 3)
        Array to be used for Epochs generation
        First column: Event onset indices
        Second column: Zeros
        Third column: Event IDs.
    event_id : dict
        Label encoding of the events
        keys --> Names of the cues
        values --> Cue (event) IDs.

Author:
    Karahan Yilmazer

Email:
    yilmazerkarahan@gmail.com
'''

import re
from datetime import datetime, timezone

import mne
import numpy as np
import pyxdf
from sklearn.preprocessing import LabelEncoder

# %%
# XDF file name
file = ''
# Device name ('uhb' or 'ac')
device = 'uhb'
# Whether markers should be appended as a new channel to the Raw object
with_stim = False

# Read the XDF file
streams, header = pyxdf.load_xdf(file)

# Initialize lists for the streams
marker_streams = []
data_stream = []

for stream in streams:
    # Get the relevant values from the stream
    stream_name = stream['info']['name'][0].lower()
    stream_type = stream['info']['type'][0].lower()

    # Assign streams to their corresponding lists
    if 'marker' in stream_type:
        marker_streams.append(stream)
        print(f'{stream_name} --> marker stream')

    elif stream_type in ('data', 'eeg'):
        data_stream.append(stream)
        print(f'{stream_name} --> data stream')

    else:
        print(f'{stream_name} --> not sure what to do with this')

print()

# Check whether there is only data stream found
if len(data_stream) == 1:
    data_stream = data_stream[0]
elif len(data_stream) == 0:
    raise Exception('No data stream found!')
else:
    raise Exception('Multiple data streams found!')

# Get the sampling frequency
sfreq = float(data_stream['info']['nominal_srate'][0])

# uhb: Unicorn Hybrid Black (g.tec)
if device == 'uhb':
    # Get the data --> only the first 8 rows are for the electrodes
    data = data_stream["time_series"].T[:8]
    # Scale the data
    # data = data * 1e-6
    # Set the channel names as they are not included in the stream
    ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
# ac: actiCHamp (Brain Products)
elif device == 'ac':
    # Initialize an empty array for the channel names
    ch_names = []
    # Navigate to the list of channels in the data stream
    channels = data_stream['info']['desc'][0]['channels'][0]['channel']
    # Iterate over the channels
    for i in range(len(channels)):
        # Append the channel name to the list
        ch_names.append(channels[i]['label'][0])
    # Search for the 'Markers' string that Brain Products puts in the
    # electrode list
    if 'Markers' in ch_names:
        # Remove it from the electrodes list
        ch_names.remove('Markers')
        # Get the data --> remove the row corresponding to 'Markers'
        data = data_stream["time_series"].T[:-1]
    # If it's not in the list
    else:
        # Just take the whole time_series array
        data = data_stream["time_series"].T
else:
    # Get the data
    data = data_stream["time_series"].T
    # Initialize an empty array for the channel names
    ch_names = []
    # Navigate to the list of channels in the data stream
    channels = data_stream['info']['desc'][0]['channels'][0]['channel']

    # Iterate over the channels
    for i in range(len(channels)):
        # Append the channel name to the list
        ch_names.append(channels[i]['label'][0])

# Define the channel types
ch_types = ['eeg'] * len(ch_names)

# Get the time stamps of the EEG recording
raw_time = data_stream["time_stamps"]

# Check if there are available marker streams
if marker_streams:
    for stream in marker_streams:
        # Get the cues
        cues = stream['time_series']

        # If there are no cues
        if not cues:
            # Skip the stream
            print(f"Skipping {stream['info']['name'][0]}\n")
            continue

        # Create a new list consisting of only cue strings
        # since cues is normally a list of lists containing one string
        tmp_list = []
        for cue in cues:
            # Discard the local clock
            # e.g.: 'cue_rest-1649695139.3753629-3966.48780' --> 'cue_rest'
            if '-' in cue[0]:
                tmp_str = cue[0].split('-')[0]

            # Discard the number of occurence at the end of cues
            # e.g.: ['block_begin_1'] --> 'block_begin'
            tmp_str = re.split(r'_\d+$', tmp_str)[0]

            tmp_list.append(tmp_str)
        # Overwrite cues with the newly created list
        cues = tmp_list

        # Get the time stamps of the cues
        cue_times = stream['time_stamps']
        # Get the smallest time stamp of both the data and marker stream
        offset = min(cue_times[0], raw_time[0])
        # Convert the corrected time stamps into indices
        cue_indices = (np.atleast_1d(cue_times) - offset) * sfreq
        cue_indices = cue_indices.astype(int)

        # Initialize the label encoder
        le = LabelEncoder()
        # Encode the cues using the label encoder
        cues_encoded = le.fit_transform(cues)

        if with_stim:
            # Initalize the STIM channel
            stim = np.zeros(data[0].shape)

            # Get the indices of the cues
            # cue_left_idx = list(le.classes_).index('cue_and_imag_left')
            # cue_right_idx = list(le.classes_).index('cue_and_imag_right')

            # Put 0s and 1s to the indices where cues were shown
            # stim[cue_indices[cues_encoded==cue_left_idx]] = 0
            # stim[cue_indices[cues_encoded==cue_right_idx]] = 1
            stim[cue_indices] = cues_encoded + 1

            # Append the STIM channel to the EEG data array
            data = np.concatenate((data, stim.reshape(1, -1)))
            # Add stim to the channel types
            ch_types.append('stim')
            # Add STIM as a channel
            ch_names.append('STIM')

info = mne.create_info(ch_names, sfreq, ch_types)

# Add the system name to the info
if device == 'uhb':
    info['description'] = 'Unicorn Hybrid Black'
elif device == 'ac':
    info['description'] = 'actiCHamp'

raw = mne.io.RawArray(data, info)

# Set the measurement date
tmp_dt = datetime.strptime(header['info']['datetime'][0], "%Y-%m-%dT%H:%M:%S%z")
tmp_dt = tmp_dt.astimezone(timezone.utc)
raw.set_meas_date(tmp_dt)

if device == 'uhb':
    # Create a montage out of the 10-20 system
    montage = mne.channels.make_standard_montage('standard_1020')

    # Apply the montage
    raw.set_montage(montage)

# A line to supress an error message
raw._filenames = [file]

# Convert time stamps of the Raw object to indices
raw_indices = raw.time_as_index(times=raw.times)

# Raise and error if the cue index is larger than the maximum
# index determined by the EEG recording
if cue_indices.max() > raw_indices.max():
    raise Exception(
        'Cue index is larger than the largest sample index of the Raw object!'
    )

# Initialize the event array
event_arr = np.zeros((len(cue_indices), 3), dtype=int)

# Store the event information in an array of shape (len(cues), 3)
event_arr[:, 0] = cue_indices
event_arr[:, 2] = cues_encoded

# Create a class-encoding correspondence dictionary for the Epochs
# object
event_id = dict(zip(list(le.classes_), range(len(le.classes_))))

# %%
