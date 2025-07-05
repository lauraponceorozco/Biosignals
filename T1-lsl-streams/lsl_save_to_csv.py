# %%
import os
from datetime import datetime

import pandas as pd
from pylsl import StreamInlet, resolve_stream

# %%
# Define the LSL stream name
stream_name = 'Unicorn'

# Connect to the stream
stream = resolve_stream('name', stream_name)
inlet = StreamInlet(stream[0])

# Get the current time
time_now = datetime.now()
# Prepare a string in format YYYY_MM_DD_HH_MinMin
date_time = (
    f'{time_now.year}_{str(time_now.month).zfill(2)}_'
    f'{str(time_now.day).zfill(2)}_{time_now.hour}_{time_now.minute}'
)

# Define the recording file name
file_name = 'unicorn_recording_' + date_time + '.csv'

# Create a folder called 'data' if there is none already
if 'data' not in os.listdir():
    os.mkdir('data')

# Redefine the recording file name to save the recording in the 'data' folder
file_name = os.path.join('data', file_name)

# Initialize lists for storing data
sample_list = []
timestamp_list = []

# Empty the inlet
inlet.flush()

# Run the program
try:
    while True:
        # Pull a sample from the inlet
        sample, timestamp = inlet.pull_sample()

        # Append the sample and timestamp to their corresponding lists
        sample_list.append(sample)
        timestamp_list.append(timestamp)

# When the program is interrupted by the user
except KeyboardInterrupt:
    # Define the channel names
    columns = [
        'Fz',
        'C3',
        'Cz',
        'C4',
        'Pz',
        'PO7',
        'Oz',
        'PO8',
        'ACC_X',
        'ACC_Y',
        'ACC_Y',
        'GYRO_X',
        'GYRO_Y',
        'GYRO_Z',
        'Battery',
        'Counter',
        'Validation',
    ]

    # Save the data in a DataFrame
    df = pd.DataFrame(sample_list, index=timestamp_list, columns=columns)
    # Export the DataFrame to a CSV file
    df.to_csv(file_name)
# %%
