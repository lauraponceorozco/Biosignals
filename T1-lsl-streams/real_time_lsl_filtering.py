# %%
"""
Real-time filtering of incoming Unicorn Hybrid Black EEG data streamed through
LSL. The filtered data is then streamed back out under a new LSL stream.

Author:
    Karahan Yilmazer

Email:
    yilmazerkarahan@gmail.com
"""

import mne
import numpy as np
import tqdm
from pylsl import StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_streams
from scipy import signal


# %%
def receive_eeg_samples(
    inlet, samples_buffer, timestamps_buffer, buffer_size=5000, chunk_size=100
):
    """
    Receives new EEG samples and timestamps from the LSL input and stores them in the buffer variables
    :param samples_buffer: list of samples(each itself a list of values)
    :param timestamps_buffer: list of time-stamps
    :return: updated samples_buffer and timestamps_buffer with the most recent 150 samples
    """

    # Pull a chunk of maximum chunk_size samples
    chunk, timestamps = inlet.pull_chunk(max_samples=chunk_size)

    # If there are no new samples
    if chunk == []:
        # Return the unchanged buffer
        return samples_buffer, timestamps_buffer, 0
    # If there are new samples
    else:
        # Get the number of newly fetched samples
        n_new_samples = len(chunk)

        # Convert the chunk into a np.array and transpose it
        samples = [sample[:8] for sample in chunk]

        # Extend the buffer with the data
        samples_buffer.extend(samples)
        # Extend the buffer with time stamps
        timestamps_buffer.extend(timestamps)

        # Get the last buffer_size samples and time stamps from the buffer
        data_from_buffer = samples_buffer[-buffer_size:]
        timestamps_from_buffer = timestamps_buffer[-buffer_size:]

        return data_from_buffer, timestamps_from_buffer, n_new_samples


if __name__ == '__main__':
    # Search for available LSL streams
    print("Looking for an LSL stream...")
    infos = resolve_streams()

    # Predicate to look for in the stream name
    pred = 'UN'
    # pred = 'Liesl'

    # Search for the stream with the defined predicate (optional)
    info_inlet = None
    for info in infos:
        if pred in info.name():
            info_inlet = info
            break

    # Connect to the LSL stream
    if info_inlet is not None:
        inlet = StreamInlet(info, max_buflen=5)
        inlet_name = inlet.info().name()
        print("Connected to the inlet:", inlet_name)
    else:
        print(f"Could not find a stream starting with {pred}.")

    # Supress MNE info messages and only show warnings
    mne.set_log_level('WARNING')

    # Define the outlet stream name
    outlet_name = inlet_name + '_filtered'

    # Get the sampling frequency of the device
    sfreq = inlet.info().nominal_srate()

    # Define the outlet information
    info_outlet = StreamInfo(
        name=outlet_name,
        type='eeg',
        channel_count=8,
        channel_format='float32',
        source_id=inlet.info().source_id(),
        nominal_srate=sfreq,
    )

    # Create the LSL outlet
    outlet = StreamOutlet(info_outlet)

    # Define the channel names
    ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

    # Create the info structure needed by MNE
    info = mne.create_info(ch_names, sfreq, 'eeg')

    # Add the system name to the info
    info['description'] = 'Unicorn Hybrid Black'

    # Initialize the marker list
    marker_list = []

    # Initalize the buffers
    samples_buffer = []
    timestamps_buffer = []

    # Set the values for buffering
    buffer_size = 2500
    chunk_size = 125
    delay = 25

    # Define the filter
    sos = signal.butter(10, (0.5, 30), fs=sfreq, btype='bandpass', output='sos')

    # Progress bar
    pbar = tqdm.tqdm(desc='Calibration', total=buffer_size)
    pbar_closed = False
    old_val = 0

    # Start accepting samples from the stream
    inlet.open_stream()

    # Empty the buffer for a fresh start
    flush_clock = local_clock()
    inlet.flush()

    try:
        while True:
            # Get the EEG samples, time stamps and the index that tells from which point on the new samples have been appended to the buffer
            samples_buffer, timestamps_buffer, n_new_samples = receive_eeg_samples(
                inlet,
                samples_buffer,
                timestamps_buffer,
                buffer_size=buffer_size,
                chunk_size=chunk_size,
            )

            # If the calibration is offer
            if pbar_closed:
                # If there are new samples available
                if n_new_samples > 0:
                    # Convert the lists into a numpy array
                    buffer_array = np.array(samples_buffer)
                    # Filter the data using an IIR filter
                    filt_data = signal.sosfilt(sos, buffer_array, axis=0)
                    # Push the newly filtered data to the stream
                    outlet.push_chunk(filt_data[-n_new_samples:, :].tolist())
            else:
                # Get the number of samples in the buffer
                len_buffer = len(timestamps_buffer)
                # Calculate the update step
                update_val = len_buffer - old_val
                # Store the length of the buffer for the next iteration
                old_val = len_buffer
                # Update the progress bar
                pbar.update(update_val)
                # If the buffer is full
                if len_buffer == buffer_size:
                    # Close the progress bar
                    pbar.close()
                    # Break out of the loop
                    pbar_closed = True

    except KeyboardInterrupt:
        # Kill the outlet
        outlet.__del__()
# %%
