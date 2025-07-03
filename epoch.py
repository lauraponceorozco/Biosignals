import pandas as pd
import numpy as np
#we want to create eeg epochs from 100ms before stimulu to 600 ms after stimulus
eeg = pd.read_csv("eeg_data.csv")           # contains timestamp, eeg activity of each channel(?)
stim = pd.read_csv("stimulus_log.csv")      # contains timestamp,stimulus_idx,stimulus_name,is_target

sfreq = 256  # sampling rate in Hz
pre = 0.1    # 100 ms before stimulus
post = 0.6   # 600 ms after stimulus

epochs = []     # Will hold EEG epochs: (samples, channels)
labels = []     # Will hold 1 (target) or 0 (non-target) for each epoch


#get just the timestamps
eeg_timestamps = eeg['timestamp'].to_numpy()

# Get just the EEG values (samples, channels)
eeg_data = eeg.drop(columns=['timestamp']).to_numpy()

for _, row in stim.iterrows(): 
    stim_time = row['timestamp']       # When the stimulus happened
    is_target = row['is_target']       # 1 = the user focused on it, 0 = they didn't
    t_start = stim_time - pre
    t_end = stim_time + post


    idx_start = np.searchsorted(eeg_timestamps, t_start) #gives you the position (index) of t_start value in the eeg_timestamps array
    idx_end = np.searchsorted(eeg_timestamps, t_end)

    expected_len = int((pre + post) * sfreq) #calculates expected samples in each window
    if idx_end - idx_start == expected_len: #if the segment that we are choosing is too short(for example at the end of the recording) we skip it
        epoch = eeg_data[idx_start:idx_end, :] #Selects all rows from idx_start to idx_end - 1, and all columns(the different channels)

    epochs.append(epoch) #shape: (samples_in_epoch, channels)
    labels.append(is_target)

X = np.stack(epochs)  # input. shape: (n_trials, n_samples, n_channels) instead of having many individual 2d arrays [(),(),()], we have 1  3d array [[()],[()], [()]]
y = np.array(labels)  # output