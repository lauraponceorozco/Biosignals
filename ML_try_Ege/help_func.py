from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def interpolate_plateaus_in_ux(df_eeg, column='timestamp_ux'):
    """
    Linearly interpolates plateau segments in the given timestamp column of df_eeg.

    Parameters
    ----------
    df_eeg : pd.DataFrame
        DataFrame containing the timestamp column to be processed.
    column : str
        Name of the column with timestamp values to be interpolated.

    Returns
    -------
    df_eeg : pd.DataFrame
        Modified DataFrame with the interpolated values replacing the original column.
    """
    ux = df_eeg[column].values
    ux_new = np.zeros_like(ux)

    # Find where value changes
    change_indices = np.where(np.diff(ux) != 0)[0] + 1
    segment_boundaries = np.concatenate(([0], change_indices))

    for i in range(len(segment_boundaries) - 1):
        start = segment_boundaries[i]
        end = segment_boundaries[i + 1]

        # Handle short segments
        if (end - start) <= 1:
            ux_new[start:end] = ux[start]
            continue

        # Interpolate linearly
        ux_new[start:end] = np.linspace(ux[start], ux[end], num=(end - start), endpoint=False)

    # Final segment (if any)
    if segment_boundaries[-1] < len(ux) - 1:
        ux_new[segment_boundaries[-1]:] = ux[segment_boundaries[-1]:]

    # Overwrite in DataFrame
    df_eeg[column] = ux_new
    return df_eeg

def extract_and_average_epochs_by_stimulus(df_eeg, df_gui, fs=128, post_time=0.6, n_average=0, normalization="A1",blink_channel_idx=0, blink_threshold=120):
    """
    Extracts EEG epochs aligned to GUI stimulus timestamps,
    applies selected normalization (A1, A2, A3),
    and returns the averaged epoch per stimulus.

    Parameters
    ----------
    df_eeg : pd.DataFrame
        EEG data with 'timestamp' column and one column per channel.
    df_gui : pd.DataFrame
        GUI data with 'timestamp' and 'stimulus' columns.
    fs : int or float
        Sampling frequency in Hz.
    post_time : float
        Time after stimulus in seconds.
    n_average : int
        Number of epochs to average per stimulus. 0 means use all.
    normalization : str
        'A1' = Demean each epoch, then average  
        'A2' = Average all epochs, then demean  
        'A3' = Just average without any demeaning

    Returns
    -------
    averaged_epochs : dict
        Dictionary with keys {1, 2, 3}, each containing array of shape (epoch_len, n_channels)
    """
    eeg_t = df_eeg['timestamp_ux'].values
    eeg_X = df_eeg.drop(columns=['timestamp','timestamp_ux']).values
    n_samples, n_channels = eeg_X.shape

    epoch_len = int(post_time * fs)
    epoch_offsets = np.arange(epoch_len)[None, :]

    averaged_epochs = {}

    for stim in [0, 1, 2]:
        gui_t = df_gui[df_gui['stimulus'] == stim]['timestamp'].values 
        gui_eeg_idx = np.searchsorted(eeg_t, gui_t)
        valid_mask = (gui_eeg_idx + epoch_len < n_samples)
        gui_eeg_idx = gui_eeg_idx[valid_mask]

        if len(gui_eeg_idx) == 0:
            averaged_epochs[stim] = np.full((epoch_len, n_channels), np.nan)
            continue

        if n_average > 0:
            gui_eeg_idx = gui_eeg_idx[:n_average]

        epochs = eeg_X[gui_eeg_idx[:, None] + epoch_offsets]  # (n_epochs, epoch_len, n_channels)
        breakpoint()
        # Fast vectorized blink rejection (based on peak-to-peak amplitude)
        blink_channel = epochs[:, :, blink_channel_idx]
        ptp_amplitude = np.ptp(blink_channel, axis=1)
        keep_mask = ptp_amplitude <= blink_threshold
        epochs = epochs[keep_mask]

        if normalization == "A1":
            epoch_mean = epochs.mean(axis=1, keepdims=True)
            epochs_demeaned = epochs - epoch_mean
            averaged = epochs_demeaned.mean(axis=0)

        elif normalization == "A2":
            averaged = epochs.mean(axis=0)
            averaged = averaged - averaged.mean(axis=0, keepdims=True)

        elif normalization == "A3":
            averaged = epochs.mean(axis=0)

        else:
            raise ValueError("Invalid normalization type. Use 'A1', 'A2', or 'A3'.")

        averaged_epochs[stim] = averaged

    return averaged_epochs

def collect_target_nontarget_epochs(df_eeg, df_gui, fs=128, post_time=0.6, n_average=0, normalization="A1", blink_channel_idx=0, blink_threshold=120):
    """
    Collects target and non-target averaged epochs from all trials.
    
    Returns
    -------
    X_target : list of np.array
        Each array is (epoch_len, n_channels) for target stimulus.
    X_nontarget : list of np.array
        Each array is (epoch_len, n_channels) for averaged non-targets.
    """
    X_target = []
    X_nontarget = []

    for trial_id, trial_gui in df_gui.groupby('trial'):
        # Extract averaged epochs for this trial
        avg_epochs = extract_and_average_epochs_by_stimulus(
            df_eeg, trial_gui,
            fs=fs,
            post_time=post_time,
            n_average=n_average,
            normalization=normalization,
            blink_channel_idx=blink_channel_idx,
            blink_threshold=blink_threshold
        )
        
        target_stim = trial_gui['target'].iloc[0]  # e.g., 0, 1, or 2
        
        # Skip if target epoch is missing (e.g., due to blink rejection)
        if np.isnan(avg_epochs[target_stim]).all():
            continue
        
        # Append target
        X_target.append(avg_epochs[target_stim])
        
        # Average non-targets
        non_target_stims = [s for s in [0, 1, 2] if s != target_stim]
        non_target_epochs = [
            avg_epochs[s] for s in non_target_stims if not np.isnan(avg_epochs[s]).all()
        ]
        if non_target_epochs:
            avg_nontarget = np.mean(non_target_epochs, axis=0)
            X_nontarget.append(avg_nontarget)

    # Combine for final X and y arrays (optional stacking)
    return X_target, X_nontarget

def extract_features_from_averaged_epochs(averaged_epochs, fs=250, feature_type="B1"):
    """
    Extracts features from averaged EEG epochs using different strategies (B1-B5).

    Parameters
    ----------
    averaged_epochs : dict
        Dictionary with keys {1, 2, 3}, each containing a NumPy array of shape (epoch_len, n_channels),
        representing the averaged EEG response for that stimulus.
    fs : int
        Sampling frequency in Hz.
    feature_type : str
        One of the following options:

        - "B1" - Time-series from 290 ms to end of epoch (shape: T × C)
        - "B2" - Same as B1, but decimated by 4 (shape: T/4 × C)
        - "B3" - P300 amplitude: mean ±10 ms around peak in 290-500 ms window (shape: C,)
        - "B4" - P300 amplitude: max value in 290-500 ms (shape: C,)
        - "B5" - P300 amplitude at exactly 310 ms (shape: C,)

    Returns
    -------
    features : dict
        Dictionary with keys {1, 2, 3}, each containing:
        - For B1 and B2: a time - channel matrix (2D array)
        - For B3-B5: a channel-wise summary vector (1D array)
    """
    features = {}
    start_ms = 290
    p300_window = (290, 500)
    window_size_ms = 10
    t_300_ms = 310

    for stim, epoch in averaged_epochs.items():
        if epoch is None or np.all(np.isnan(epoch)):
            features[stim] = None
            continue

        n_time, n_channels = epoch.shape
        time_vector = np.arange(n_time) * (1000 / fs)  # time in ms

        if feature_type == "B1":
            # Time-series from 290 ms onward (shape: T × C)
            mask = time_vector >= start_ms
            feat = epoch[mask]  # shape: (T_b1, C)

        elif feature_type == "B2":
            # Time-series from 290 ms onward, decimated by 4 (shape: T/4 × C)
            mask = time_vector >= start_ms
            feat = epoch[mask][::4]  # shape: (T_b2, C)

        elif feature_type == "B3":
            # Peak in 290–500 → average ±10 ms window around it
            mask = (time_vector >= p300_window[0]) & (time_vector <= p300_window[1])
            windowed = epoch[mask]
            time_in_window = time_vector[mask]
            peak_idx = np.argmax(windowed, axis=0)
            feat = []
            for ch in range(n_channels):
                center_time = time_in_window[peak_idx[ch]]
                win_mask = (time_vector >= center_time - window_size_ms) & (time_vector <= center_time + window_size_ms)
                feat.append(epoch[win_mask, ch].mean())
            feat = np.array(feat)  # shape: (C,)

        elif feature_type == "B4":
            # Max amplitude between 290–500 ms (shape: C,)
            mask = (time_vector >= p300_window[0]) & (time_vector <= p300_window[1])
            feat = epoch[mask].max(axis=0)

        elif feature_type == "B5":
            # Amplitude at exactly 300 ms (shape: C,)
            idx_300 = np.argmin(np.abs(time_vector - t_300_ms))
            feat = epoch[idx_300]

        else:
            raise ValueError("Invalid feature_type. Use 'B1', 'B2', 'B3', 'B4', or 'B5'.")

        features[stim] = feat

    return features
