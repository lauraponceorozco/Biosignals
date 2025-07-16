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

def collect_target_nontarget_diff_epochs(df_eeg, df_gui, fs=128, post_time=0.6, n_average=0, normalization="A1", blink_channel_idx=0, blink_threshold=120):
    """
    Collects target and non-target averaged epochs from all trials.
    
    Returns
    -------
    X_target : list of np.array
        Each array is (epoch_len, n_channels) for target stimulus.
    X_nontarget : list of np.array
        Each array is (epoch_len, n_channels) for averaged non-targets.
    X_diff : list of np.array
        Each array is (epoch_len, n_channels) for the difference between target and non-targets.
    """
    X_target = []
    X_nontarget = []
    X_diff = []

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
            X_diff.append(avg_epochs[target_stim] - avg_nontarget)

    # Combine for final X and y arrays (optional stacking)
    return X_target, X_nontarget, X_diff

def extract_features_from_averaged_epochs(averaged_epochs, fs=250, feature_type="B1", timepoint="300"):
    """
    Extracts features from averaged EEG epochs using different strategies (B1-B5),
    optionally tailored for P300-type windows (200, 300, 400).

    Parameters
    ----------
    averaged_epochs : dict
        Dictionary with keys {1, 2, 3}, each containing a NumPy array of shape (epoch_len, n_channels).
    fs : int
        Sampling frequency in Hz.
    feature_type : str
        Feature extraction type: "B1", "B2", "B3", "B4", "B5".
    timepoint : str
        Either "200", "300", or "400", controls the time windows:
            - "300": B3/B4 use 300–400 ms, B5 at 350 ms
            - "200": B3/B4 use 250–300 ms, B5 at 280 ms
            - "400": B3/B4 use 400–500 ms, B5 at 450 ms

    Returns
    -------
    features : dict
        Extracted features for each stimulus.
    """
    features = {}

    # Time parameters
    if timepoint == "300":
        p300_window = (300, 400)
        t_fixed = 350
    elif timepoint == "200":
        p300_window = (250, 300)
        t_fixed = 280
    elif timepoint == "400":
        p300_window = (400, 500)
        t_fixed = 450
    else:
        raise ValueError("Invalid timepoint. Choose '200', '300', or '400'.")

    start_ms = 290
    window_size_ms = 15  # ±10 ms around peak

    for stim, epoch in averaged_epochs.items():
        if epoch is None or np.all(np.isnan(epoch)):
            features[stim] = None
            continue

        n_time, n_channels = epoch.shape
        time_vector = np.arange(n_time) * (1000 / fs)  # time in ms

        if feature_type == "B1":
            mask = time_vector >= start_ms
            feat = epoch[mask]

        elif feature_type == "B2":
            mask = time_vector >= start_ms
            feat = epoch[mask][::4]

        elif feature_type == "B3":
            # Peak (by absolute value) in p300_window → average ±10 ms
            mask = (time_vector >= p300_window[0]) & (time_vector <= p300_window[1])
            windowed = epoch[mask]
            time_in_window = time_vector[mask]
            feat = []
            for ch in range(n_channels):
                peak_idx = np.argmax(np.abs(windowed[:, ch]))
                center_time = time_in_window[peak_idx]
                win_mask = (time_vector >= center_time - window_size_ms) & (time_vector <= center_time + window_size_ms)
                feat.append(epoch[win_mask, ch].mean())
            feat = np.array(feat)

        elif feature_type == "B4":
            # Max absolute amplitude in p300_window
            mask = (time_vector >= p300_window[0]) & (time_vector <= p300_window[1])
            feat = np.abs(epoch[mask]).max(axis=0)

        elif feature_type == "B5":
            # Amplitude at t_fixed
            idx_fixed = np.argmin(np.abs(time_vector - t_fixed))
            feat = epoch[idx_fixed]

        else:
            raise ValueError("Invalid feature_type. Use 'B1'–'B5'.")

        features[stim] = feat

    return features



def collect_combined_features_all_sessions(
    base_folder,
    fs=128,
    post_time=0.6,
    n_average=0,
    normalization="A1",
    feature_types=["B3"],  # Can be ["B3", "B4", "B5"]
    blink_channel_idx=0,
    blink_threshold=120
):
    """
    Processes all sessions and extracts combined 300 ms and 200 ms features
    for both target and non-target stimuli per trial.

    Returns
    -------
    X : np.ndarray
        Combined feature matrix (n_trials × 2, n_features)
    y : np.ndarray
        Labels (1 = target, 0 = non-target)
    """
    all_X = []
    all_y = []
    trial_offset = 0

    for session_folder in sorted(Path(base_folder).glob("session_*")):
        eeg_path = session_folder / "eeg_data.csv"
        gui_path = session_folder / "gui_data.csv"
        
        if not eeg_path.exists() or not gui_path.exists():
            continue

        df_eeg = pd.read_csv(eeg_path)
        df_gui = pd.read_csv(gui_path)
        df_eeg = interpolate_plateaus_in_ux(df_eeg)

        df_gui['trial'] += trial_offset
        trial_offset = df_gui['trial'].max() + 1

        for trial_id, trial_gui in df_gui.groupby("trial"):
            averaged_epochs = extract_and_average_epochs_by_stimulus(
                df_eeg=df_eeg,
                df_gui=trial_gui,
                fs=fs,
                post_time=post_time,
                n_average=n_average,
                normalization=normalization,
                blink_channel_idx=blink_channel_idx,
                blink_threshold=blink_threshold
            )

            target = trial_gui["target"].iloc[0]
            stimuli = [0, 1, 2]

            for stim in stimuli:
                trial_features = []

                for feature_type in feature_types:
                    features_300 = extract_features_from_averaged_epochs(
                        averaged_epochs,
                        fs=fs,
                        feature_type=feature_type,
                        timepoint="300"
                    )
                    features_400 = extract_features_from_averaged_epochs(
                        averaged_epochs,
                        fs=fs,
                        feature_type=feature_type,
                        timepoint="400"
                    )

                    f300 = features_300.get(stim)
                    f400 = features_400.get(stim)

                    if f300 is None or f400 is None:
                        trial_features = None
                        break

                    trial_features.extend(f300)
                    trial_features.extend(f400)

                if trial_features is not None:
                    all_X.append(trial_features)
                    all_y.append(1 if stim == target else 0)

    X = np.vstack(all_X)
    y = np.array(all_y)
    return X, y
