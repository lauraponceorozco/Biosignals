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


def align_all_timestamps(df_eeg,df_gui):
    # no interpolation! Instead: use lsl timestamps and correct duration

    # zero the timestamps
    t_ref = df_eeg['timestamp_ux'].iloc[0]
    t_ux = df_eeg['timestamp_ux'] - df_eeg['timestamp_ux'].iloc[0]
    t_lsl = df_eeg['timestamp'] - df_eeg['timestamp'].iloc[0] #also start at zero and at the eqal time as t_ux

    t_gui = df_gui['timestamp'] - t_ref # use the same reference as for t_ux

    # asign times back to the dataframe
    df_eeg['timestamp_ux'] = t_ux
    df_eeg['timestamp'] = t_lsl
    df_gui['timestamp'] = t_gui

    # more precise timing: in delete the very last row, where ux timestamps repeat
    #print(len(df_eeg))
    for i in range(len(df_eeg) - 1, -1, -1): # loop reverse
        if df_eeg['timestamp_ux'].iloc[i] == df_eeg['timestamp_ux'].iloc[i - 1]:
            df_eeg = df_eeg.drop(i)
    #        print(f"Drop row {i} with repeated ux timestamp")
        else:
            break
    #print(df_eeg['timestamp_ux'])


    # scale the duration of the lsl timestamps to match the total recording duration
    t_end_ux = df_eeg['timestamp_ux'].iloc[-1]
    t_end_lsl = df_eeg['timestamp'].iloc[-1]
    scale_factor = t_end_ux / t_end_lsl
    df_eeg['timestamp'] = df_eeg['timestamp'] * scale_factor

    # The code was using timestamp_ux for plotting, so overwrite with timestamp instead
    df_eeg['timestamp_ux'] = df_eeg['timestamp']
    #print(df_eeg)
    #print(df_gui)

    return df_eeg, df_gui



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
        # breakpoint()
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
    optionally tailored for 300ms or 200ms P300-type windows.

    Parameters
    ----------
    averaged_epochs : dict
        Dictionary with keys {1, 2, 3}, each containing a NumPy array of shape (epoch_len, n_channels).
    fs : int
        Sampling frequency in Hz.
    feature_type : str
        Feature extraction type: "B1", "B2", "B3", "B4", "B5".
    timepoint : str
        Either "300" or "200", controls the time windows:
            - "300": B3/B4 use 300–400 ms, B5 at 350 ms
            - "200": B3/B4 use 250–300 ms, B5 at 280 ms

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
    else:
        raise ValueError("Invalid timepoint. Choose '300' or '200'.")

    start_ms = 290
    window_size_ms = 10  # For ±10 ms around peak

    for stim, epoch in averaged_epochs.items():
        if epoch is None or np.all(np.isnan(epoch)):
            features[stim] = None
            continue

        n_time, n_channels = epoch.shape
        time_vector = np.arange(n_time) * (1000 / fs)  # time in ms

        if feature_type == "B1":
            # Time-series from 290 ms onward
            mask = time_vector >= start_ms
            feat = epoch[mask]

        elif feature_type == "B2":
            # Time-series from 290 ms onward, decimated by 4
            mask = time_vector >= start_ms
            feat = epoch[mask][::4]

        elif feature_type == "B3":
            # Peak in p300_window → average ±10 ms
            mask = (time_vector >= p300_window[0]) & (time_vector <= p300_window[1])
            windowed = epoch[mask]
            time_in_window = time_vector[mask]
            peak_idx = np.argmax(windowed, axis=0)
            feat = []
            for ch in range(n_channels):
                center_time = time_in_window[peak_idx[ch]]
                win_mask = (time_vector >= center_time - window_size_ms) & (time_vector <= center_time + window_size_ms)
                feat.append(epoch[win_mask, ch].mean())
            feat = np.array(feat)

        elif feature_type == "B4":
            # Max amplitude in p300_window
            mask = (time_vector >= p300_window[0]) & (time_vector <= p300_window[1])
            feat = epoch[mask].max(axis=0)

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
        #df_eeg = interpolate_plateaus_in_ux(df_eeg)
        df_eeg, df_gui = align_all_timestamps(df_eeg, df_gui)

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
                    features_200 = extract_features_from_averaged_epochs(
                        averaged_epochs,
                        fs=fs,
                        feature_type=feature_type,
                        timepoint="200"
                    )

                    f300 = features_300.get(stim)
                    f200 = features_200.get(stim)

                    if f300 is None or f200 is None:
                        trial_features = None
                        break

                    trial_features.extend(f300)
                    trial_features.extend(f200)

                if trial_features is not None:
                    all_X.append(trial_features)
                    all_y.append(1 if stim == target else 0)

    X = np.vstack(all_X)
    y = np.array(all_y)
    return X, y


def collect_combined_features_all_sessions_multiclass(
    base_folder,
    fs=128,
    post_time=0.6,
    n_average=0,
    normalization="A1",
    feature_types=["B3"],
    blink_channel_idx=0,
    blink_threshold=120
):
    """
    For each trial, extract and concatenate the features of all 3 stimuli (0,1,2),
    returning one feature vector per trial. Label is the true target stimulus (0/1/2).

    Returns
    -------
    X : np.ndarray
        Feature matrix (n_trials, 3 × n_features_per_stimulus)
    y : np.ndarray
        Labels (0, 1, or 2) → the target stimulus of the trial
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
        df_eeg, df_gui = align_all_timestamps(df_eeg, df_gui)

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

            full_trial_features = []
            success = True

            for stim in [0, 1, 2]:
                stim_features = []

                for feature_type in feature_types:
                    f300 = extract_features_from_averaged_epochs(
                        averaged_epochs,
                        fs=fs,
                        feature_type=feature_type,
                        timepoint="300"
                    ).get(stim)

                    f200 = extract_features_from_averaged_epochs(
                        averaged_epochs,
                        fs=fs,
                        feature_type=feature_type,
                        timepoint="200"
                    ).get(stim)

                    if f300 is None or f200 is None:
                        success = False
                        break

                    stim_features.extend(f300)
                    stim_features.extend(f200)

                if not success:
                    break

                full_trial_features.extend(stim_features)

            if success:
                all_X.append(full_trial_features)
                all_y.append(trial_gui["target"].iloc[0])  # Label = target stim (0,1,2)

    X = np.vstack(all_X)
    y = np.array(all_y)
    return X, y

