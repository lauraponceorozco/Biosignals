from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from IPython.display import display
import os
import matplotlib.pyplot as plt

def annotate_flash_and_target_columns(df):
    """
    Adds 'is_target' and 'which_one' columns to the EEG DataFrame based on event codes.

    Parameters
    ----------
    df : pd.DataFrame
        Input EEG DataFrame. Must contain 'Event Id' column.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of the original df with 'is_target' and 'which_one' columns added.
    """

    def parse_event_codes(entry):
        if pd.isna(entry):
            return []
        try:
            codes = str(entry).split(",")[0].split(":")
            return [int(code) for code in codes if code.strip().isdigit()]
        except:
            return []

    def filter_marker(code_list, valid_set):
        if not code_list:
            return -1
        for code in code_list:
            if code in valid_set:
                return code
        return -1

    # Mappings
    target_mapping = {33285: 1, 33286: 0}
    flash_mapping = {
        33025: "row_1", 33026: "row_2", 33027: "row_3",
        33031: "column_1", 33032: "column_2", 33033: "column_3"
    }

    # Parse events
    events_raw = df["Event Id"].values
    parsed_events = [parse_event_codes(e) for e in events_raw]

    # Assign values
    events_target = [filter_marker(e, {33285, 33286}) for e in parsed_events]
    events_flash = [filter_marker(e, set(flash_mapping.keys())) for e in parsed_events]

    # Build new columns
    is_target_col = [target_mapping.get(code, np.nan) for code in events_target]
    which_one_col = [flash_mapping.get(code, None) for code in events_flash]

    # Add to DataFrame
    df_out = df.copy()
    df_out["is_target"] = is_target_col
    df_out["which_one"] = which_one_col

    return df_out

def add_trial_count_column_for_flashes_only(df, flashes_per_trial=72):
    """
    Adds a 'trial_count' column only to rows where both 'is_target' and 'which_one' are present.
    Labels every group of `flashes_per_trial` valid flashes as one trial (0, 1, 2, ...).

    Parameters
    ----------
    df : pd.DataFrame
        Annotated EEG DataFrame that includes 'is_target' and 'which_one'.
    flashes_per_trial : int
        Number of valid flash rows per trial (default: 72).

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with 'trial_count' column added only for valid flash rows.
    """

    df_out = df.copy()
    trial_count = [np.nan] * len(df)

    # Get the indices of rows that have valid flashes
    valid_flash_indices = df[(df['is_target'].notna()) & (df['which_one'].notna())].index.tolist()

    # Assign trial number every 72 valid flashes
    for i, idx in enumerate(valid_flash_indices):
        trial_num = i // flashes_per_trial
        trial_count[idx] = trial_num

    df_out['trial_count'] = trial_count
    return df_out


def epoch_grouped_trialwise_average(df, fs=128, tmin=-0.2, tmax=1.0,
                                    blink_channel_idx=0, blink_threshold=120,
                                    normalization="A1", blink_rejection=True):
    """
    Returns 6 flash-averaged epochs (14 channels) per trial as 1 row each.
    Each channel is stored as a vector. Also returns is_target per flash type.

    Parameters
    ----------
    df : pd.DataFrame with 'trial_count', 'which_one', 'is_target', EEG columns
    fs : int, sampling frequency
    tmin, tmax : float, time window for epochs
    blink_channel_idx : int, channel index used for blink rejection
    blink_threshold : float, max peak-to-peak amplitude
    normalization : str, 'A1', 'A2', or 'A3'
    blink_rejection : bool, whether to apply blink-based epoch rejection

    Returns
    -------
    pd.DataFrame with columns: trial_count, which_one, is_target, and 14 channel vectors
    """

    eeg_cols = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
                'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    pre_samples = int(abs(tmin) * fs)
    post_samples = int(tmax * fs)
    epoch_len = pre_samples + post_samples
    epoch_offsets = np.arange(-pre_samples, post_samples)

    valid = df[df["trial_count"].notna()].copy()
    valid["trial_count"] = valid["trial_count"].astype(int)

    grouped = valid.groupby(["trial_count", "which_one"])
    output_rows = []

    for (trial, flash), group_df in grouped:
        event_idxs = group_df.index.to_numpy()
        label = group_df["is_target"].mode()[0]  # 0 or 1 (should be consistent)

        epochs = []

        for idx in event_idxs:
            if idx - pre_samples < 0 or idx + post_samples >= len(df):
                continue

            epoch = df.loc[idx + epoch_offsets, eeg_cols].to_numpy()
            if epoch.shape[0] != epoch_len:
                continue

            if blink_rejection:
                blink_signal = epoch[:, blink_channel_idx]
                if np.ptp(blink_signal) > blink_threshold:
                    continue

            if normalization == "A1":
                epoch = epoch - epoch.mean(axis=0, keepdims=True)
            elif normalization == "A2":
                pass
            elif normalization == "A3":
                pass
            else:
                raise ValueError("Invalid normalization type.")

            epochs.append(epoch)

        if not epochs:
            continue

        avg_epoch = np.mean(epochs, axis=0)

        if normalization == "A2":
            avg_epoch = avg_epoch - avg_epoch.mean(axis=0, keepdims=True)

        row = {
            "trial_count": trial,
            "which_one": flash,
            "is_target": int(label)
        }
        for ch_idx, ch in enumerate(eeg_cols):
            row[ch] = avg_epoch[:, ch_idx]
        output_rows.append(row)

    return pd.DataFrame(output_rows)

def plot_6_flash_erp_curves(df_flashwise_averages, fs=128, tmin=-0.2, out_dir="new_paradigm", save_name = "erp_target_vs_nontarget.png"):
    """
    Plots and saves 14-channel ERP comparison with 6 curves:
    2 blue (target_row, target_column), 4 red (non-targets: 2 rows, 2 cols).
    """

    os.makedirs(out_dir, exist_ok=True)

    eeg_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    # Extract waveforms based on label patterns
    def get_avg(df, target, label):
        subset = df[(df["is_target"] == target) & (df["which_one"] == label)]
        if subset.empty:
            return {ch: np.zeros(len(subset.iloc[0][ch])) for ch in eeg_channels}
        return {
            ch: np.mean(np.stack(subset[ch].values), axis=0)
            for ch in eeg_channels
        }

    # Choose the fixed 6 labels based on naming pattern
    target_row_label = df_flashwise_averages[df_flashwise_averages["is_target"] == 1]["which_one"].apply(
        lambda x: x if "row" in x.lower() else None).dropna().unique()[0]
    target_column_label = df_flashwise_averages[df_flashwise_averages["is_target"] == 1]["which_one"].apply(
        lambda x: x if "column" in x.lower() else None).dropna().unique()[0]

    # Get remaining non-targets
    row_nontargets = df_flashwise_averages[
        (df_flashwise_averages["is_target"] == 0) & 
        (df_flashwise_averages["which_one"].str.contains("row", case=False))
    ]["which_one"].value_counts().index.tolist()
    col_nontargets = df_flashwise_averages[
        (df_flashwise_averages["is_target"] == 0) & 
        (df_flashwise_averages["which_one"].str.contains("column", case=False))
    ]["which_one"].value_counts().index.tolist()

    row_nontargets = [r for r in row_nontargets if r != target_row_label][:2]
    col_nontargets = [c for c in col_nontargets if c != target_column_label][:2]

    # Sanity fallback
    if len(row_nontargets) < 2 or len(col_nontargets) < 2:
        print("⚠️ Not enough non-target labels to plot all 6 curves.")
        return

    # Retrieve averaged curves
    A = get_avg(df_flashwise_averages, 1, target_row_label)
    B = get_avg(df_flashwise_averages, 1, target_column_label)

    C = get_avg(df_flashwise_averages, 0, row_nontargets[0])
    D = get_avg(df_flashwise_averages, 0, row_nontargets[1])
    E = get_avg(df_flashwise_averages, 0, col_nontargets[0])
    F = get_avg(df_flashwise_averages, 0, col_nontargets[1])

    # Time axis
    epoch_len = len(next(iter(A.values())))
    time_vector = (np.arange(epoch_len) - int(abs(tmin) * fs)) / fs * 1000  # ms

    # Plot
    fig, axes = plt.subplots(7, 2, figsize=(13, 20), sharex=True)
    for i, ax in enumerate(axes.flat):
        if i >= len(eeg_channels):
            ax.axis("off")
            continue
        ch = eeg_channels[i]

        ax.plot(time_vector, A[ch], label=f'Target {target_row_label}', color='blue')
        ax.plot(time_vector, B[ch], label=f'Target {target_column_label}', color='blue', linestyle='--')

        ax.plot(time_vector, C[ch], label=f'Non-Target {row_nontargets[0]}', color='red')
        ax.plot(time_vector, D[ch], label=f'Non-Target {row_nontargets[1]}', color='red', linestyle='--')

        ax.plot(time_vector, E[ch], label=f'Non-Target {col_nontargets[0]}', color='red', linestyle='-.')
        ax.plot(time_vector, F[ch], label=f'Non-Target {col_nontargets[1]}', color='red', linestyle=':')

        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(ch, fontsize=10)
        ax.set_ylabel("µV")
        ax.grid(True)

    axes[-1, 0].set_xlabel("Time (ms)")
    axes[-1, 1].set_xlabel("Time (ms)")
    axes[0, 1].legend(loc='upper right', fontsize=8)

    plt.suptitle("ERP: 2 Targets vs 4 Non-Targets (Row/Column)", fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(out_dir, save_name)
    plt.savefig(save_path, dpi=300)
    plt.show()




def process_and_combine_all_csvs(folder_path, fs=128, tmin=-0.2, tmax=1.0,
                                 normalization="A1", blink_channel_idx=0,
                                 blink_threshold=120,blink_rejection=True):
    """
    Processes all .csv files in a folder and combines the trialwise flash averages,
    adjusting trial_count across files to avoid overlap.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing CSV EEG files.
    fs : int
        Sampling frequency in Hz.
    tmin, tmax : float
        Epoch start and end relative to event (in seconds).
    normalization : str
        'A1', 'A2', or 'A3' normalization scheme.
    blink_channel_idx : int
        Index of channel for blink rejection.
    blink_threshold : float
        Peak-to-peak rejection threshold.

    Returns
    -------
    pd.DataFrame
        Combined flash-wise trial average DataFrame from all files.
    """

    folder_path = Path(folder_path)
    all_csvs = sorted(folder_path.glob("*.csv"))

    all_results = []
    trial_offset = 0

    for file in all_csvs:
        df_eeg = pd.read_csv(file)
        df_annotated = annotate_flash_and_target_columns(df_eeg)
        df_with_trials = add_trial_count_column_for_flashes_only(df_annotated)

        df_flashwise_averages = epoch_grouped_trialwise_average(df_with_trials,
                                                                fs=fs, tmin=tmin, tmax=tmax,
                                                                normalization=normalization,
                                                                blink_channel_idx=blink_channel_idx,
                                                                blink_threshold=blink_threshold,
                                                                blink_rejection=blink_rejection)

        # Shift trial counts to make them unique across files
        df_flashwise_averages["trial_count"] += trial_offset
        trial_offset = df_flashwise_averages["trial_count"].max() + 1

        all_results.append(df_flashwise_averages)

    return pd.concat(all_results, ignore_index=True)

def plot_topoplots_from_combined(df_combined, fs=128, tmin=-0.2,
                                  actual_timepoints_ms=[460, 700],
                                  display_labels_ms=["230", "350"],
                                  out_dir="new_paradigm/topoplots",
                                  addendum=""):
    """
    Plots and saves topoplots at specified timepoints, while labeling using display ms.

    Parameters
    ----------
    df_combined : pd.DataFrame
        Output from process_and_combine_all_csvs().
    fs : int
        Sampling frequency (Hz).
    tmin : float
        Epoch start time in seconds (used for aligning the index).
    actual_timepoints_ms : list of int
        Real timepoints in ms to extract from waveform (e.g., [460, 700]).
    display_labels_ms : list of str
        Labels for plot/saving (e.g., ["230", "350"]) for naming consistency.
    out_dir : str
        Folder where topoplots will be saved.
    """

    assert len(actual_timepoints_ms) == len(display_labels_ms), "Mismatch in timepoint-label pairs."

    # === Channel setup ===
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    n_channels = len(channel_names)

    # === Create output directory ===
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # === Separate conditions ===
    def get_group_avg(df, is_target, include):
        subset = df[(df["is_target"] == is_target) & (df["which_one"].str.contains(include))]
        if subset.empty:
            return np.zeros((len(subset.iloc[0][channel_names[0]]), n_channels))
        return np.mean(
            np.stack([np.stack([row[ch] for ch in channel_names], axis=1)
                      for _, row in subset.iterrows()]), axis=0)

    target_avg = get_group_avg(df_combined, is_target=1, include="")
    nontarget_avg = get_group_avg(df_combined, is_target=0, include="")
    diff_avg = target_avg - nontarget_avg

    # === MNE info ===
    info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # === Plot each timepoint ===
    for true_ms, label_ms in zip(actual_timepoints_ms, display_labels_ms):
        time_idx = int((tmin + (true_ms / 1000.0)) * fs)

        val_target = target_avg[time_idx]
        val_nontarget = nontarget_avg[time_idx]
        val_diff = diff_avg[time_idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for i, (data, title) in enumerate(zip(
            [val_target, val_nontarget, val_diff],
            ["Target", "Non-target", "Target − Non-target"]
        )):
            vmin, vmax = (np.min(val_diff), np.max(val_diff)) if i == 2 else (np.min(data), np.max(data))

            im, _ = mne.viz.plot_topomap(
                data,
                pos=info,
                axes=axes[i],
                show=False,
                cmap='RdBu_r',
                contours=0,
                names=channel_names,
                vlim=(vmin, vmax)
            )
            axes[i].set_title(f"{title} @ {label_ms} ms")
            plt.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)

        fig.suptitle(f"Topomap at {label_ms} ms", fontsize=16)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"Topomap_{label_ms}ms_{addendum}.png", dpi=300)
        plt.show()




### ML methods
def extract_features_from_df_combined(df_combined, fs=128, feature_types=["B3", "B4", "B5"]):
    """
    Extracts features (B3, B4, B5) from df_combined per trial × flash.
    
    Parameters
    ----------
    df_combined : pd.DataFrame
        Flash-wise averaged ERP data with trial_count, which_one, is_target, and channel vectors.
    fs : int
        Sampling frequency.
    feature_types : list
        List of feature types to compute: any of ["B3", "B4", "B5"]

    Returns
    -------
    X : np.ndarray of shape (n_trials * 6, n_features)
    y : np.ndarray of shape (n_trials * 6,)
    """

    import numpy as np

    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    timepoints = {
        "300": {"win": (600, 800), "center": 700},
        "200": {"win": (400, 600), "center": 460}
    }

    def time_to_idx(ms):
        return int((ms / 1000.0) * fs)

    def extract_features(epoch, feature_type, window, center):
        t = np.arange(epoch.shape[0]) * 1000 / fs  # ms
        mask = (t >= window[0]) & (t <= window[1])
        sub_epoch = epoch[mask]
        t_window = t[mask]

        if feature_type == "B3":
            # abs peak ±10 ms
            feat = []
            for ch in range(epoch.shape[1]):
                abs_sig = np.abs(sub_epoch[:, ch])
                peak_idx = np.argmax(abs_sig)
                peak_time = t_window[peak_idx]
                around_mask = (t >= peak_time - 10) & (t <= peak_time + 10)
                feat.append(epoch[around_mask, ch].mean())
            return np.array(feat)

        elif feature_type == "B4":
            # now uses max(abs(signal))
            return np.abs(sub_epoch).max(axis=0)

        elif feature_type == "B5":
            center_idx = np.argmin(np.abs(t - center))
            return epoch[center_idx, :]

        else:
            raise ValueError("Invalid feature type.")


    X = []
    y = []

    for trial_id, trial_df in df_combined.groupby("trial_count"):
        # Sort in order: column_1, column_2, column_3, row_1, row_2, row_3
        col_df = trial_df[trial_df["which_one"].str.contains("column", case=False)].sort_values("which_one")
        row_df = trial_df[trial_df["which_one"].str.contains("row", case=False)].sort_values("which_one")
        ordered_df = pd.concat([col_df, row_df])

        for _, row in ordered_df.iterrows():
            epoch = np.stack([row[ch] for ch in channel_names], axis=1)  # shape: time × ch
            feature_vector = []

            for tp in ["300", "200"]:
                for ft in feature_types:
                    feat = extract_features(epoch, ft, window=timepoints[tp]["win"], center=timepoints[tp]["center"])
                    feature_vector.extend(feat)

            X.append(feature_vector)
            y.append(int(row["is_target"]))

    return np.array(X), np.array(y)