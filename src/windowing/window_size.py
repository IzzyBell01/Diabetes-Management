import numpy as np
import pandas as pd


def window_size(segments, window_minutes=150, stride_minutes=15, sample_every=5):
    """
    Build sliding windows across multiple continuous CGM segments.

    Parameters
    ----------
    segments : list[pd.DataFrame]
        Continuous, preprocessed DataFrames (each with 'cbg' and optionally 'carbInput')
    window_minutes : int
        Duration of each window in minutes (default 150 = 2.5 hours)
    stride_minutes : int
        Step size between consecutive windows (default 15)
    sample_every : int
        Sampling interval in minutes (default 5)

    Returns
    -------
    X : np.ndarray
        Array of CGM sequences, shape (n_windows, window_length)
    y : np.ndarray
        Binary labels (1 = meal in window, 0 = no meal)
    meta : pd.DataFrame
        Metadata for each window (segment id, start/end times, etc.)
    """

    X_all, y_all, meta_all = [], [], []

    # number of samples per window and stride
    W = int(window_minutes / sample_every)
    S = int(stride_minutes / sample_every)

    for seg_idx, df in enumerate(segments):
        if "cbg" not in df.columns:
            continue

        # Fill missing carbInput column with zeros (if not present)
        if "carbInput" not in df.columns:
            df["carbInput"] = 0

        # Convert index to numeric time (minutes) if datetime
        if isinstance(df.index, pd.DatetimeIndex):
            t = (df.index - df.index[0]).total_seconds() / 60.0
        else:
            t = np.arange(len(df)) * sample_every

        cbg = df["cbg"].astype("float32").values
        carbs = df["carbInput"].fillna(0).values

        if len(df) < W:
            # Skip too-short segments
            continue

        starts = np.arange(0, len(df) - W + 1, S)

        for s in starts:
            e = s + W
            window_cbg = cbg[s:e]
            window_carbs = carbs[s:e]
            label = 1 if (window_carbs > 0).any() else 0

            X_all.append(window_cbg)
            y_all.append(label)
            meta_all.append({
                "segment": seg_idx,
                "start_idx": s,
                "end_idx": e,
                "start_time_min": t[s],
                "end_time_min": t[e - 1],
                "label": label
            })

    # --- Convert outputs ---
    X = np.asarray(X_all, dtype="float32")
    y = np.asarray(y_all, dtype="int32")
    meta = pd.DataFrame(meta_all)

    print(f"✅ Created {len(X)} windows across {len(segments)} segments "
          f"(window={window_minutes}min, stride={stride_minutes}min).")

    return X, y, meta