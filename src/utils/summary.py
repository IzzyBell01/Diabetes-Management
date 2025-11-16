import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def meal_summary(df, key):
    meals = df["carbInput"].dropna()
    positive_meals = meals[meals > 0]

    # Robust split detection
    if "train" in key.lower():
        split = "train"
    elif "test" in key.lower():
        split = "test"
    else:
        split = "unknown"

    # Extract dataset and patient more safely
    parts = key.split("_")
    dataset = parts[0] if len(parts) > 0 else "unknown"
    patient = parts[1] if len(parts) > 1 else "unknown"

    return {
        "dataset": dataset,
        "patient": patient,
        "split": split,
        "total_rows": len(df),
        "total_meal_annotations": len(meals),
        "positive_meals": len(positive_meals),
        "total_CHO_grams": positive_meals.sum(),
    }


def build_summary(patient_data):
    """Builds a summary DataFrame from all patients (raw CHO)."""
    summaries = [meal_summary(df, k) for k, df in patient_data.items()]
    summary_df = (
        pd.DataFrame(summaries)
        .sort_values(["dataset", "positive_meals"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary_df


def build_label_summary(model_input_data):
    """
    Summarize how many 0/1 labels we have per patient after windowing.

    Parameters
    ----------
    model_input_data : dict
        {
          patient_key: {
            "X": np.ndarray (n_windows, timesteps, features),
            "y": np.ndarray (n_windows,),
            "meta": dict (optional)
          },
          ...
        }

    Returns
    -------
    label_df : pd.DataFrame
        Columns:
        - dataset
        - patient
        - split
        - n_windows
        - n_label_0
        - n_label_1
        - positive_ratio
    """
    rows = []
    for key, data_dict in model_input_data.items():
        y = np.asarray(data_dict["y"]).astype(int).ravel()
        n_windows = len(y)
        n1 = int(y.sum())
        n0 = int(n_windows - n1)

        # Reuse your robust parsing from meal_summary
        parts = key.split("_")
        dataset = parts[0] if len(parts) > 0 else "unknown"
        patient = parts[1] if len(parts) > 1 else "unknown"

        # Detect split from key
        key_lower = key.lower()
        if "train" in key_lower:
            split = "train"
        elif "test" in key_lower:
            split = "test"
        else:
            split = "unknown"

        rows.append(
            {
                "dataset": dataset,
                "patient": patient,
                "split": split,
                "n_windows": n_windows,
                "n_label_0": n0,
                "n_label_1": n1,
                "positive_ratio": n1 / n_windows if n_windows > 0 else np.nan,
            }
        )

    label_df = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "split", "positive_ratio"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    return label_df


def plot_label_counts(label_summary_df, title_suffix="(window labels)"):
    """
    Bar chart of label 0 vs 1 counts per patient.

    Expects columns:
        - 'dataset'
        - 'patient'
        - 'split'
        - 'n_label_0'
        - 'n_label_1'
    """
    df = label_summary_df.copy()

    # Build a compact x-axis label like "Ohio2018_559_train"
    df["x_label"] = (
        df["dataset"].astype(str)
        + "_"
        + df["patient"].astype(str)
        + "_"
        + df["split"].astype(str)
    )

    x = np.arange(len(df))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width / 2, df["n_label_0"], width, label="label 0 (no meal)")
    ax.bar(x + width / 2, df["n_label_1"], width, label="label 1 (meal)")

    ax.set_xticks(x)
    ax.set_xticklabels(df["x_label"], rotation=90)
    ax.set_ylabel("Number of windows")
    ax.set_title(f"Label distribution: 0 vs 1 per patient {title_suffix}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
