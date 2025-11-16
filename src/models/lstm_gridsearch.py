import os
import random
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
SEED = 42
MAX_COMBOS = 10                 # how many hyperparameter combinations per patient
VAL_SPLIT = 0.2                 # 80% train / 20% validation
OUT_DIR = "results/lstm_gridsearch"  # where to save curves & summaries

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# ------------------------------------------------


def build_model(
    timesteps,
    features,
    n_lstm_layers=1,
    lr=1e-3,
    batchnorm=True,
    dense1_units=64,
    dense2_units=32,
    lstm_dropout=0.2,
    lstm_recurrent_dropout=0.0,
    dense_dropout=0.3,
):
    """
    Build a binary LSTM classifier with 1–3 stacked LSTM layers,
    1–2 Dense layers, dropout scheduling, and optional BatchNorm.
    """

    inputs = layers.Input(shape=(timesteps, features))

    # Dropout schedule per LSTM layer
    d1 = float(np.clip(lstm_dropout, 0.0, 0.5))
    d2 = float(np.clip(lstm_dropout + 0.1, 0.0, 0.5))
    d3 = float(np.clip(lstm_dropout + 0.2, 0.0, 0.5))

    # ---- LSTM #1 ----
    x = layers.LSTM(
        128,
        return_sequences=(n_lstm_layers > 1),
        dropout=d1,
        recurrent_dropout=lstm_recurrent_dropout,
    )(inputs)

    # ---- LSTM #2 ----
    if n_lstm_layers >= 2:
        x = layers.LSTM(
            64,
            return_sequences=(n_lstm_layers > 2),
            dropout=d2,
            recurrent_dropout=lstm_recurrent_dropout,
        )(x)

    # ---- LSTM #3 ----
    if n_lstm_layers >= 3:
        x = layers.LSTM(
            32,
            return_sequences=False,
            dropout=d3,
            recurrent_dropout=lstm_recurrent_dropout,
        )(x)

    # ---- Dense layer 1 ----
    if dense1_units is not None and dense1_units > 0:
        x = layers.Dense(dense1_units)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dense_dropout)(x)

    # ---- Dense layer 2 ----
    if dense2_units is not None and dense2_units > 0:
        x = layers.Dense(dense2_units)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    # ---- Output layer ----
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tune_threshold_by_f1(y_true, y_prob, grid=np.linspace(0.05, 0.95, 37)):
    """Find the threshold giving the best F1-score."""
    best = {"t": 0.5, "acc": 0, "prec": 0, "rec": 0, "f1": 0}
    for t in grid:
        pred = (y_prob >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best.update({"t": float(t), "prec": prec, "rec": rec, "f1": f1})
    return best


def plot_curves(hist, save_path):
    """Save loss and accuracy curves from model training."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(hist.history["loss"], label="train")
    ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    if "accuracy" in hist.history:
        ax[1].plot(hist.history["accuracy"], label="train")
    if "val_accuracy" in hist.history:
        ax[1].plot(hist.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def run_personalized_lstm_search(model_input_data):
    """
    Personalized grid search: one model per patient.
    """

    print("=== Starting personalized LSTM grid search ===")
    results = []
    start_all = datetime.now()

    for pid, data_dict in model_input_data.items():
        print(f"\n--- Processing {pid} ---")

        X_all = data_dict["X"]
        y_all = data_dict["y"]

        if len(X_all) < 80 or y_all.sum() == 0:
            print(f"Skipping {pid}: insufficient windows or no positive labels.")
            continue

        # Ensure correct shape
        if X_all.ndim == 2:
            X_all = X_all[..., np.newaxis]

        n = len(X_all)
        split_idx = int(n * (1 - VAL_SPLIT))
        X_tr, y_tr = X_all[:split_idx], y_all[:split_idx]
        X_va, y_va = X_all[split_idx:], y_all[split_idx:]

        if y_tr.sum() == 0 or y_va.sum() == 0:
            print(f"Skipping {pid}: no positive labels in train/validation.")
            continue

        # ---- Class weights per label (0 and 1) for THIS patient ----
        n_samples = len(y_tr)
        n_pos = int(y_tr.sum())
        n_neg = n_samples - n_pos

        # avoid division by zero
        n_pos = max(n_pos, 1)
        n_neg = max(n_neg, 1)

        # raw imbalance ratio: how many more negatives than positives
        raw_ratio = n_neg / n_pos

        # cap the ratio to avoid extreme weights (stabilises training)
        max_ratio = 10.0
        eff_ratio = min(raw_ratio, max_ratio)

        # keep class 0 as baseline, upweight class 1
        w0 = 1.0
        w1 = eff_ratio

        class_weights = {0: w0, 1: w1}

        print(
            f"Class weights for {pid}: "
            f"n_neg={n_neg}, n_pos={n_pos}, raw_ratio={raw_ratio:.2f}, "
            f"effective_ratio={eff_ratio:.2f} -> "
            f"w0={w0:.2f}, w1={w1:.2f}"
        )

        timesteps = X_tr.shape[1]
        features = X_tr.shape[2]

        # ----- Hyperparameter search space (stabilised) -----
        n_layers_options = [1, 2]                 # avoid too-deep LSTMs for imbalanced data
        lr_options = [1e-3, 3e-4]                 # slightly narrower, stable learning rates
        batch_size_options = [128, 256]
        epochs_options = [20, 35]
        batchnorm_options = [True, False]
        dense1_units_options = [32, 64]           # moderate dense capacity
        dense2_units_options = [0, 16, 32]        # 0 = no second dense layer
        lstm_dropout_options = [0.1, 0.2]         # base dropout
        lstm_rec_dropout_options = [0.0, 0.05]    # tiny recurrent dropout or none

        combos = list(
            itertools.product(
                n_layers_options,
                lr_options,
                batch_size_options,
                epochs_options,
                batchnorm_options,
                dense1_units_options,
                dense2_units_options,
                lstm_dropout_options,
                lstm_rec_dropout_options,
            )
        )
        random.shuffle(combos)
        combos = combos[:MAX_COMBOS]

        best_f1, best_info = -1.0, None

        for (
            n_layers,
            lr,
            bs,
            epochs_budget,
            bn,
            d1_units,
            d2_units,
            lstm_do,
            lstm_rdo,
        ) in combos:

            tf.keras.backend.clear_session()

            model = build_model(
                timesteps=timesteps,
                features=features,
                n_lstm_layers=n_layers,
                lr=lr,
                batchnorm=bn,
                dense1_units=d1_units,
                dense2_units=d2_units,
                lstm_dropout=lstm_do,
                lstm_recurrent_dropout=lstm_rdo,
            )

            es = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,              # a bit stricter
                restore_best_weights=True,
                verbose=0,
            )
            rlrop = callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=0,
            )

            hist = model.fit(
                X_tr,
                y_tr.astype(int),
                validation_data=(X_va, y_va.astype(int)),
                epochs=epochs_budget,
                batch_size=bs,
                class_weight=class_weights,
                callbacks=[es, rlrop],
                verbose=0,
            )

            y_prob = model.predict(X_va, verbose=0).ravel()
            best = tune_threshold_by_f1(y_va, y_prob)

            if best["f1"] > best_f1:
                best_f1 = best["f1"]
                best_info = {
                    "pid": pid,
                    "n_lstm_layers": n_layers,
                    "lr": lr,
                    "batch_size": bs,
                    "epochs_budget": epochs_budget,
                    "batchnorm": bn,
                    "dense1_units": d1_units,
                    "dense2_units": d2_units,
                    "lstm_dropout": lstm_do,
                    "lstm_recurrent_dropout": lstm_rdo,
                    "best_t": best["t"],
                    "best_f1": best["f1"],
                    "prec": best["prec"],
                    "rec": best["rec"],
                }

                curves_path = os.path.join(OUT_DIR, f"{pid}_curves.png")
                plot_curves(hist, curves_path)

        if best_info:
            results.append(best_info)
            print(
                f"→ Best F1 for {pid}: {best_info['best_f1']:.3f} "
                f"@ t={best_info['best_t']:.2f} "
                f"(P={best_info['prec']:.3f}, R={best_info['rec']:.3f})"
            )
        else:
            print(f"No valid model found for {pid}.")

    # ----- SAVE RESULTS -----
    if results:
        df_summary = pd.DataFrame(results)

        # 1) Save timestamped summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_summary_path = os.path.join(
            OUT_DIR, f"summary_{timestamp}.csv"
        )
        df_summary.to_csv(run_summary_path, index=False)
        print(f"\n💾 Saved RUN summary: {run_summary_path}")

        # 2) Append to global summary (summary_all_runs.csv)
        global_summary_path = os.path.join(
            OUT_DIR, "summary_all_runs.csv"
        )

        if os.path.exists(global_summary_path):
            old_df = pd.read_csv(global_summary_path)
            combined = pd.concat([old_df, df_summary], ignore_index=True)
        else:
            combined = df_summary.copy()

        combined.to_csv(global_summary_path, index=False)
        print(f"📚 Updated GLOBAL summary: {global_summary_path}")

    else:
        print("\n⚠️ No results to save (no patient produced a valid model).")
