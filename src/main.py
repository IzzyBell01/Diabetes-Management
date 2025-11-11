from utils.load_data import load_patient_data
from utils.summary import build_summary
from preprocessing.preprocess import preprocess_patient
from windowing.window_size import window_size
from models.lstm_gridsearch import run_personalized_lstm_search


# ---------- PIPELINE 1: Single-patient pipeline ----------
def run_single_patient_pipeline(data, patient_key):
    print(f"\n=== Running pipeline for patient {patient_key} ===")
    df = data[patient_key]
    segments, summary = preprocess_patient(
        df, time_col="5minute_intervals_timestamp", cbg_col="cbg",
        normalize=True, long_gap_thresh=120
    )
    X, y, meta = window_size(segments, window_minutes=120, stride_minutes=45, sample_every=5)
    print(f" Total windows: {len(X)}, Positive (meal) windows: {y.sum()}")
    model_input_data = {patient_key: {"X": X, "y": y, "meta": meta}}
    run_personalized_lstm_search(model_input_data)


# ---------- PIPELINE 2: Loop over all patients (individual training) ----------
def run_all_patients_individual(data):
    print("\n=== Running pipeline for ALL patients (individual models) ===")
    for patient_key in data.keys():
        if "training" not in patient_key:
            continue  # skip test sets
        try:
            run_single_patient_pipeline(data, patient_key)
        except Exception as e:
            print(f"⚠️ Skipping {patient_key} due to error: {e}")


# ---------- PIPELINE 3: Generalized model ----------
def run_generalized_model(data):
    print("\n=== Running GENERALIZED model across all patients ===")
    all_segments = []
    for patient_key, df in data.items():
        if "training" not in patient_key:
            continue
        try:
            segments, _ = preprocess_patient(df, time_col="5minute_intervals_timestamp", cbg_col="cbg")
            all_segments.extend(segments)
        except Exception as e:
            print(f"⚠️ Skipping {patient_key}: {e}")
    # Combine all patients into one big dataset
    X, y, meta = window_size(all_segments, window_minutes=120, stride_minutes=45, sample_every=5)
    print(f"🧩 Combined dataset: {len(X)} windows ({y.sum()} positive).")
    model_input_data = {"Generalized_Model": {"X": X, "y": y, "meta": meta}}
    run_personalized_lstm_search(model_input_data)


# ---------- MAIN ----------
if __name__ == "__main__":
    base_dir = "/Users/isabellemueller/BME unibern/Diabetes Management/Ohio Data"
    data = load_patient_data(base_dir)
    summary_df = build_summary(data)

    # Choose which pipeline to run:
    mode = "all"  # options: "single", "all", "generalized"

    if mode == "single":
        run_single_patient_pipeline(data, "588-ws-training_processed") #<--- select patient

    elif mode == "all":
        run_all_patients_individual(data)

    elif mode == "generalized":
        run_generalized_model(data)