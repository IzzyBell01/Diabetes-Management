import matplotlib.pyplot as plt

def plot_meal_counts(summary_df):
    """Bar chart of annotated meals per patient, split by train/test."""
    # Clean up split column
    summary_df["split"] = summary_df["split"].fillna("").str.lower().str.strip()

    # Robust filtering for 'train' and 'test'
    train_df = summary_df[summary_df["split"].str.contains("train", na=False)]
    test_df  = summary_df[summary_df["split"].str.contains("test", na=False)]

    # If both are empty, warn user
    if train_df.empty and test_df.empty:
        print("⚠️ Warning: No train/test split detected in summary_df['split'] column.")
        print("Unique values found:", summary_df["split"].unique())
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # (a) TRAIN
    if not train_df.empty:
        axes[0].bar(train_df["dataset"] + "_" + train_df["patient"],
                    train_df["positive_meals"], color='mediumseagreen', edgecolor='black')
        axes[0].set_title("Train Set: Annotated Meals per Patient")
        axes[0].set_xlabel("Patient")
        axes[0].set_ylabel("Number of CHO meals (>0 g)")
        axes[0].tick_params(axis='x', rotation=90)
    else:
        axes[0].text(0.5, 0.5, "No train data found", ha="center", va="center")

    # (b) TEST
    if not test_df.empty:
        axes[1].bar(test_df["dataset"] + "_" + test_df["patient"],
                    test_df["positive_meals"], color='lightcoral', edgecolor='black')
        axes[1].set_title("Test Set: Annotated Meals per Patient")
        axes[1].set_xlabel("Patient")
        axes[1].tick_params(axis='x', rotation=90)
    else:
        axes[1].text(0.5, 0.5, "No test data found", ha="center", va="center")

    plt.suptitle("Count of Annotated Meals per Patient (Ohio2018 + Ohio2020)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

'''
def plot_meal_distributions(patient_data):
    """Plots CHO histograms and CBG time series for each patient."""
    for key, df in patient_data.items():
        meals = df['carbInput'].dropna()
        positive_meals = meals[meals > 0]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(positive_meals, bins=15, edgecolor='black', color='coral')
        plt.title(f"{key} – CHO Meal Distribution")
        plt.xlabel("CHO (grams)")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        plt.plot(df['5minute_intervals_timestamp'], df['cbg'], color='navy')
        plt.title(f"{key} – CBG over Time")
        plt.xlabel("5-minute interval timestamp")
        plt.ylabel("CBG (mg/dL)")

        plt.tight_layout()
        plt.show()

'''