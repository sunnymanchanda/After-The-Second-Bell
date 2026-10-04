"""
Audio Model Evaluation Script
==============================

What this does
---------------
1. Loads the ground-truth file (Vox_dataset.csv).
2. Loads every model's prediction CSV (open-source models: one folder per
   model, containing a CSV of the same name; closed-source models:
   vox_gemini_flash3.8 and vox_gpt_audio1.5 which only cover a SUBSET of
   Vox_dataset's questions).
3. Matches each model's rows to the ground truth using the key
   (sample_name, category, question).
      - Rows in the model file that don't exist in Vox_dataset  -> DISCARDED
      - Rows in Vox_dataset that don't exist in the model file  -> DISCARDED
      - Only rows present in BOTH are scored (this naturally handles the
        closed-source "small subset" case too - whatever subset they have
        gets matched against Vox_dataset, everything else ignored).
4. Compares predicted_answer vs the ground-truth answer using a lenient
   normalization (strip whitespace, lowercase, numeric-aware -> "3" == "3.0").
5. Computes accuracy per BIN, per CATEGORY, and OVERALL, for every model.
6. Writes:
      - merged_all_models.csv      -> every scored row, every model, with
                                       ground truth, predicted answer, and
                                       True/False correctness
      - accuracy_by_bin.csv        -> model x bin accuracy table
      - accuracy_by_category.csv   -> model x category accuracy table
      - accuracy_overall.csv       -> one accuracy number per model

How to use
----------
1. Edit the CONFIG section below:
     - BASE_DIR                -> the folder that contains Vox_dataset.csv
                                    and all the model folders
     - VOX_DATASET_FILENAME    -> name of the ground truth csv
     - OPEN_SOURCE_MODEL_FOLDERS -> list the open-source model folder names
     - CLOSED_SOURCE_MODEL_FOLDERS -> list the closed-source model folder names
     - EXCLUDE_FOLDERS         -> folders to ignore (e.g. "Extra", "outputs")
2. Run:  python evaluate_models.py
3. Find the 4 output CSVs in BASE_DIR/eval_outputs/
"""

import os
import glob
import pandas as pd

# ============================== CONFIG ===================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOX_DATASET_FILENAME = "Vox_dataset.csv"

# Open-source models: each is a folder under BASE_DIR containing a CSV of
# the SAME NAME as the folder (e.g. DeSTA2.5-Audio-Llama-3.1-8B/DeSTA2.5-Audio-Llama-3.1-8B.csv)
OPEN_SOURCE_MODEL_FOLDERS = [
    "DeSTA2.5-Audio-Llama-3.1-8B",
    "gemma-3n-E2B-it",
    "gemma-3n-E4B-it",
    "Kimi-Audio-7B-Instruct",
    "MiMo-Audio-7B-Instruct",
    "omnivinci",
    "Voxtral-Small-24B-2507",
]

# Closed-source models: only cover a SMALL SUBSET of Vox_dataset's questions
CLOSED_SOURCE_MODEL_FOLDERS = [
    "vox_gemini_flash3.8",
    "vox_gpt_audio1.5",
]

# Folders to never treat as models
EXCLUDE_FOLDERS = ["Extra", "outputs"]

OUTPUT_DIR_NAME = "eval_outputs"

KEY_COLS = ["sample_name", "category", "question"]

# ============================== HELPERS ===================================

def find_model_csv(folder_path, folder_name):
    """Find the CSV inside a model folder. Prefers a csv named exactly like
    the folder, otherwise falls back to the first csv found."""
    exact = os.path.join(folder_path, folder_name + ".csv")
    if os.path.exists(exact):
        return exact
    candidates = glob.glob(os.path.join(folder_path, "*.csv"))
    if not candidates:
        return None
    return candidates[0]


def normalize_answer(ans):
    """Lenient normalization for comparing answers:
       - strips whitespace
       - lowercases
       - treats numeric answers equivalently regardless of formatting
         (e.g. "3" == "3.0" == " 3 ")
    """
    if pd.isna(ans):
        return None
    s = str(ans).strip().lower()
    if s == "":
        return None
    # try numeric normalization
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        pass
    return s


def is_correct(gt_answer, pred_answer):
    ngt = normalize_answer(gt_answer)
    npred = normalize_answer(pred_answer)
    if ngt is None or npred is None:
        return False
    return ngt == npred


def build_sample_bin_map(base_dir, all_model_folders, exclude_folders):
    """Some model files may be missing the 'bin' column (this can happen for
    the closed-source models, which only had a subset of questions run).
    Since 'bin' is purely a function of the audio sample's length, it is the
    same for a given sample_name across every model. We scan every available
    model CSV that DOES have a bin column and build a sample_name -> bin
    lookup so it can be filled in wherever it's missing."""
    sample_bin = {}
    for folder in all_model_folders:
        if folder in exclude_folders:
            continue
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        csv_path = find_model_csv(folder_path, folder)
        if not csv_path:
            continue
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in ["sample_name", "bin"])
        except Exception:
            continue
        if "bin" not in df.columns:
            continue
        for _, row in df.dropna(subset=["bin"]).iterrows():
            sample_bin.setdefault(row["sample_name"], row["bin"])
    return sample_bin


def load_and_score_model(model_name, csv_path, vox_df, sample_bin_map):
    """Load one model's predictions, join with ground truth on
    (sample_name, category, question), discard anything that doesn't match
    on both sides, score the remainder."""
    df = pd.read_csv(csv_path)

    missing_key_cols = [c for c in KEY_COLS if c not in df.columns]
    if missing_key_cols:
        raise ValueError(
            f"[{model_name}] missing required column(s) {missing_key_cols} "
            f"in {csv_path}. Found columns: {list(df.columns)}"
        )
    if "predicted_answer" not in df.columns:
        raise ValueError(
            f"[{model_name}] missing 'predicted_answer' column in {csv_path}. "
            f"Found columns: {list(df.columns)}"
        )

    # Fill bin if missing entirely, or fill individual missing bin values
    if "bin" not in df.columns:
        df["bin"] = df["sample_name"].map(sample_bin_map)
    else:
        blank = df["bin"].isna()
        if blank.any():
            df.loc[blank, "bin"] = df.loc[blank, "sample_name"].map(sample_bin_map)

    # INNER JOIN -> automatically discards:
    #   - model rows whose (sample_name, category, question) isn't in Vox_dataset (extra Qs)
    #   - vox rows this model didn't answer (missing Qs)
    merged = pd.merge(
        df[KEY_COLS + ["predicted_answer", "bin"]],
        vox_df[KEY_COLS + ["answer"]],
        on=KEY_COLS,
        how="inner",
    )

    merged["model_name"] = model_name
    merged["correct"] = merged.apply(
        lambda r: is_correct(r["answer"], r["predicted_answer"]), axis=1
    )
    merged = merged.rename(columns={"answer": "ground_truth_answer"})

    n_model_rows = len(df)
    n_vox_rows = len(vox_df)
    n_scored = len(merged)
    print(
        f"  {model_name}: model_rows={n_model_rows}, vox_rows={n_vox_rows}, "
        f"scored(matched)={n_scored}, "
        f"discarded_from_model={n_model_rows - n_scored}, "
        f"discarded_from_vox={n_vox_rows - n_scored}"
    )

    return merged[
        ["model_name", "sample_name", "category", "bin", "question",
         "ground_truth_answer", "predicted_answer", "correct"]
    ]


def accuracy_table(all_scored_df, group_col):
    grp = (
        all_scored_df.groupby(["model_name", group_col])["correct"]
        .agg(correct_count="sum", total_count="count")
        .reset_index()
    )
    grp["accuracy_%"] = (grp["correct_count"] / grp["total_count"] * 100).round(2)
    return grp


def overall_accuracy_table(all_scored_df):
    grp = (
        all_scored_df.groupby("model_name")["correct"]
        .agg(correct_count="sum", total_count="count")
        .reset_index()
    )
    grp["accuracy_%"] = (grp["correct_count"] / grp["total_count"] * 100).round(2)
    return grp.sort_values("accuracy_%", ascending=False)


# ============================== MAIN ===================================

def main():
    vox_path = os.path.join(BASE_DIR, VOX_DATASET_FILENAME)
    vox_df = pd.read_csv(vox_path)

    missing = [c for c in ["sample_name", "category", "question", "answer"] if c not in vox_df.columns]
    if missing:
        raise ValueError(f"Vox_dataset.csv is missing columns: {missing}")

    all_model_folders = OPEN_SOURCE_MODEL_FOLDERS + CLOSED_SOURCE_MODEL_FOLDERS
    print("Building sample_name -> bin lookup from available model files...")
    sample_bin_map = build_sample_bin_map(BASE_DIR, all_model_folders, EXCLUDE_FOLDERS)
    print(f"  bin lookup built for {len(sample_bin_map)} samples\n")

    all_scored = []

    print("Scoring open-source models:")
    for folder in OPEN_SOURCE_MODEL_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        csv_path = find_model_csv(folder_path, folder)
        if not csv_path:
            print(f"  !! No CSV found for {folder}, skipping.")
            continue
        scored = load_and_score_model(folder, csv_path, vox_df, sample_bin_map)
        all_scored.append(scored)

    print("\nScoring closed-source models (subset of questions):")
    for folder in CLOSED_SOURCE_MODEL_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        csv_path = find_model_csv(folder_path, folder)
        if not csv_path:
            print(f"  !! No CSV found for {folder}, skipping.")
            continue
        scored = load_and_score_model(folder, csv_path, vox_df, sample_bin_map)
        all_scored.append(scored)

    if not all_scored:
        raise RuntimeError("No models were scored. Check BASE_DIR and folder names.")

    merged_all = pd.concat(all_scored, ignore_index=True)

    bin_acc = accuracy_table(merged_all, "bin")
    cat_acc = accuracy_table(merged_all, "category")
    overall_acc = overall_accuracy_table(merged_all)

    out_dir = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    merged_all.to_csv(os.path.join(out_dir, "merged_all_models.csv"), index=False)
    bin_acc.to_csv(os.path.join(out_dir, "accuracy_by_bin.csv"), index=False)
    cat_acc.to_csv(os.path.join(out_dir, "accuracy_by_category.csv"), index=False)
    overall_acc.to_csv(os.path.join(out_dir, "accuracy_overall.csv"), index=False)

    print(f"\nDone. Outputs written to: {out_dir}")
    print("\nOverall accuracy per model:")
    print(overall_acc.to_string(index=False))


if __name__ == "__main__":
    main()