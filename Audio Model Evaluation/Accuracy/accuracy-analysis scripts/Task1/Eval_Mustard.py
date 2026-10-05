"""
Audio Model Evaluation Script - MUSTARD dataset
=================================================

Same approach as the Vox_dataset evaluation script, adapted for Mustard:

  - Ground truth = Mustard.csv (already contains sample_name, category,
    question, answer, source, audio_length_sec, bin, question_id - so unlike
    Vox_dataset, 'bin' does NOT need to be derived from model files; it is
    taken directly from Mustard.csv after the join).
  - Join key = (sample_name, category, question) - already unique in
    Mustard.csv (0 duplicates), same as Vox.
  - Open-source models: one folder per model containing a CSV of the same
    name.
  - Closed-source models (mustard_gemini_flash3.8, mustard_gpt_audio1.5):
    only cover a small subset of Mustard's questions - inner-join logic
    naturally restricts scoring to just that subset.
  - Extra questions in a model's file not present in Mustard.csv -> DISCARDED.
  - Mustard.csv questions a model didn't answer -> DISCARDED.
  - Answer comparison: lenient normalization (strip/lowercase, numeric-aware).
  - answer_type buckets (based on the GROUND TRUTH answer only):
        Yes/No, Numeric, Ordinal (distinct-speaker / before-after-first-last
        / "The person who says '...'" style answers), Other (safety net -
        e.g. "They speak the same number of times"). 'Other' rows are
        EXCLUDED from every accuracy computation, not marked True/False.

Outputs (in BASE_DIR/eval_outputs/):
  - merged_all_models.csv
  - accuracy_by_bin.csv
  - accuracy_by_category.csv
  - accuracy_by_answer_type.csv
  - accuracy_overall.csv

Edit the CONFIG section below, then run: python evaluate_models_mustard.py
"""

import os
import glob
import pandas as pd

# ============================== CONFIG ===================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROUND_TRUTH_FILENAME = "Mustard.csv"

OPEN_SOURCE_MODEL_FOLDERS = [
    "DeSTA2.5-Audio-Llama-3.1-8B",
    "gemma-3n-E2B-it",
    "gemma-3n-E4B-it",
    "Kimi-Audio-7B-Instruct",
    "MiMo-Audio-7B-Instruct",
    "OmniVinci",
    "Voxtral-Small-24B-2507",
]

CLOSED_SOURCE_MODEL_FOLDERS = [
    "mustard_gemini_flash3.8",
    "mustard_gpt_audio1.5",
]

# Folders to never treat as models (e.g. "Extra", any results/summary folders)
EXCLUDE_FOLDERS = ["Extra"]

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


def classify_answer_type(gt_answer):
    """Classify a ground-truth answer into Yes/No, Numeric, Ordinal, or Other.
    'Ordinal' covers any answer that identifies a specific speaker/position
    rather than a count or yes/no: "the second distinct speaker",
    "Before"/"After"/"First"/"Last", and 'The person who says "..."' style
    answers. Anything left over (e.g. "They speak the same number of times")
    falls into 'Other' and is excluded from scoring entirely."""
    if pd.isna(gt_answer):
        return "Other"
    s = str(gt_answer).strip().lower()
    if s in ("yes", "no"):
        return "Yes/No"
    try:
        float(s)
        return "Numeric"
    except ValueError:
        pass
    if (
        "distinct speaker" in s
        or s in ("before", "after", "first", "last")
        or s.startswith("the person who says")
    ):
        return "Ordinal"
    return "Other"


def build_sample_bin_map(base_dir, all_model_folders, exclude_folders):
    """Fallback only: if some model file is missing 'bin' AND the row can't
    be matched to Mustard.csv's own bin for some reason, fill it from any
    other model file that has it for that sample_name. (Not usually needed
    since Mustard.csv already carries bin directly.)"""
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


def load_and_score_model(model_name, csv_path, gt_df, sample_bin_map):
    """Load one model's predictions, join with ground truth on
    (sample_name, category, question), discard anything that doesn't match
    on both sides, score the remainder. 'bin' is taken from the GROUND
    TRUTH (Mustard.csv), not from the model file."""
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

    # INNER JOIN -> automatically discards:
    #   - model rows whose (sample_name, category, question) isn't in Mustard.csv (extra Qs)
    #   - ground-truth rows this model didn't answer (missing Qs)
    # bin/answer/source/answer_type all come from ground truth (gt_df), not the model file.
    merged = pd.merge(
        df[KEY_COLS + ["predicted_answer"]],
        gt_df[KEY_COLS + ["answer", "bin", "source"]],
        on=KEY_COLS,
        how="inner",
    )

    # Safety net: fill any missing bin from other model files (shouldn't be needed)
    if merged["bin"].isna().any():
        blank = merged["bin"].isna()
        merged.loc[blank, "bin"] = merged.loc[blank, "sample_name"].map(sample_bin_map)

    merged["model_name"] = model_name
    merged["answer_type"] = merged["answer"].apply(classify_answer_type)

    is_other = merged["answer_type"] == "Other"
    merged["correct"] = merged.apply(
        lambda r: is_correct(r["answer"], r["predicted_answer"]), axis=1
    ).astype(object)
    merged.loc[is_other, "correct"] = pd.NA
    merged["excluded_other"] = is_other

    merged = merged.rename(columns={"answer": "ground_truth_answer"})

    n_model_rows = len(df)
    n_gt_rows = len(gt_df)
    n_scored = len(merged)
    print(
        f"  {model_name}: model_rows={n_model_rows}, ground_truth_rows={n_gt_rows}, "
        f"matched={n_scored}, "
        f"discarded_from_model={n_model_rows - n_scored}, "
        f"discarded_from_ground_truth={n_gt_rows - n_scored}"
    )

    return merged[
        ["model_name", "sample_name", "category", "bin", "source", "answer_type",
         "question", "ground_truth_answer", "predicted_answer", "correct", "excluded_other"]
    ]


def accuracy_table(all_scored_df, group_col):
    """Excludes 'Other'-type rows (excluded_other=True) from both the
    numerator and denominator - they are not scored at all."""
    scored = all_scored_df[~all_scored_df["excluded_other"]]
    grp = (
        scored.groupby(["model_name", group_col])["correct"]
        .agg(correct_count="sum", total_count="count")
        .reset_index()
    )
    grp["accuracy_%"] = (grp["correct_count"] / grp["total_count"] * 100).round(2)
    return grp


def overall_accuracy_table(all_scored_df):
    """Excludes 'Other'-type rows (excluded_other=True) from both the
    numerator and denominator - they are not scored at all."""
    scored = all_scored_df[~all_scored_df["excluded_other"]]
    grp = (
        scored.groupby("model_name")["correct"]
        .agg(correct_count="sum", total_count="count")
        .reset_index()
    )
    grp["accuracy_%"] = (grp["correct_count"] / grp["total_count"] * 100).round(2)
    return grp.sort_values("accuracy_%", ascending=False)


# ============================== MAIN ===================================

def main():
    gt_path = os.path.join(BASE_DIR, GROUND_TRUTH_FILENAME)
    gt_df = pd.read_csv(gt_path)

    required = ["sample_name", "category", "question", "answer", "bin"]
    missing = [c for c in required if c not in gt_df.columns]
    if missing:
        raise ValueError(f"{GROUND_TRUTH_FILENAME} is missing columns: {missing}")
    if "source" not in gt_df.columns:
        gt_df["source"] = pd.NA

    all_model_folders = OPEN_SOURCE_MODEL_FOLDERS + CLOSED_SOURCE_MODEL_FOLDERS
    sample_bin_map = build_sample_bin_map(BASE_DIR, all_model_folders, EXCLUDE_FOLDERS)

    all_scored = []

    print("Scoring open-source models:")
    for folder in OPEN_SOURCE_MODEL_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        csv_path = find_model_csv(folder_path, folder)
        if not csv_path:
            print(f"  !! No CSV found for {folder}, skipping.")
            continue
        scored = load_and_score_model(folder, csv_path, gt_df, sample_bin_map)
        all_scored.append(scored)

    print("\nScoring closed-source models (subset of questions):")
    for folder in CLOSED_SOURCE_MODEL_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        csv_path = find_model_csv(folder_path, folder)
        if not csv_path:
            print(f"  !! No CSV found for {folder}, skipping.")
            continue
        scored = load_and_score_model(folder, csv_path, gt_df, sample_bin_map)
        all_scored.append(scored)

    if not all_scored:
        raise RuntimeError("No models were scored. Check BASE_DIR and folder names.")

    merged_all = pd.concat(all_scored, ignore_index=True)

    bin_acc = accuracy_table(merged_all, "bin")
    cat_acc = accuracy_table(merged_all, "category")
    type_acc = accuracy_table(merged_all, "answer_type")
    overall_acc = overall_accuracy_table(merged_all)

    out_dir = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    merged_all.to_csv(os.path.join(out_dir, "merged_all_models.csv"), index=False)
    bin_acc.to_csv(os.path.join(out_dir, "accuracy_by_bin.csv"), index=False)
    cat_acc.to_csv(os.path.join(out_dir, "accuracy_by_category.csv"), index=False)
    type_acc.to_csv(os.path.join(out_dir, "accuracy_by_answer_type.csv"), index=False)
    overall_acc.to_csv(os.path.join(out_dir, "accuracy_overall.csv"), index=False)

    print(f"\nDone. Outputs written to: {out_dir}")
    print("\nOverall accuracy per model:")
    print(overall_acc.to_string(index=False))


if __name__ == "__main__":
    main()