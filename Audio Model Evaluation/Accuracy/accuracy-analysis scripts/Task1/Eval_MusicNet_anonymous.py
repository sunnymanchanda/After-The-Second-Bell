"""
Audio Model Evaluation Script - MusicNet dataset
===================================================

Same approach as the Vox/Mustard/DASED evaluation scripts, adapted for
MusicNet (a MUSIC/NOTE-EVENT dataset - answers are counts, Yes/No, or
note & instrument names like "C4", "violin", "acoustic grand piano").

  - Ground truth = MusicNet.csv (contains Sl_no, sample_name, bin,
    category, question, program_family, program, reference_depth,
    ground_truth, audio_length_sec).
  - Join key = (sample_name, category, question) - UNLIKE Vox/Mustard/
    DASED, this is *not* unique in MusicNet's ground truth: 721 rows
    (680 groups) share a key with at least one other row, and 458 of
    those groups even have a *different* ground_truth answer. A plain
    inner join on the key alone would either blow up into a many-to-many
    match or silently score a model's answer against the wrong row.
    Fix: an occurrence-rank ("first time we see this key", "second
    time", ...) computed independently in each file is appended to the
    join key. This was verified against DeSTA2.5's predictions file,
    where every one of the 22,441 rows lines up 1:1 in order with the
    ground truth - the rank trick reproduces that alignment safely for
    any model file, while still discarding genuinely extra/missing rows
    the way a normal inner join would.
  - Model files: column names are NOT guaranteed to match the ground
    truth's (e.g. DeSTA2.5's file uses 'samplename' and 'modelanswer'
    instead of 'sample_name' and 'predicted_answer'). Columns are
    normalized per-file before joining - see normalize_model_columns().
  - Open-source models: one folder per model containing a CSV (not
    always named exactly like the folder - falls back to the first csv
    found in the folder, same as the DASED script).
  - Closed-source model: MusicNet_gemini_flash3_8_predictions ONLY -
    covers only a subset of the dataset's questions; inner-join logic
    naturally restricts scoring to just that subset.
  - Extra questions in a model's file not present in ground truth -> DISCARDED.
  - Ground-truth questions a model didn't answer -> DISCARDED.
  - Answer comparison: lenient normalization (strip/lowercase, numeric-aware).
  - answer_type buckets (based on the GROUND TRUTH answer only):
        Yes/No, Numeric, Note/Instrument Label (named notes/instruments -
        "C4", "violin", "acoustic grand piano", etc.), Other (safety net
        for any NaN/unexpected ground truth value - not currently
        expected to occur). 'Other' rows are EXCLUDED from every accuracy
        computation, not marked True/False.

Outputs (in BASE_DIR/eval_outputs/):
  - merged_all_models.csv
  - accuracy_by_bin.csv
  - accuracy_by_category.csv
  - accuracy_by_answer_type.csv
  - accuracy_by_program_family.csv   (MusicNet-specific extra slice)
  - accuracy_overall.csv

Edit the CONFIG section below, then run: python evaluate_models_musicnet.py
"""

import os
import glob
import pandas as pd

# ============================== CONFIG ===================================

# TODO: point this at the folder that directly contains MusicNet.csv and
# one subfolder per model (mirrors the Datased layout you already used).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROUND_TRUTH_FILENAME = "MusicNet.csv"

OPEN_SOURCE_MODEL_FOLDERS = [
    "DeSTA2.5-Audio-Llama-3.1-8B",
    "gemma-3n-E2B-it",
    "gemma-3n-E4B-it",
    "Kimi-Audio-7B-Instruct_predictions",
    "MiMo-Audio-7B-Instruct_predictions",
    "omnivinci",
    "Voxtral-Small-24B-2507",
]

# NOTE: only ONE closed-source model for this dataset
CLOSED_SOURCE_MODEL_FOLDERS = [
    "MusicNet_gemini_flash3_8_predictions",
]

# Folders to never treat as models
EXCLUDE_FOLDERS = ["extra"]

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


def normalize_model_columns(df, model_name, csv_path):
    """Model prediction files aren't guaranteed to use the ground truth's
    exact column names (e.g. DeSTA2.5 uses 'samplename'/'modelanswer'
    instead of 'sample_name'/'predicted_answer'). Detect the right columns
    by a loose (case/underscore/space-insensitive) name match and rename
    them onto the standard names the rest of the script expects."""
    def norm(s):
        return s.lower().replace(" ", "").replace("_", "")

    lookup = {norm(c): c for c in df.columns}

    def find(*aliases):
        for a in aliases:
            key = norm(a)
            if key in lookup:
                return lookup[key]
        return None

    sample_col = find("sample_name", "samplename", "sample")
    category_col = find("category")
    question_col = find("question")
    pred_col = find("predicted_answer", "modelanswer", "model_answer", "prediction", "answer")

    missing = []
    if not sample_col:
        missing.append("sample_name/samplename")
    if not category_col:
        missing.append("category")
    if not question_col:
        missing.append("question")
    if not pred_col:
        missing.append("predicted_answer/modelanswer")
    if missing:
        raise ValueError(
            f"[{model_name}] couldn't find required column(s) {missing} in "
            f"{csv_path}. Found columns: {list(df.columns)}"
        )

    return df.rename(columns={
        sample_col: "sample_name",
        category_col: "category",
        question_col: "question",
        pred_col: "predicted_answer",
    })


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
    """Classify a ground-truth answer into Yes/No, Numeric, Note/Instrument
    Label, or Other. MusicNet's non-numeric non-yes/no answers are note
    names (e.g. "C4", "Eb4/D#4") or instrument names (e.g. "violin",
    "acoustic grand piano"), so they're grouped as 'Note/Instrument Label'.
    'Other' is a safety net for missing/unexpected ground truth (not
    expected to occur given the data inspected) and is excluded from
    scoring entirely."""
    if pd.isna(gt_answer):
        return "Other"
    s = str(gt_answer).strip().lower()
    if s == "":
        return "Other"
    if s in ("yes", "no"):
        return "Yes/No"
    try:
        float(s)
        return "Numeric"
    except ValueError:
        pass
    return "Note/Instrument Label"


def build_sample_bin_map(base_dir, all_model_folders, exclude_folders):
    """Fallback only: if some model file is missing 'bin' AND the row can't
    be matched to the ground truth's own bin for some reason, fill it from
    any other model file that has it for that sample_name. (Not usually
    needed since the ground truth already carries bin directly.)"""
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
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        try:
            df = normalize_model_columns(df, folder, csv_path)
        except ValueError:
            continue
        if "bin" not in df.columns:
            continue
        for _, row in df.dropna(subset=["bin"]).iterrows():
            sample_bin.setdefault(row["sample_name"], row["bin"])
    return sample_bin


def add_dup_rank(df, key_cols):
    """Occurrence rank within each key group, computed independently per
    file. Appending this to the join key turns a many-to-many join (caused
    by MusicNet's non-unique (sample_name, category, question) key) back
    into a safe 1:1 join: the Nth time a given key appears in the model
    file is matched to the Nth time it appears in the ground truth, in
    original row order. Genuinely extra/missing occurrences still fall out
    of the join naturally, same as a normal inner join would."""
    df = df.copy()
    df["_dup_rank"] = df.groupby(key_cols).cumcount()
    return df


def load_and_score_model(model_name, csv_path, gt_df, gt_join_cols, sample_bin_map):
    """Load one model's predictions, join with ground truth on
    (sample_name, category, question, _dup_rank), discard anything that
    doesn't match on both sides, score the remainder. 'bin' is taken from
    the GROUND TRUTH, not from the model file."""
    raw = pd.read_csv(csv_path)
    df = normalize_model_columns(raw, model_name, csv_path)

    missing_key_cols = [c for c in KEY_COLS if c not in df.columns]
    if missing_key_cols:
        raise ValueError(
            f"[{model_name}] missing required column(s) {missing_key_cols} "
            f"in {csv_path}. Found columns: {list(df.columns)}"
        )

    df = add_dup_rank(df, KEY_COLS)

    # INNER JOIN on key + occurrence rank -> automatically discards:
    #   - model rows whose (sample_name, category, question, rank) isn't in ground truth
    #     (extra Qs, or an Nth occurrence the model made up)
    #   - ground-truth rows this model didn't answer (missing Qs, or a
    #     duplicate-key occurrence the model skipped)
    # bin/answer/program_family/answer_type all come from ground truth (gt_df).
    merged = pd.merge(
        df[KEY_COLS + ["_dup_rank", "predicted_answer"]],
        gt_df[KEY_COLS + ["_dup_rank", "answer", "bin", "program_family"]],
        on=KEY_COLS + ["_dup_rank"],
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
        ["model_name", "sample_name", "category", "bin", "program_family", "answer_type",
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

    required = ["sample_name", "category", "question", "ground_truth", "bin"]
    missing = [c for c in required if c not in gt_df.columns]
    if missing:
        raise ValueError(f"{GROUND_TRUTH_FILENAME} is missing columns: {missing}")
    if "program_family" not in gt_df.columns:
        gt_df["program_family"] = pd.NA

    gt_df = gt_df.rename(columns={"ground_truth": "answer"})

    n_dup = gt_df.duplicated(subset=KEY_COLS).sum()
    if n_dup:
        print(
            f"NOTE: ground truth has {n_dup} rows sharing a "
            f"(sample_name, category, question) key with another row. "
            f"Using an occurrence-rank tiebreaker to join safely - see "
            f"add_dup_rank() / load_and_score_model()."
        )
    gt_df = add_dup_rank(gt_df, KEY_COLS)

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
        scored = load_and_score_model(folder, csv_path, gt_df, KEY_COLS, sample_bin_map)
        all_scored.append(scored)

    print("\nScoring closed-source model(s) (subset of questions):")
    for folder in CLOSED_SOURCE_MODEL_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        csv_path = find_model_csv(folder_path, folder)
        if not csv_path:
            print(f"  !! No CSV found for {folder}, skipping.")
            continue
        scored = load_and_score_model(folder, csv_path, gt_df, KEY_COLS, sample_bin_map)
        all_scored.append(scored)

    if not all_scored:
        raise RuntimeError("No models were scored. Check BASE_DIR and folder names.")

    merged_all = pd.concat(all_scored, ignore_index=True)

    bin_acc = accuracy_table(merged_all, "bin")
    cat_acc = accuracy_table(merged_all, "category")
    type_acc = accuracy_table(merged_all, "answer_type")
    program_family_acc = accuracy_table(merged_all, "program_family")
    overall_acc = overall_accuracy_table(merged_all)

    out_dir = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    merged_all.to_csv(os.path.join(out_dir, "merged_all_models.csv"), index=False)
    bin_acc.to_csv(os.path.join(out_dir, "accuracy_by_bin.csv"), index=False)
    cat_acc.to_csv(os.path.join(out_dir, "accuracy_by_category.csv"), index=False)
    type_acc.to_csv(os.path.join(out_dir, "accuracy_by_answer_type.csv"), index=False)
    program_family_acc.to_csv(os.path.join(out_dir, "accuracy_by_program_family.csv"), index=False)
    overall_acc.to_csv(os.path.join(out_dir, "accuracy_overall.csv"), index=False)

    print(f"\nDone. Outputs written to: {out_dir}")
    print("\nOverall accuracy per model:")
    print(overall_acc.to_string(index=False))


if __name__ == "__main__":
    main()