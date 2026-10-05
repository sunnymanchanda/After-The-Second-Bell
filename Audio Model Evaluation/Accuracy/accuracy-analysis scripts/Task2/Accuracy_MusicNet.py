"""
AQA Model Accuracy Report - MusicNet dataset (FIXED + closed-source added)
============================================================================

This extends the corrected aqa_accuracy_report.py by adding the
closed-source model (gpt-audio-1.5), mirroring the open/closed-source
split used in the DASED evaluation script:

  - OPEN_SOURCE_MODEL_FILES: the 6 open-source models, scored first.
  - CLOSED_SOURCE_MODEL_FILES: gpt-audio-1.5 ONLY, scored second, in its
    own labeled console section. It naturally only covers a subset of
    Sl_no's - the inner join restricts scoring to just that subset, same
    as every other model.
  - A new `model_type` column ("Open-Source" / "Closed-Source") is added
    to every output so you can filter/compare the two groups later.

Everything else (bug fixes 1-3 from the original FIXED version) is
unchanged:
  1. Yes/No is never conflated with numeric 1/0 - answer_type is decided
     from the ground truth BEFORE normalization, and Yes/No vs Numeric
     comparisons can never cross-match.
  2. Mismatched rows (either side) are DISCARDED via inner join, not
     scored as wrong.
  3. Duplicate Sl_no rows in a model's file are dropped (first kept) with
     a warning, instead of crashing.

What this produces
--------------------
  1. Overall accuracy per model (open + closed source together, sorted)
  2. Accuracy per model x subquestion_type
  3. Accuracy per model x answer_type (Yes/No vs Numeric)
  4. A combined long-format CSV with a `correct` flag for every scored row
  5. A pivoted summary CSV (subquestion_type as rows, model as columns)

Usage:
    python aqa_accuracy_report_musicnet.py

Edit the CONFIG section below to point at your actual file paths.
"""

import re
import pandas as pd
from pathlib import Path

# ----------------------------- CONFIG ----------------------------------

BASE_DIR = Path(__file__).resolve().parent

GROUND_TRUTH_PATH = BASE_DIR / "MusicNet.csv"

# Open-source models -> path to that model's prediction csv
OPEN_SOURCE_MODEL_FILES = {
    "gemma-3n-E2B-it":    "gemma-3n-E2B-it/gemma-3n-E2B-it_MusicNet_Task2.csv",
    "gemma-3n-E4B-it":    "gemma-3n-E4B-it/gemma-3n-E4B-it_MusicNet_Task2.csv",
    "OmniVinci":          "OmniVinci/OmniVinci_MusicNet_Task2.csv",
    "Qwen2.5-Omni-3B":    "Qwen2.5-Omni-3B/Qwen2.5-Omni-3B_MusicNet_Task2.csv",
    "R1_aqa":             "R1_aqa/R1_aqa.csv",
    "Voxtral-Small-24B":  "Voxtral-Small-24B-2507/Voxtral-Small-24B-2507_MusicNet_Task2.csv",
}

# NOTE: only ONE closed-source model for this dataset, added here.
# !!! CHECK THIS PATH !!! - guessed to follow the same
# "<folder>/<folder>_MusicNet_Task2.csv" pattern as the open-source models.
# Your folder is named "gpt-audio-1.5_MusicNet" - update the filename below
# if the actual CSV inside it is named differently.
CLOSED_SOURCE_MODEL_FILES = {
    "gpt-audio-1.5": "gpt-audio-1.5_MusicNet/gpt-audio-1.5_MusicNet.csv",
}

OUTPUT_DETAIL_CSV = BASE_DIR / "aqa_detailed_results.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "aqa_accuracy_summary.csv"
OUTPUT_ANSWER_TYPE_CSV = BASE_DIR / "aqa_accuracy_by_answer_type.csv"
OUTPUT_OVERALL_CSV = BASE_DIR / "aqa_overall_accuracy.csv"

# -------------------------------------------------------------------------

# NOTE: "yes"/"no" were REMOVED from this map on purpose (see Bug 1 above).
# This map is now ONLY used for Numeric-type ground truth rows, so a
# number-word like "three" can still match a digit "3", without any risk
# of "yes"/"no" ever being treated as 1/0.
WORD2NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "none": 0, "nothing": 0,
}


def classify_answer_type(ground_truth) -> str:
    """Classify a ground-truth answer as Yes/No, Numeric, or Other (safety
    net for anything unexpected). This MUST be decided from the ground
    truth alone, and decided BEFORE any normalization, so a model's raw
    predicted text never influences which comparison rules apply to it."""
    if pd.isna(ground_truth):
        return "Other"
    s = str(ground_truth).strip().lower()
    if s in ("yes", "no"):
        return "Yes/No"
    try:
        float(s)
        return "Numeric"
    except ValueError:
        pass
    return "Other"


def normalize_for_type(val, answer_type: str) -> str:
    """Normalize a value for comparison, using rules specific to its
    answer_type so Yes/No and Numeric answers can NEVER cross-match."""
    if pd.isna(val):
        return ""
    s = str(val).strip().lower()
    s = re.sub(r"[.\!\?]+$", "", s)          # trailing punctuation
    s = re.sub(r"\s+", " ", s).strip()

    if answer_type == "Yes/No":
        # Only ever compare within {"yes", "no"} - never touches numbers.
        if s in ("yes", "no"):
            return s
        return f"__unrecognized_yesno__:{s}"

    if answer_type == "Numeric":
        if s in WORD2NUM:
            return str(WORD2NUM[s])
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
            return str(f)
        except ValueError:
            return f"__unrecognized_numeric__:{s}"

    # "Other" rows are excluded from scoring entirely (handled by caller),
    # so their normalized value is never actually compared.
    return s


def find_col(df: pd.DataFrame, *candidates: str) -> str:
    """Case-insensitive column lookup; returns the actual column name."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def load_ground_truth(path: str) -> pd.DataFrame:
    gt = pd.read_csv(path)
    sl_col = find_col(gt, "Sl_no", "Sl_No")
    case_col = find_col(gt, "case_id")
    sub_col = find_col(gt, "subquestion_type")
    ans_col = find_col(gt, "ground_truth")

    gt = gt.rename(columns={
        sl_col: "Sl_no",
        case_col: "case_id",
        sub_col: "subquestion_type",
        ans_col: "ground_truth",
    })
    gt["Sl_no"] = gt["Sl_no"].astype(str).str.strip()
    gt["answer_type"] = gt["ground_truth"].apply(classify_answer_type)
    gt["gt_norm"] = gt.apply(
        lambda r: normalize_for_type(r["ground_truth"], r["answer_type"]), axis=1
    )
    return gt[["Sl_no", "case_id", "subquestion_type", "ground_truth", "answer_type", "gt_norm"]]


def load_model_predictions(path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    sl_col = find_col(df, "Sl_no", "Sl_No")
    ans_col = find_col(df, "predicted_answer")

    df = df.rename(columns={sl_col: "Sl_no", ans_col: "predicted_answer"})
    df["Sl_no"] = df["Sl_no"].astype(str).str.strip()

    dup_count = df["Sl_no"].duplicated().sum()
    if dup_count:
        print(f"  [WARN] {model_name}: {dup_count} duplicate Sl_no rows found - keeping first only")
        df = df.drop_duplicates(subset="Sl_no", keep="first")

    return df[["Sl_no", "predicted_answer"]]


def score_model(model_name, path, model_type, gt, all_rows, overall_rows):
    """Load one model's predictions, inner-join with ground truth, score,
    and append results into the shared accumulator lists."""
    if not Path(path).exists():
        print(f"[WARN] Skipping {model_name}: file not found at {path}")
        return

    preds = load_model_predictions(path, model_name)

    # INNER JOIN -> discards BOTH:
    #   - ground-truth rows this model didn't answer (missing predictions)
    #   - prediction rows whose Sl_no isn't in the ground truth (extras)
    # Neither is scored, neither shows up anywhere downstream.
    merged = gt.merge(preds, on="Sl_no", how="inner")

    n_gt = len(gt)
    n_model = len(preds)
    n_scored_candidates = len(merged)
    print(
        f"  {model_name}: ground_truth_rows={n_gt}, model_rows={n_model}, "
        f"matched={n_scored_candidates}, "
        f"discarded_from_ground_truth={n_gt - n_scored_candidates}, "
        f"discarded_from_model={n_model - n_scored_candidates}"
    )

    merged["pred_norm"] = merged.apply(
        lambda r: normalize_for_type(r["predicted_answer"], r["answer_type"]), axis=1
    )

    # "Other"-type ground truth rows (safety net, not expected to occur
    # for MusicNet's Yes/No + Numeric answers) are excluded from scoring
    # entirely - not marked True/False.
    is_other = merged["answer_type"] == "Other"
    merged["correct"] = (merged["gt_norm"] == merged["pred_norm"]).astype(object)
    merged.loc[is_other, "correct"] = pd.NA
    merged["excluded_other"] = is_other
    merged["model"] = model_name
    merged["model_type"] = model_type

    if is_other.sum():
        print(f"    [INFO] {model_name}: {is_other.sum()} rows had an unrecognized/Other "
              f"ground-truth format and were excluded from accuracy")

    all_rows.append(merged[[
        "Sl_no", "case_id", "subquestion_type", "answer_type", "model", "model_type",
        "ground_truth", "predicted_answer", "correct", "excluded_other"
    ]])

    scored = merged[~merged["excluded_other"]]
    overall_acc = scored["correct"].mean() * 100 if len(scored) else float("nan")
    overall_rows.append({
        "model": model_name,
        "model_type": model_type,
        "n_questions": len(scored),
        "n_correct": int(scored["correct"].sum()),
        "accuracy_pct": round(overall_acc, 2),
    })


def main():
    gt = load_ground_truth(GROUND_TRUTH_PATH)

    all_rows = []          # long-format detail rows across all models (scored only)
    overall_rows = []      # one row per model: overall accuracy

    print("Scoring open-source models:")
    for model_name, path in OPEN_SOURCE_MODEL_FILES.items():
        score_model(model_name, BASE_DIR / path, "Open-Source", gt, all_rows, overall_rows)

    print("\nScoring closed-source model(s) (subset of questions):")
    for model_name, path in CLOSED_SOURCE_MODEL_FILES.items():
        score_model(model_name, BASE_DIR / path, "Closed-Source", gt, all_rows, overall_rows)

    if not all_rows:
        print("No model files could be loaded. Check OPEN_SOURCE_MODEL_FILES / "
              "CLOSED_SOURCE_MODEL_FILES paths.")
        return

    detail = pd.concat(all_rows, ignore_index=True)
    detail.to_csv(OUTPUT_DETAIL_CSV, index=False)

    scored_detail = detail[~detail["excluded_other"]]

    # Accuracy per model x subquestion_type
    by_subq = (
        scored_detail.groupby(["model", "subquestion_type"])["correct"]
        .agg(n="count", n_correct="sum")
        .reset_index()
    )
    by_subq["accuracy_pct"] = (by_subq["n_correct"] / by_subq["n"] * 100).round(2)

    # Pivot: rows = subquestion_type, columns = model, values = accuracy_pct
    pivot = by_subq.pivot(index="subquestion_type", columns="model", values="accuracy_pct")
    pivot.to_csv(OUTPUT_SUMMARY_CSV)

    # Accuracy per model x answer_type (Yes/No vs Numeric)
    by_type = (
        scored_detail.groupby(["model", "answer_type"])["correct"]
        .agg(n="count", n_correct="sum")
        .reset_index()
    )
    by_type["accuracy_pct"] = (by_type["n_correct"] / by_type["n"] * 100).round(2)
    by_type.to_csv(OUTPUT_ANSWER_TYPE_CSV, index=False)

    overall_df = pd.DataFrame(overall_rows).sort_values("accuracy_pct", ascending=False)
    overall_df.to_csv(OUTPUT_OVERALL_CSV, index=False)

    print("\n=== Overall accuracy by model (open + closed source) ===")
    print(overall_df.to_string(index=False))

    print("\n=== Accuracy by model x subquestion_type ===")
    print(pivot.to_string())

    print("\n=== Accuracy by model x answer_type ===")
    print(by_type.to_string(index=False))

    print(
        f"\nSaved:\n  {OUTPUT_DETAIL_CSV}\n  {OUTPUT_SUMMARY_CSV}\n"
        f"  {OUTPUT_ANSWER_TYPE_CSV}\n  {OUTPUT_OVERALL_CSV}"
    )


if __name__ == "__main__":
    main()