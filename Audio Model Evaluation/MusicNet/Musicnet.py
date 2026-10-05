"""
Run Voxtral-Small-24B-2507 over music analysis questions and write predicted answers
to a CSV.

Input CSV columns expected: Slno, sample_name, bin, category, question, ...
Output CSV columns: Slno, samplename, bin, question, modelanswer

Audio files are organized by folder: {AUDIO_BASE_DIR}/{bin}/{sample_name}.wav

Extras (kept from the Voxtral script):
  * --gpu flag to control CUDA_VISIBLE_DEVICES (e.g. "0" or "0,1")
  * Checkpointing: if the output CSV already exists with partial results, rows
    whose Slno is already present are skipped and generation resumes.
"""

import os
import csv
import argparse
import traceback

# --------------------------- GPU ALLOCATION -------------------------------
# Must happen BEFORE torch/transformers are imported, since CUDA_VISIBLE_DEVICES
# is read once at process start.

def _parse_gpu_arg():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", default="7")
    known, _ = p.parse_known_args()
    return known.gpu


_gpu_arg = _parse_gpu_arg()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg
print(f"[GPU] CUDA_VISIBLE_DEVICES set to: {_gpu_arg}")

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

# ----------------------------- CONFIG ------------------------------------

MODEL_PATH = "path/to/Voxtral-Small-24B-2507"
QUESTIONS_CSV = "questions.csv"
AUDIO_BASE_DIR = "MusicNet"
AUDIO_EXT = ".wav"                                                        # <-- CHANGE if different

# Output file name/location = same convention as Audio-Flamingo-3 / Qwen2-Audio:
# .../MusicNet/<MODEL_NAME>/<MODEL_NAME>.csv
MODEL_NAME = os.path.basename(os.path.normpath(MODEL_PATH))
OUTPUT_CSV = f"{MODEL_NAME}/{MODEL_NAME}.csv"   # <-- CHANGE if different

MAX_NEW_TOKENS = 64

PROMPT_TEMPLATE = """TASK

Carefully analyze the audio recording and answer the question using only
information that can be determined directly from listening to the audio.

IMPORTANT

* The audio recording is your ONLY source of information.
* Analyze the entire recording thoroughly, paying attention to all musical notes, pitch events, instruments, and their temporal placement.
* Do not use external music theory knowledge, music databases, or assumptions outside what is directly heard in the audio.

DEFINITIONS

1. Musical Note Event
A distinct audible instance of a pitch played by a specific instrument from the moment it begins (start) to the moment it stops sounding (end).
* Every distinctly articulated note heard in the recording counts as one occurrence, even if played consecutively or at the same time as other notes.
* Pitch names are expressed using standard pitch spelling and octave notation (e.g., C5, E5, B2, Eb4/D#4).

2. Instrument Occurrence
An instance where a specific instrument (e.g., piano, violin, cello, flute) plays a musical note event.

3. Temporal Relations
* BEFORE: Note A occurs strictly before Note B if Note A completely stops sounding at or before Note B begins.
* AFTER: Note A occurs strictly after Note B if Note A begins at or after Note B completely stops sounding.
* OVERLAP: Note A and Note B overlap if they are sounding at the same time for any duration. Overlapping notes do NOT count as strictly before or strictly after one another.
* BETWEEN: Note X is between Note A and Note B if Note X begins after Note A has completely stopped sounding AND Note X stops sounding before Note B begins.

4. Event Ordering
When questions refer to specific occurrences (e.g., "the first C5 note", "the second violin note"), occurrences are ordered strictly chronologically by when they start in the audio.

ANSWERING RULES

* Determine the answer solely from listening to the audio recording and interpreting the explicit wording of the question.
* Follow all definitions above exactly.
* Answer ONLY what the question asks.
* Do not provide explanations, step-by-step reasoning, or commentary.
* Do not output any additional text beyond the required answer.
* Infer the required answer format directly from the question wording.

OUTPUT FORMAT

Infer the exact answer format required by the question:

If ANSWER_TYPE is YES_NO:
Output exactly one of:
Yes
No

If ANSWER_TYPE is INT:
Output only one non-negative integer as digits.
Examples:
0
1
2
15

Do not output words (e.g., write "2", not "two"), units, punctuation, or explanations.

If ANSWER_TYPE is ORDINAL:
Output only the ordinal in uppercase.
Examples:
FIRST
SECOND
THIRD
LAST

If ANSWER_TYPE is PITCH_OR_INSTRUMENT:
Output only the pitch name or instrument name as requested by the question.
Examples:
C5
piano
violin

If ANSWER_TYPE is TEMPORAL_RELATION:
Output exactly one of:
BEFORE
AFTER

COMMON MISTAKES TO AVOID

Question: How many times is note E5 played after the first piano note?
WRONG: The note E5 is played 3 times.
RIGHT: 3

Question: Is C5 played more often than F5?
WRONG: Yes, C5 is played more often.
RIGHT: Yes

Question: Which instrument plays the first note in the recording?
WRONG: The piano plays the first note.
RIGHT: piano

FINAL OUTPUT RULE

Your entire response must contain exactly one answer and nothing else.

Do not output:
* explanations
* reasoning
* full sentences
* prefixes such as "Answer:"
* suffixes
* punctuation
* markdown
* quotation marks

QUESTION:

{question}

ANSWER:
"""

# ---------------------------------------------------------------------------


def load_model(model_path):
    print(f"Loading model from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")
    print(f"Model is on device(s): {model.hf_device_map if hasattr(model, 'hf_device_map') else model.device}")
    return model, processor


def resolve_audio_path(bin_folder, sample_name):
    """
    Build the audio file path from bin folder and sample_name.
    Path format: {AUDIO_BASE_DIR}/{bin}/{sample_name}.wav

    Handles edge cases where sample_name might already include extension
    or unwanted extensions from CSV.
    """
    base = sample_name

    # Strip any stray annotation extensions that might be in sample_name
    for stray_ext in (".rttm", ".json", ".csv", ".txt", ".wav"):
        if base.lower().endswith(stray_ext):
            base = base[: -len(stray_ext)]
            break

    # Add .wav extension if not already present
    name = base if base.lower().endswith(AUDIO_EXT) else base + AUDIO_EXT

    audio_path = os.path.join(AUDIO_BASE_DIR, bin_folder, name)
    return audio_path


def run_inference(model, processor, audio_path, question_text):
    """
    Run inference on audio with the given question.
    """
    prompt_text = PROMPT_TEMPLATE.format(question=question_text)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(conversation)
    inputs = inputs.to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    # Remove prompt tokens from output
    generated_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    return output_text.strip()


def load_completed_sl_nos(output_csv):
    """Read rows already present in output_csv (from a previous, possibly
    interrupted run) so we can resume instead of starting over."""
    completed = set()
    if not os.path.exists(output_csv):
        return completed
    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                slno = row.get("Slno")
                if slno is not None:
                    completed.add(slno)
    except Exception as e:
        print(f" !! could not read existing output for checkpointing: {e}")
    return completed


def main():
    global AUDIO_BASE_DIR

    parser = argparse.ArgumentParser(
        description="Run Voxtral-Small-24B-2507 music analysis inference on audio dataset"
    )
    parser.add_argument("--questions_csv", default=QUESTIONS_CSV,
                        help="Path to input CSV with questions")
    parser.add_argument("--audio_base_dir", default=AUDIO_BASE_DIR,
                        help="Base directory containing folder-organized audio files")
    parser.add_argument("--output_csv", default=OUTPUT_CSV,
                        help="Path to output CSV for predictions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N rows (for testing)")
    parser.add_argument(
        "--gpu",
        default="0",
        help='GPU allocation, e.g. "0" for a single GPU or "0,1" for multiple. '
             "Sets CUDA_VISIBLE_DEVICES; must match what was used at import time.",
    )
    args = parser.parse_args()

    AUDIO_BASE_DIR = args.audio_base_dir

    # Make sure the output folder exists
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model, processor = load_model(MODEL_PATH)

    # Read input CSV
    print(f"Reading questions from {args.questions_csv}...")
    with open(args.questions_csv, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    if args.limit:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} questions.")

    # Output fieldnames
    fieldnames = ["Slno", "samplename", "bin", "question", "modelanswer"]

    # ---- CHECKPOINT: figure out what's already been done ----
    completed_sl_nos = load_completed_sl_nos(args.output_csv)
    resuming = len(completed_sl_nos) > 0
    if resuming:
        print(f"Resuming: {len(completed_sl_nos)} rows already completed in {args.output_csv}, will skip them.")

    remaining_rows = [r for r in rows if r.get("Slno") not in completed_sl_nos]
    print(f"{len(remaining_rows)} of {len(rows)} rows left to process.")

    # Append if resuming an existing file, else write fresh with header
    file_mode = "a" if resuming else "w"

    with open(args.output_csv, file_mode, newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if not resuming:
            writer.writeheader()

        for i, row in enumerate(remaining_rows):
            slno = row.get("Slno", "")
            sample_name = row.get("sample_name", "")
            bin_folder = row.get("bin", "")
            question_text = row.get("question", "")

            audio_path = resolve_audio_path(bin_folder, sample_name)

            print(f"[{i+1}/{len(remaining_rows)}] Slno={slno} sample={sample_name} bin={bin_folder}")

            # Check if audio file exists
            if not os.path.exists(audio_path):
                print(f" !! audio file not found: {audio_path}")
                answer = "ERROR_AUDIO_NOT_FOUND"
            else:
                try:
                    answer = run_inference(model, processor, audio_path, question_text)
                    print(f" Answer: {answer}")
                except Exception as e:
                    print(f" !! inference failed: {e}")
                    traceback.print_exc()
                    answer = "ERROR_INFERENCE_FAILED"

            # Write output row
            writer.writerow(
                {
                    "Slno": slno,
                    "samplename": sample_name,
                    "bin": bin_folder,
                    "question": question_text,
                    "modelanswer": answer,
                }
            )
            f_out.flush()

    print(f"\nDone. Predictions written to {args.output_csv}")


if __name__ == "__main__":
    main()