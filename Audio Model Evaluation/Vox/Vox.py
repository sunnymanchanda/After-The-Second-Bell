"""
Run Voxtral-Small-24B-2507 over the AQA question set and write predicted answers
to a new CSV (ground truth is never read from Questions.csv, and is not
written to the output).

Input CSV columns expected: Sl_no, sample_name, question, ground_truth, audio_length_sec
Output CSV columns:         Sl_no, sample_name, question, predicted_answer

Adds:
  * --gpu flag to control CUDA_VISIBLE_DEVICES (e.g. "0" or "0,1")
  * Checkpointing: if OUTPUT_CSV already exists with partial results, rows
    whose Sl_no is already present are skipped and generation resumes from
    where it left off instead of starting over.
"""

import os
import csv
import argparse
import traceback

# --------------------------- GPU ALLOCATION -------------------------------
# Must happen BEFORE torch/transformers are imported, since CUDA_VISIBLE_DEVICES
# is read once at process start and controls which physical GPUs this process
# can see at all.

def _parse_gpu_arg():
    """Peek at --gpu early (argparse proper runs again below, this is just
    to set the env var before importing torch)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", default="0")
    known, _ = p.parse_known_args()
    return known.gpu


_gpu_arg = _parse_gpu_arg()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg
print(f"[GPU] CUDA_VISIBLE_DEVICES set to: {_gpu_arg}")

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

# ----------------------------- CONFIG ------------------------------------

MODEL_PATH = "path/to/Voxtral-Small-24B-2507"
QUESTIONS_CSV = "Questions.csv"
AUDIO_DIR = "Sample"   # <-- CHANGE if different
AUDIO_EXT = ".wav"                                     # <-- CHANGE if different

# Output file name = model name (derived from MODEL_PATH's folder name)
MODEL_NAME = os.path.basename(os.path.normpath(MODEL_PATH))
OUTPUT_CSV = f"{MODEL_NAME}.csv"

MAX_NEW_TOKENS = 32

PROMPT_TEMPLATE = """TASK

Carefully analyze the audio recording and answer the question using only
information that can be determined from the recording.

IMPORTANT

* The audio recording is the only source of information available to you.
* You do NOT have access to RTTM files, JSON files, CSV files, timestamps,
  speaker IDs, speaker annotations, or any other annotation metadata.
* Do not refer to, assume, or use information from such metadata.
* Do not use external knowledge or information outside the provided audio.
* Analyze the full recording and pay attention to all audio relevant to the
  question.

DEFINITIONS

1. Distinct Speaker

A distinct speaker is a different individual voice heard in the recording.
Repeated speech from the same speaker refers to the same speaker identity.

2. Speaking Occurrence

A speaking occurrence is one continuous episode of speech by the same speaker.

* Consecutive speech portions belonging to the same speaker are treated as one
  speaking occurrence.
* A new speaking occurrence for the same speaker begins only when at least one
  different speaker speaks between the two occurrences.
* Silence alone does not create a new speaking occurrence.
* Do not treat separate annotation rows as separate speaking occurrences.
* When the question uses the term "speaking turn", it refers to the same
  counting unit as "speaking occurrence" under this definition.

Example:

Speaker A -> Speaker A -> Speaker B -> Speaker A

Speaker A has 2 speaking occurrences.
Speaker B has 1 speaking occurrence.

3. Audio Event

An audio event is a distinct, identifiable sound or phenomenon occurring at a
particular time or over a particular time interval in the recording. An audio
event may be speech, an environmental sound, or a musical event.

4. Event Occurrence

An event occurrence is one particular instance of an audio event in the
recording.

5. Overlapping Speech

Overlapping speech occurs when two or more speakers are speaking at the same
time.

6. Speaker Ordering

Speakers are ordered according to the time at which they first begin speaking
in the recording.

The first speaker is the first distinct speaker to begin speaking.
The second speaker is the second distinct speaker to begin speaking.
The third speaker is the third distinct speaker to begin speaking, and so on.

If two speakers begin speaking at exactly the same time, do not make an
assumption about their ordering.

ANSWERING RULES

* Determine the answer from the audio and the wording of the question.
* Follow all definitions above exactly.
* Answer only what the question asks.
* Do not use external knowledge or assumptions.
* Do not provide explanations or reasoning.
* Do not output any additional text beyond the answer.
* Infer the required answer type from the question itself.
* The answer-type examples below are illustrative and are NOT an exhaustive
  list of possible answer types.
* The question may require other forms of answers depending on its wording,
  including but not limited to a number, Yes/No, a speaker ordinal, a temporal
  relation such as BEFORE or AFTER, an event or sound name, or another
  concise answer explicitly requested by the question.
* Always output the answer in the form required by the question.
* When the question asks for a categorical or descriptive answer, output the
  requested answer itself rather than explaining it or placing it in a
  sentence.
* Do not convert an answer into a different representation unless the question
  requires it. For example, if the question requires a digit, output a digit;
  if it requires an ordinal such as FIRST or SECOND, output that ordinal; if
  it requires a relation such as BEFORE or AFTER, output that relation.
* The examples below demonstrate the required output style for common answer
  types. They do not limit the possible answers to only those examples.

OUTPUT FORMAT

Infer the required answer format from the question.

If ANSWER_TYPE is YES_NO:
Output exactly one of:
Yes
No

If ANSWER_TYPE is INT:
Output only one non-negative integer.

Examples:
0
1
2
15

Do not output words, units, punctuation, decimal places, or explanations.

If ANSWER_TYPE is SPEAKER_ORDINAL:
Output only the speaker ordinal in uppercase.

Examples:
FIRST
SECOND
THIRD
FOURTH

Do not output the word "speaker" or any explanation.

For questions requiring another categorical or descriptive answer:
Output only the answer requested by the question, using the appropriate
human-readable form.

For example, if the question asks for a temporal relationship, possible
answers may include:
BEFORE
AFTER

If the question asks which audio event or sound:
Output only the event or sound name.
Do not include words such as "event" or "sound", punctuation, or explanations.

For any other question type:
Determine from the wording of the question exactly what form of answer is
required and output only that answer.

Do not assume that the examples above represent all possible answer types.
The answer must be determined by the question.

EXAMPLES

Question: How many distinct speakers are heard in the recording?
Answer type: INT
Answer: 3

Question: Does the second distinct speaker speak between the first time the first distinct speaker speaks and the first time the third distinct speaker speaks?
Answer type: YES_NO
Answer: Yes

Question: Which speaker speaks last in the recording, the first distinct speaker or the second distinct speaker?
Answer type: SPEAKER_ORDINAL
Answer: FIRST

Question: Does the second distinct speaker begin speaking before or after the third distinct speaker?
Answer type: YES_NO
Answer: YES

Question: Does the first distinct speaker make their final speaking occurrence before or after the fourth distinct speaker finishes speaking for the last time?	
Answer type: Before_After
Answer: After
COMMON MISTAKE TO AVOID

Question: How many distinct speakers are heard between two points in the recording?
Answer type: INT

WRONG (do not answer like this):
There are two distinct speakers heard between those two points.

RIGHT:
2

Never restate the question, never use words like "two" or "twice" in place
of a digit, and never wrap the answer in a sentence.

FINAL OUTPUT RULE

Your entire response must contain exactly one answer and nothing else.

Do not output:

* explanations
* reasoning
* sentences
* prefixes such as "Answer:"
* suffixes
* punctuation
* markdown
* quotation marks
* additional words

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


def resolve_audio_path(sample_name):
    """Build the audio file path from sample_name. Adjust if your
    sample_name already includes an extension or subfolders."""
    base = sample_name

    # strip any stray annotation extension that sometimes leaks into
    # sample_name (e.g. "zzyyo.rttm" -> "zzyyo")
    for stray_ext in (".rttm", ".json", ".csv", ".txt"):
        if base.lower().endswith(stray_ext):
            base = base[: -len(stray_ext)]
            break

    name = base if base.lower().endswith(AUDIO_EXT) else base + AUDIO_EXT
    return os.path.join(AUDIO_DIR, name)


def run_inference(model, processor, audio_path, question_text):
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

    generated_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    return output_text.strip()


def load_completed_sl_nos(output_csv):
    """Read whatever rows already exist in output_csv (from a previous,
    possibly interrupted run) so we can resume instead of starting over."""
    completed = set()
    if not os.path.exists(output_csv):
        return completed
    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sl_no = row.get("Sl_no")
                if sl_no is not None:
                    completed.add(sl_no)
    except Exception as e:
        print(f"  !! could not read existing output for checkpointing: {e}")
    return completed


def main():
    global AUDIO_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_csv", default=QUESTIONS_CSV)
    parser.add_argument("--audio_dir", default=AUDIO_DIR)
    parser.add_argument("--output_csv", default=OUTPUT_CSV)
    parser.add_argument("--limit", type=int, default=None, help="only run first N rows (for testing)")
    parser.add_argument(
        "--gpu",
        default="0",
        help='GPU allocation, e.g. "0" for a single GPU or "0,1" for multiple. '
             "Sets CUDA_VISIBLE_DEVICES; must match what was used at import time.",
    )
    args = parser.parse_args()

    AUDIO_DIR = args.audio_dir

    model, processor = load_model(MODEL_PATH)

    with open(args.questions_csv, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    if args.limit:
        rows = rows[: args.limit]

    fieldnames = ["Sl_no", "sample_name", "question", "predicted_answer"]

    # ---- CHECKPOINT: figure out what's already been done ----
    completed_sl_nos = load_completed_sl_nos(args.output_csv)
    resuming = len(completed_sl_nos) > 0
    if resuming:
        print(f"Resuming: {len(completed_sl_nos)} rows already completed in {args.output_csv}, will skip them.")

    remaining_rows = [r for r in rows if r.get("Sl_no") not in completed_sl_nos]
    print(f"{len(remaining_rows)} of {len(rows)} rows left to process.")

    # Open in append mode if resuming an existing file, else write fresh with header.
    file_mode = "a" if resuming else "w"
    write_header = not resuming

    with open(args.output_csv, file_mode, newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(remaining_rows):
            sl_no = row.get("Sl_no")
            sample_name = row.get("sample_name")
            question_text = row.get("question")

            audio_path = resolve_audio_path(sample_name)
            print(f"[{i+1}/{len(remaining_rows)}] Sl_no={sl_no} sample={sample_name}")

            if not os.path.exists(audio_path):
                print(f"  !! audio file not found: {audio_path}")
                answer = "ERROR_AUDIO_NOT_FOUND"
            else:
                try:
                    answer = run_inference(model, processor, audio_path, question_text)
                except Exception as e:
                    print(f"  !! inference failed: {e}")
                    traceback.print_exc()
                    answer = "ERROR_INFERENCE_FAILED"

            writer.writerow(
                {
                    "Sl_no": sl_no,
                    "sample_name": sample_name,
                    "question": question_text,
                    "predicted_answer": answer,
                }
            )
            f_out.flush()

    print(f"Done. Predictions written to {args.output_csv}")


if __name__ == "__main__":
    main()
