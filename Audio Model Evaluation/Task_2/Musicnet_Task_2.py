"""
Run Gemma-3n-E4B-it over the MusicNet (Task 2) question set and write
predicted answers to a CSV. Ground truth is never read and never written to the output.
GPU(s) are chosen with the --gpu flag, e.g.  --gpu 0

Input CSV columns expected: Sl_no, sample_name, case_id, case_type, target_event,
                            subquestion_type, question, ground_truth
                            (sample_name looks like "1727.csv" -> audio file 1727.wav)
Output CSV columns:         Sl_No, sample_name, case_id, case_type, target_event,
                            subquestion_type, question, predicted_answer
                            (ground_truth is NOT read and NOT written)

CHECKPOINTING / RESUME
  * Every row is written and fsync'd to disk as soon as it is predicted.
  * If the script is stopped and re-run, rows already in the output CSV are
    skipped automatically.
  * Rows that previously failed (ERROR_*) are re-tried on resume.
  * Use --restart to ignore existing output and start from scratch.
"""

import os
import csv
import glob
import argparse
import traceback

# --------------------------- GPU ALLOCATION -------------------------------
# Must happen BEFORE torch/transformers are imported: CUDA_VISIBLE_DEVICES is
# read once at process start and decides which physical GPUs this process sees.
# Gemma-3n-E4B is small (~16 GB in bf16), so a single GPU is enough.

def _parse_gpu_arg():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", default="0")
    known, _ = p.parse_known_args()
    return known.gpu


_gpu_arg = _parse_gpu_arg()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg
print(f"[GPU] CUDA_VISIBLE_DEVICES set to: {_gpu_arg}")

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# ----------------------------- CONFIG ------------------------------------

MODEL_PATH = "path/to/gemma-3n-E4B-it"
MODEL_NAME = os.path.basename(os.path.normpath(MODEL_PATH))

TASK_DIR = "Task_2"

QUESTIONS_CSV = os.path.join(TASK_DIR, "MusicNet.csv")
AUDIO_DIR = os.path.join(TASK_DIR, "METADATA", "BIN_1_0-30s")        # folder that holds the .wav files
AUDIO_EXT = ".wav"
OUTPUT_CSV = os.path.join(TASK_DIR, MODEL_NAME, f"{MODEL_NAME}_MusicNet_Task2.csv")

MAX_NEW_TOKENS = 32

FIELDNAMES = ["Sl_No", "sample_name", "case_id", "case_type", "target_event",
              "subquestion_type", "question", "predicted_answer"]

# NOTE: the prompt contains literal braces (e.g. "{B, C}"), so the question is
# inserted with .replace("{question}", ...) and NOT with str.format().
PROMPT_TEMPLATE = r"""
TASK

Carefully analyze the entire provided MusicNet audio recording and answer
the question using only information that can be determined from the audio.

IMPORTANT

Analyze the FULL provided audio before answering.

Pay attention to all musical notes, instruments, their temporal positions,
repetitions, and simultaneous or overlapping events that are relevant to
the question.

Do not use external knowledge or assumptions about the musical piece.

==================================================
MUSICNET DEFINITIONS
==================================================

1. MUSICAL EVENT

A musical event is one identifiable musical note occurrence in the
recording.

A musical event can be described by:

- note / pitch
- instrument
- temporal position

Examples:

- C5
- Violin
- Violin playing C5

==================================================
2. NOTE-EVENT OCCURRENCE
==================================================

For this benchmark, EACH INDIVIDUAL MUSICAL NOTE EVENT is one occurrence.

Do NOT merge separate note events because they:

- have the same pitch,
- have the same instrument,
- are adjacent,
- overlap,
- occur very close together.

Each separate note event must be treated as a separate occurrence.

Examples:

Two separate C5 notes = TWO occurrences.

Three separate notes played by the violin = THREE occurrences.

A piano C5 and a violin C5 = TWO note-event occurrences.

For counting questions, count every qualifying note occurrence separately.

==================================================
3. NOTE / PITCH IDENTITY
==================================================

Questions use human-readable musical note names such as:

- B2
- C3
- C5
- E5
- F#5
- Gb5

Treat enharmonic spellings such as F#5 and Gb5 as the same pitch.

Do not confuse pitch identity with instrument identity.

For example:

- Piano C5 and Violin C5 are two separate note occurrences.
- Both are C5 for a question asking only about C5.

==================================================
4. INSTRUMENT IDENTITY
==================================================

Questions use human-readable individual instrument names such as:

- Acoustic Grand Piano
- Violin
- Viola
- Cello
- Contrabass
- Trumpet
- Trombone
- Flute
- Clarinet
- Acoustic Bass

Use the specific instrument named in the question.

Do NOT replace an individual instrument with a broader instrument group.

For example:

- Violin is not the same as Strings.
- Cello is not the same as Strings.
- Trumpet is not the same as Brass.

When a question refers to an instrument only, consider all note
occurrences played by that instrument regardless of pitch.

When a question refers to both an instrument and a note, both properties
must match.

Example:

"How many Violin C5 notes are played?"

Count only note occurrences that are both:

- played by the violin
- C5

==================================================
5. TEMPORAL ORDER
==================================================

Determine temporal order from when musical events occur in the audio.

Do not rely on annotation order, musical score order, or assumptions
about the structure of the piece.

==================================================
6. FIRST / LAST / ORDINAL OCCURRENCES
==================================================

When a question refers to:

- first
- second
- third
- last

identify the relevant occurrences according to their temporal position
in the recording.

Examples:

"the first C5 note"
= the earliest C5 occurrence.

"the second Violin note"
= the second-earliest violin note occurrence.

"the third Cello note"
= the third-earliest cello note occurrence.

"the last piano note"
= the latest piano note occurrence.

Do not confuse:

- the first occurrence of a pitch,
- the first occurrence of an instrument,
- the first occurrence of an instrument + pitch combination.

For example:

"the first Violin C5 note"

means the earliest note occurrence satisfying BOTH:

- Violin
- C5

==================================================
7. BEFORE
==================================================

Event X is BEFORE event Y when X finishes before or exactly when Y
begins.

Conceptually:

    X.end <= Y.start

If two events overlap in time, X is NOT before Y.

==================================================
8. AFTER
==================================================

Event X is AFTER event Y when X begins at or after the time Y finishes.

Conceptually:

    X.start >= Y.end

If two events overlap in time, X is NOT after Y.

==================================================
9. OVERLAP
==================================================

Two events overlap when they occur simultaneously for a non-zero
time interval.

Conceptually:

    X.start < Y.end
    AND
    Y.start < X.end

Do not treat simultaneously occurring notes as sequential.

==================================================
10. BETWEEN
==================================================

When the question refers to an event occurring BETWEEN event A and
event B, consider the temporal region after A finishes and before
B begins.

A target event X is inside BETWEEN(A,B) when:

    X.start >= A.end
    AND
    X.end <= B.start

Thus, "between A and B" refers to the interval bounded by the end of
A and the beginning of B.

Do NOT interpret "between" using annotation order or score order.

==================================================
TASK 2 — MULTI-CONSTRAINT TEMPORAL REASONING
==================================================

Task 2 evaluates reasoning over TWO temporal constraints within the
same audio recording.

A Task-2 question may define two temporal intervals:

    I1 = BETWEEN(A, B)

and:

    I2 = BETWEEN(C, D)

where A, B, C, and D are identified musical events.

You must identify the relevant events from the audio, determine the
corresponding temporal intervals, and then perform the operation
requested by the question.

The question may ask about:

- events in I1,
- events in I2,
- whether I1 and I2 overlap,
- events satisfying BOTH I1 and I2,
- events satisfying AT LEAST ONE of I1 or I2.

==================================================
11. INTERSECTION
==================================================

When the question asks for events satisfying BOTH temporal conditions,
count only events that satisfy both conditions simultaneously.

This corresponds to:

    I1 ∩ I2

Each qualifying note occurrence is counted EXACTLY ONCE.

Example:

If:

    I1 contains A, B, C

and:

    I2 contains B, C, D

then:

    I1 ∩ I2 = {B, C}

Therefore the intersection contains TWO target events.

Do NOT count shared events twice.

==================================================
12. UNION
==================================================

When the question asks for events satisfying AT LEAST ONE of the two
temporal conditions, count every target occurrence satisfying either
condition.

This corresponds to:

    I1 ∪ I2

Each qualifying note occurrence is counted EXACTLY ONCE.

If an event belongs to both intervals, count it only once.

Example:

If:

    I1 contains A, B, C

and:

    I2 contains B, C, D

then:

    I1 ∪ I2 = {A, B, C, D}

Therefore the union contains FOUR target events.

Do NOT independently add the two counts and double-count shared events.

==================================================
13. OVERLAP CONTROL
==================================================

When the question asks whether two temporal intervals overlap,
determine whether the intervals share any non-zero amount of time.

Answer:

Yes

when they overlap.

Answer:

No

when they do not overlap.

Do NOT infer interval overlap merely from the presence or absence of
target musical events.

==================================================
14. MULTI-CONSTRAINT COUNTING
==================================================

Do NOT answer a multi-constraint question by independently solving
the two conditions and blindly combining their counts.

Instead:

1. identify the relevant reference events,
2. determine the temporal interval or intervals,
3. identify target note occurrences satisfying each condition,
4. apply the requested operation,
5. count each qualifying occurrence exactly once.

For an intersection question:

    count(I1 ∩ I2)

For a union question:

    count(I1 ∪ I2)

==================================================
15. QUESTION-SPECIFIC INTERPRETATION
==================================================

The exact wording of the question determines the required computation.

Examples:

"How many piano notes occur between the first violin note and the
third cello note?"

→ Count piano note occurrences inside that temporal interval.

"How many C5 notes occur between the first violin note and the
third cello note?"

→ Count all C5 occurrences inside that interval, regardless of
instrument.

"How many C5 notes occur within both intervals?"

→ Count C5 occurrences satisfying both temporal constraints.

"How many C5 notes occur in either interval?"

→ Count C5 occurrences satisfying at least one temporal constraint,
counting a shared occurrence only once.

"Do the two intervals overlap?"

→ Determine whether the two temporal intervals overlap and answer Yes
or No.

==================================================
ANSWERING RULES
==================================================

* Analyze the entire provided audio before answering.
* Follow all definitions above exactly.
* Answer only what the question asks.
* Use only information that can be determined from the audio.
* Do not use external knowledge.
* Do not make assumptions about the musical piece.
* Do not use annotation metadata, timestamps, or hidden information.
* Do not provide explanations or reasoning.
* Do not restate the question.
* Do not provide calculations.
* Do not add "Answer:" or any other prefix.
* Do not add commentary.
* Output only the requested answer.
* Use the answer representation required by the question.

==================================================
OUTPUT FORMAT
==================================================

INTEGER QUESTIONS

If the question asks "How many..." or otherwise requests a count:

Output ONLY one non-negative integer.

Examples:

0
1
2
15

Do not output:

- "two"
- "2 notes"
- "There are 2"
- explanations
- punctuation

YES/NO QUESTIONS

If the question asks whether something occurs, overlaps, precedes,
follows, or otherwise asks a Yes/No question:

Output exactly one of:

Yes
No

Do not output a sentence.

NOTE IDENTIFICATION

If the question asks which note:

Output only the human-readable note name.

Examples:

C5
E5
F#5

INSTRUMENT IDENTIFICATION

If the question asks which instrument:

Output only the human-readable instrument name.

Examples:

Violin
Cello
Acoustic Grand Piano
Flute

ORDINAL ANSWERS

If the question asks for an ordinal:

Output only the ordinal in uppercase.

Examples:

FIRST
SECOND
THIRD
FOURTH

TEMPORAL RELATION

If the question asks for a temporal relation:

Output only:

BEFORE

or:

AFTER

For any other answer type:

Output only the concise answer explicitly requested by the question.

==================================================
COMMON MISTAKES TO AVOID
==================================================

1. Do NOT merge separate occurrences of the same note.

2. Do NOT confuse pitch identity with instrument identity.

3. Do NOT assume that all C5 notes are played by the same instrument.

4. Do NOT assume that all notes played by one instrument have the same
   pitch.

5. Do NOT treat overlapping musical events as sequential.

6. Do NOT use annotation order to determine temporal order.

7. Do NOT interpret "between" using annotation order or musical score
   order.

8. Do NOT confuse the first occurrence of an instrument with the first
   occurrence of an instrument + note combination.

9. For INTERSECTION questions, count only events satisfying BOTH
   constraints.

10. For UNION questions, count each qualifying event only once.

11. Do NOT simply add the counts from two overlapping intervals for a
    union question.

12. Do NOT answer from assumptions about what normally occurs in music.

13. Do NOT use hidden timestamps or metadata.

==================================================
FINAL OUTPUT RULE
==================================================

Your entire response must contain EXACTLY ONE ANSWER and nothing else.

Do not output:

- explanations
- reasoning
- sentences
- prefixes such as "Answer:"
- suffixes
- punctuation
- markdown
- quotation marks
- additional words

QUESTION:

{question}

ANSWER:
"""

# ---------------------------------------------------------------------------


def get_field(row, name):
    """Case-insensitive column lookup (handles Sl_No vs Sl_no etc.)."""
    for k, v in row.items():
        if k is not None and k.strip().lower() == name.lower():
            return v
    return None


def load_model(model_path):
    print(f"Loading model from {model_path} ...")
    print(f"torch sees {torch.cuda.device_count()} GPU(s)")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")
    print(f"Model is on device(s): {model.hf_device_map if hasattr(model, 'hf_device_map') else model.device}")
    return model, processor


def resolve_audio_path(sample_name, audio_dir):
    """Build the audio file path from sample_name (with or without .wav)."""
    base = str(sample_name).strip()

    # strip any stray annotation extension that leaks into sample_name
    for stray_ext in (".rttm", ".json", ".csv", ".txt", ".mid", ".midi"):
        if base.lower().endswith(stray_ext):
            base = base[: -len(stray_ext)]
            break

    name = base if base.lower().endswith(AUDIO_EXT) else base + AUDIO_EXT
    return os.path.join(audio_dir, name)


def run_inference(model, processor, audio_path, question_text):
    # .replace instead of .format because the prompt has literal { } characters
    prompt_text = PROMPT_TEMPLATE.replace("{question}", question_text)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # The processor loads the audio itself from the path.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    generated_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return output_text.strip()


def make_key(sl_no, sample_name, question):
    """Unique id for a row, used to detect already-finished rows on resume."""
    return (str(sl_no).strip(), str(sample_name).strip(), str(question).strip())


def load_checkpoint(output_csv):
    """Read the existing output CSV. Returns (good_rows, done_keys).
    Rows whose answer starts with ERROR_ are dropped so they get re-tried."""
    good_rows, done_keys = [], set()
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return good_rows, done_keys

    with open(output_csv, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ans = (r.get("predicted_answer") or "").strip()
            if ans.startswith("ERROR_"):
                continue
            good_rows.append({k: r.get(k, "") for k in FIELDNAMES})
            done_keys.add(make_key(r.get("Sl_No"), r.get("sample_name"), r.get("question")))
    return good_rows, done_keys


def rewrite_clean_output(output_csv, good_rows):
    """Atomically rewrite the output with only good rows (drops ERROR_ rows)."""
    tmp = output_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(good_rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output_csv)


def find_questions_csv(path):
    if os.path.exists(path):
        return path
    if not os.path.isdir(TASK_DIR):
        raise FileNotFoundError(f"TASK_DIR does not exist: {TASK_DIR}  (check spelling/case)")
    candidates = glob.glob(os.path.join(TASK_DIR, "**", "*.csv"), recursive=True)
    if len(candidates) == 1:
        print(f"Questions CSV '{path}' not found; using {candidates[0]}")
        return candidates[0]
    raise FileNotFoundError(
        f"Questions CSV not found at '{path}'. CSVs under {TASK_DIR}: {candidates}. "
        f"Pass the right one with --questions_csv."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_csv", default=QUESTIONS_CSV)
    parser.add_argument("--audio_dir", default=AUDIO_DIR)
    parser.add_argument("--output_csv", default=OUTPUT_CSV)
    parser.add_argument("--limit", type=int, default=None, help="only run first N rows (for testing)")
    parser.add_argument("--gpu", default=_gpu_arg,
                        help='GPU ids, e.g. "0" or "0,1". Read at start-up, before torch is imported.')
    parser.add_argument("--restart", action="store_true",
                        help="ignore any existing output and start from scratch")
    args = parser.parse_args()

    questions_csv = find_questions_csv(args.questions_csv)
    audio_dir = args.audio_dir

    n_wav = len(glob.glob(os.path.join(audio_dir, "*" + AUDIO_EXT)))
    print(f"Audio dir: {audio_dir}  ({n_wav} {AUDIO_EXT} files found)")
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio dir does not exist: {audio_dir}  (check spelling/case)")
    if n_wav == 0:
        print("  WARNING: no audio files found here - check --audio_dir / AUDIO_DIR.")

    with open(questions_csv, "r", newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))
    if args.limit:
        rows = rows[: args.limit]

    # ---- checkpoint handling ----
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if args.restart or not os.path.exists(args.output_csv):
        good_rows, done_keys = [], set()
        rewrite_clean_output(args.output_csv, [])
        print("Starting fresh.")
    else:
        good_rows, done_keys = load_checkpoint(args.output_csv)
        rewrite_clean_output(args.output_csv, good_rows)   # drop old ERROR_ rows
        print(f"Resuming: {len(done_keys)} rows already done, they will be skipped.")

    todo = [
        r for r in rows
        if make_key(get_field(r, "Sl_No"), get_field(r, "sample_name"), get_field(r, "question"))
        not in done_keys
    ]
    print(f"{len(todo)} rows to process out of {len(rows)}.")
    if not todo:
        print("Nothing to do. Exiting.")
        return

    model, processor = load_model(MODEL_PATH)

    with open(args.output_csv, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)

        for i, row in enumerate(todo):
            sl_no = get_field(row, "Sl_No")
            sample_name = get_field(row, "sample_name")
            question_text = get_field(row, "question")

            audio_path = resolve_audio_path(sample_name, audio_dir)
            print(f"[{i+1}/{len(todo)}] Sl_No={sl_no} sample={sample_name}")

            if not os.path.exists(audio_path):
                print(f"  !! audio file not found: {audio_path}")
                answer = "ERROR_AUDIO_NOT_FOUND"
            else:
                try:
                    answer = run_inference(model, processor, audio_path, question_text)
                except torch.cuda.OutOfMemoryError:
                    print("  !! CUDA out of memory")
                    torch.cuda.empty_cache()
                    answer = "ERROR_OOM"
                except Exception as e:
                    print(f"  !! inference failed: {e}")
                    traceback.print_exc()
                    answer = "ERROR_INFERENCE_FAILED"

            writer.writerow(
                {
                    "Sl_No": sl_no,
                    "sample_name": sample_name,
                    "case_id": get_field(row, "case_id"),
                    "case_type": get_field(row, "case_type"),
                    "target_event": get_field(row, "target_event"),
                    "subquestion_type": get_field(row, "subquestion_type"),
                    "question": question_text,
                    "predicted_answer": answer,
                }
            )
            # checkpoint: force this row onto disk immediately
            f_out.flush()
            os.fsync(f_out.fileno())

    print(f"Done. Predictions written to {args.output_csv}")


if __name__ == "__main__":
    main()
    