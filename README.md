# After-The-Second-Bell
Part of the benchmark for audio compositional reasoning

# Audio Model Evaluation
This repository contains the evaluation code used to evaluate Audio-Language Models on different audio datasets.

## Repository Structure

```text
Audio Model Evaluation/
│
├── DataSed/
│   ├── datased.csv
│   ├── DataSED.py
│   ├── audios/
│
├── MusicNet/
│   ├── MusicNet.csv
│   ├── Musicnet.py
│   ├── audios/
│   
│
├── Mustard/
│   ├── mustard.csv
│   ├── Mustard.py
│   ├── audios/
│   
│
├── Task_2/
│   ├── MusicNet_Task2.csv
│   ├── Musicnet_Task_2.py
│   ├── audios/
│   
│
├── Vox/
│   ├── Vox.csv
│   ├── Vox.py
│   ├── audios/
│   
├── Accuracy/
│   └── accuracy-analysis scripts
│
└── README.md
```

## Evaluation Pipeline

The evaluation is performed in two steps:

### 1. Model Evaluation

The Python evaluation script takes:

- Audio files
- Questions CSV
- A selected Audio-Language Model

and generates a prediction CSV containing the model's answers.

```text
Audio + Questions
       ↓
   Model Inference
       ↓
   Prediction CSV
```

### 2. Accuracy Calculation

The generated predictions are compared with the ground-truth answers using the corresponding accuracy-calculation script.

```text
Prediction CSV + Ground Truth
              ↓
      Accuracy Calculation
              ↓
        Accuracy Results
```

## Datasets

The repository currently contains evaluation code for:

- DataSED
- MusicNet
- MUSTARD
- Vox
- MusicNet Task 2

Each dataset has its own evaluation script and corresponding accuracy calculation.

## Models

The repository provides an example evaluation for a model. The same evaluation procedure can be applied to other Audio-Language Models by changing the model configuration/path in the corresponding evaluation script.

Therefore, the code structure is not limited to a single model.

For example:

```text
Dataset
   ├── Model A evaluation
   ├── Model B evaluation
   ├── Model C evaluation
   └── Accuracy calculation
```

## Running Evaluation

A typical evaluation script can be run as:

```bash
python <evaluation_script>.py --gpu 0
```

For testing a small number of samples, where supported:

```bash
python <evaluation_script>.py --gpu 0 --limit 10
```

The GPU can be changed according to the available hardware.

## Output

The evaluation scripts generate a CSV containing the model predictions.

The accuracy scripts then use the prediction and ground-truth answers to calculate the performance of the model.

The same process can be repeated for different models and datasets.
