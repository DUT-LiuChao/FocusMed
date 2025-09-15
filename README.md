# Project Guide

This project covers five stages: dataset download, focus extraction & preprocessing, model training (QLoRA), multidimensional quality assessment, and result selection.

> **TL;DR**  
> 1) Download HQS & MeQSum → 2) `focus_extract.py` → 3) `train.py` (QLoRA) → 4) `assessment.py` → 5) `select.py`.

---

## 0) Prerequisites

- Python ≥ 3.9  
- PyTorch (CUDA recommended)  
- Transformers / PEFT (for QLoRA)  
- Other common NLP deps (datasets, sentencepiece, accelerate, etc.)

> Install your environment as needed, for example:
```bash
pip install -r requirements.txt
```

---

## 1) Download Datasets

Download and prepare the following datasets:

- **HQS** (MEDIQA 2021 Task 1): <https://github.com/abachaa/MEDIQA2021/tree/main/Task1>  
- **MeQSum**: <https://github.com/abachaa/MeQSum>

Organize them under a data directory of your choice, e.g.:
```
data/
  HQS/
    ……
  MeQSum/
    train.json
    dev.json
    test.json
```

> Adjust file names/structures to match your local copies.

---

## 2) Focus Extraction & Preprocessing

Run focus extraction and preprocessing:

```bash
python focus_extract.py   --model_name_and_path <MODEL_NAME_OR_PATH>   --save_path <OUTPUT_DIR>   --dataset_path <DATASET_DIR>   --type train
```

**Arguments**
- `--model_name_and_path`: Base model or local checkpoint used for focus extraction.
- `--save_path`: Directory to save the processed data.
- `--dataset_path`: Directory containing the raw datasets (e.g., `data/HQS` or `data/MeQSum`).
- `--type`: Split to process (`train`, `dev`, `test`).

> Note: The original flag had a typo (`--datset_path`). This README uses `--dataset_path`.

---

## 3) Model Training (QLoRA)

We train with **QLoRA**:

```bash
python train.py --train_args_file train_args.json


```

---

## 4) Multidimensional Quality Assessment

Compute scores for three evaluation metrics:

```bash
python assessment.py --data_path <EVAL_INPUT_PATH>
```

- `--data_path`: File or directory containing predictions (and references if needed by your script).  
- The script outputs per-example and/or aggregate scores for the three indicators used by your pipeline.

---

## 5) Result Selection (Weighted Integration)

Depending on the model used in the **extraction** phase and the **training** phase, you may produce **four** different result sets.  
1) Run **Step 4** to evaluate each set separately.  
2) Then compute weighted scores and pick the highest:

```bash
python select.py   --path1 <RESULTS_SET_1>   --path2 <RESULTS_SET_2>   --path3 <RESULTS_SET_3>   --path4 <RESULTS_SET_4>
```

The script aggregates the evaluation outputs, applies weights, and selects the final best-scoring result.

---



## Suggested Repository Structure

```
.
├─ data/
│  ├─ HQS/
│  └─ MeQSum/
├─ component/
│  ├─ argument.py
│  ├─ collator.py
│  ├─ dataset.py
│  ├─ loss.py
│  ├─ model.py
│  ├─ template.py
│  ├─ trainer.py
│  └─utils.py
├─ assessment.py
├─ focus_extract.py
├─ select.py
├─ train.py
├─ train_args.json
├─ README.md
└─ requirements.txt
```

---

