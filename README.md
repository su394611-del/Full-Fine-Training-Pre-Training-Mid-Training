# DistilBERT Fine-Tuning for Customer Support Classification

End-to-end **Hugging Face Transformers + PyTorch** project that fine-tunes `distilbert-base-uncased` to classify customer-support messages into four categories: **billing**, **technical**, **account**, and **feature_request**.

This repository includes the original Python training program and an equivalent, cell-by-cell Jupyter Notebook for local development or Google Colab. It also compares **full fine-tuning** against training only the **classification head with a frozen DistilBERT backbone**.

> **:** The included dataset has only **30 hand-written examples** (24 training and 6 validation). Accuracy and F1 measured on this split are **not representative of production performance**. No evaluation scores are claimed in this README.

## Project files

```text
.
├── README.md
├── requirements.txt
├── Full_Fine_Tuning_Using_DistilBERT.py
├── Full_Fine_Tuning_Using_DistilBERT.ipynb
└── .gitignore
```

| File | Purpose |
| --- | --- |
| `Full_Fine_Tuning_Using_DistilBERT.py` | Standalone Python script (the uploaded source, preserved) |
| `Full_Fine_Tuning_Using_DistilBERT.ipynb` | Notebook version organized into sequential learning sections |
| `requirements.txt` | Python runtime and notebook dependencies |
| `.gitignore` | Prevents virtual environments, generated checkpoints, and caches from being committed |

## Learning objectives

- Inspect the installed PyTorch/CUDA environment and choose CPU or GPU automatically.
- Create a labeled dataset with **billing**, **technical**, **account**, and **feature_request** classes.
- Split the data using stratified sampling and prepare a Hugging Face `DatasetDict`.
- Tokenize text with `AutoTokenizer` and dynamically pad batches.
- Fine-tune `AutoModelForSequenceClassification` using Hugging Face `Trainer`.
- Compute accuracy, weighted F1, a classification report, and a confusion matrix.
- Save a checkpoint and classify unseen support queries with confidence scores.
- Freeze the DistilBERT encoder and train the classification head for comparison.

## Prerequisites

- **Python 3.11 or 3.12 recommended** (Windows 11, Linux, macOS, or Google Colab).
- Internet access for the first download of [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased), unless its files are cached.
- Sufficient disk space for dependencies, the model, and training checkpoints.
- An **NVIDIA CUDA GPU is optional**; CPU training works but is slower. Installing `torch` alone does **not** guarantee CUDA support.

### Install dependencies (Windows PowerShell / VS Code terminal)

```powershell
# Optional but recommended: a fresh Python 3.12 Conda environment
conda create -n distilbert-ft python=3.12 -y
conda activate distilbert-ft

# Run from the project directory
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Or install the packages directly:

```powershell
python -m pip install "torch>=2.6,<3" "transformers>=4.46,<6" "datasets>=3,<6" "accelerate>=1,<2" "numpy>=1.26,<3" "pandas>=2.2,<4" "scikit-learn>=1.4,<2" "jupyterlab>=4,<5" "ipykernel>=6.29,<8" "nbformat>=5.10,<6"
```

To use an NVIDIA GPU, follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your operating system and supported CUDA build. Verify your active Python environment:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

### Run as a Python script

```powershell
python Full_Fine_Tuning_Using_DistilBERT.py
```

The script runs both experiments **sequentially**: first the full fine-tune, then the frozen-backbone comparison. The initial model download can take time; the model may also be loaded a second time for the comparison.

### Run as a Jupyter Notebook or in VS Code

```powershell
jupyter lab
```

Open `Full_Fine_Tuning_Using_DistilBERT.ipynb`, select the environment where you installed the dependencies, and choose **Run All**. You can also open the `.ipynb` file in VS Code with the Jupyter extension. **Run cells in order**, since later sections reuse variables from earlier cells.

For Google Colab, upload the `.ipynb` file and install packages in a notebook cell if needed:

```python
%pip install "transformers>=4.46,<6" "datasets>=3,<6" "accelerate>=1,<2" "numpy>=1.26,<3" "pandas>=2.2,<4" "scikit-learn>=1.4,<2"
```

Colab usually provides PyTorch already. For GPU acceleration, select a GPU runtime and rerun the notebook. If you change dependencies, restart the kernel/runtime before rerunning.

## Workflow

```text
30 labeled customer-support messages
          │
          ▼
Stratified train/validation split (24 / 6)
          │
          ▼
DistilBERT tokenizer → dynamic batch padding
          │
          ├──────── Full fine-tuning ────────┐
          │      DistilBERT + classifier      │
          │                                   ▼
          │                         Accuracy, F1, report,
          │                         confusion matrix and
          │                         example predictions
          │
          └──────── Frozen backbone ────────┐
                 Classification head only   ▼
                                    Accuracy and F1
          │
          ▼
Compare the two approaches (educational only)
```

### Included training settings

| Setting | Full fine-tuning | Frozen-backbone baseline |
| --- | --- | --- |
| Pretrained model | `distilbert-base-uncased` | `distilbert-base-uncased` |
| Epochs | 4 | 6 |
| Train batch size | 8 | 8 |
| Learning rate | `2e-5` | `1e-3` |
| Sequence length | 128 tokens | 128 tokens |
| Evaluation | Every epoch | Every epoch |
| Trainable layers | Full DistilBERT + head | `pre_classifier` + `classifier` (encoder frozen) |
| Metrics | Accuracy, weighted F1, report, confusion matrix | Accuracy, weighted F1 |

### Generated outputs

The script writes artifacts under these directories:

```text
support_classifier/
    best_model/            # Selected fine-tuned model and tokenizer
    checkpoint-*/          # Trainer checkpoints (if retained)
support_classifier_frozen/ # Frozen-backbone training logs/outputs
```

These paths are excluded from Git by `.gitignore`. The script also prints example classifications and comparison metrics to the terminal/notebook output. Model files are **not** included in this repository.

## Code compatibility and limitations

- The program checks `TrainingArguments` and `Trainer` signatures to accommodate selected differences between Transformers 4.x and 5.x; this is **not a guarantee of compatibility with every future version**.
- The six-example validation split is extremely small. Even stratification cannot make its metrics reliable for deployment.
- Full and frozen-backbone runs start from independently loaded pretrained encoders. Results may vary by software version, GPU, stochastic initialization, and training conditions.
- The notebook is an **equivalent conversion of the Python program**, not a separately trained or benchmarked model. Notebook cells depend on running earlier cells first.
- Running the project downloads an upstream model; review its license/model card and the relevant library licenses before commercial reuse.
- This is **sequence classification with an encoder model**. In the next iteration, we will introduce mid-training for our own LLM and use consumer-grade Nvidia GPUs, not A100/H100.

## Troubleshooting

**`ModuleNotFoundError`**: Confirm the selected Python interpreter or notebook kernel and install dependencies with `python -m pip install -r requirements.txt` or `%pip install ...` in the active notebook environment.

**CUDA unavailable**: `nvidia-smi` showing a GPU does not establish that your PyTorch build has CUDA enabled. Check `torch.version.cuda` and `torch.cuda.is_available()`; CPU execution remains supported.

**Out of memory**: Lower `per_device_train_batch_size` in both `TrainingArguments` configurations from `8` to `4` or `2`. Free GPU memory and restart the kernel before retrying.

**Slow CPU training**: Use a supported GPU runtime; the script performs two separate training experiments.

**Model download fails**: Verify network access to Hugging Face and any proxy/firewall restrictions, then retry.

**Notebook variables undefined**: Restart the kernel and run the cells from the beginning in order.

## References

- [DistilBERT model on Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

## License

This project is licensed under the Apache License, Version 2.0.© 2026 Sudhanshu Shukla · Licensed under the Apache License, Version 2.0.
