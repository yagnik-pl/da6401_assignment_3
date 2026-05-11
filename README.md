# DA6401 Assignment 3: Transformer NMT

This folder implements the Transformer from "Attention Is All You Need" from
scratch in PyTorch for German-to-English translation on Multi30k.

## Files

- `model.py`: scaled dot-product attention, multi-head attention, masks,
  sinusoidal/learned positional encodings, encoder, decoder, Transformer.
- `lr_scheduler.py`: Noam learning-rate scheduler.
- `dataset.py`: Multi30k loading, spaCy tokenization, vocabulary, dataloaders.
- `train.py`: label smoothing, training, greedy decoding, BLEU, checkpoints,
  W&B-compatible CSV logging.
- `vocabs.json`: vocabularies used by the saved checkpoint for inference.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the baseline:

```bash
python train.py --experiment single --epochs 10 --use-wandb
```

Run the full report ablation set:

```bash
python train.py --experiment all_report --epochs 10 --use-wandb
```

Use `--device cpu` only if CUDA is unavailable. The default model is compact
enough for Multi30k while still implementing the required architecture.

## Inference

```python
from model import Transformer

model = Transformer()
model.eval()
print(model.infer("Das ist ein Test."))
```

`Transformer()` loads `vocabs.json` and caches the best checkpoint under
`checkpoints/`. If the checkpoint is missing, it downloads it from the configured
Google Drive file/folder in `model.py`.

## CSV Outputs

Report data is generated under `outputs/csv/` when training runs.

- `all_epoch_metrics.csv`: train loss, validation loss, validation accuracy,
  learning rate, prediction confidence.
- `all_step_metrics.csv`: per-step train loss, learning rate, accuracy, and
  prediction confidence.
- `scaling_ablation_grad_norms.csv`: first-1000-step Query/Key gradient norms.
- `encoder_attention_heads.csv`: last encoder layer attention weights by head.
- `all_bleu_scores.csv`: validation and test BLEU for each run.
- `label_smoothing_prediction_confidence.csv`: confidence values for smoothing
  comparison.
- `report_index.csv`: run metadata and paths to run-specific CSV files.

The model uses Pre-LayerNorm residual blocks with a final stack LayerNorm. This
keeps training stable for the small-resource Multi30k experiments while still
preserving the Transformer sublayer structure required in the assignment.
