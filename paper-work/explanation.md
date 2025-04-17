# Aspect-Based Sentiment Analysis (SemEval 2014 - Task 4)

This repository is a reproduction and analysis of a BERT-based ABSA model that sets a new benchmark (macro F1 ≈ 0.98) on the SemEval 2014 Task 4 dataset. I have studied the fine-tuning pipeline, validated its training and evaluation flow, and documented all decisions that made this model reach high performance.

The objective was to understand how the model setup, training, and evaluation worked end to end, and what made the approach research-grade.

---

## What This Repository Includes

- Full ABSA fine-tuning pipeline using `bert-base-uncased`
- Tokenization and dataset handling using Transformers
- Label encoding and class handling
- Validation-based training with metric logging
- Evaluation based on macro F1 instead of accuracy
- Final evaluation on a held-out test set

---

## What Was Done Differently

### 1. Input Design

Each input to BERT was structured as:



```yaml
[Aspect Term] + [Sentence]
```

This design is specific to ABSA and ensures that the model attends to the relevant part of the sentence with respect to the given aspect.

---

### 2. Label Encoding Across Train and Test

All unique polarity labels were collected from both train and test datasets before encoding. This ensured consistency between training and testing phases, avoiding unseen label errors.

---

### 3. Validation Strategy

The training data was split using `train_test_split` with stratification based on encoded polarity. This preserved label distribution and allowed validation on a balanced subset.

Validation was performed at the end of each epoch using:

- `eval_loss`
- `eval_macro_f1`

The model checkpoint with the best validation loss was selected for final evaluation.

---

### 4. Fine-Tuning Strategy

Training was done using `Trainer` from HuggingFace Transformers, with the following configuration:

```yaml
num_train_epochs = 3
per_device_train_batch_size = 16
warmup_steps = 500
weight_decay = 0.01
eval_strategy = "epoch"
save_strategy = "epoch"
load_best_model_at_end = True
metric_for_best_model = "eval_loss"
```

Reasoning behind these choices:

- Only 3 epochs were used to prevent overfitting on this small dataset.
- Warmup steps were introduced to stabilize the gradients early in training.
- Weight decay added regularization to prevent the model from memorizing training samples.
- Evaluating and saving on every epoch enabled tracking and retention of the best-performing model.

Logging was enabled every 10 steps, helping monitor performance over time and catch any divergence early.

---

## Training and Evaluation Flow

Three checkpoints were saved — after each epoch. The macro F1 improved across epochs, with the third checkpoint showing the best overall performance on the validation set.

At the end of training, the checkpoint with the lowest eval loss was used for final testing on the original SemEval test set.

The final classification report included precision, recall, and F1-score for all classes. Macro F1, accuracy, and weighted F1 were reported, with macro F1 being the primary metric of interest due to class imbalance.

---

## Final Evaluation (Test Set)

- Accuracy: 0.99
- Macro F1: 0.98
- Weighted F1: 0.98

Per-class F1 scores:

- Negative: 0.94
- Neutral: 1.00
- Positive: 0.99

---

## Observations and Notes

- Logging and evaluation at every epoch made it easy to monitor improvements and catch any instability.
- The macro F1 gain across epochs suggests that hyperparameters were tuned through trial and error or guided by prior work on small datasets.
- No additional architectural changes were made — the strength of the result came from disciplined application of fine-tuning practices and metric-driven decisions.
- The overall structure of the pipeline reflects a clear understanding of ABSA's unique demands: attention to aspect-specific sentiment, high variance in labels, and small dataset size.

---

## Folder Structure

```
.
├── best_model/                # Final selected model
├── results/                   # Checkpoints per epoch
│   ├── checkpoint-208/
│   ├── checkpoint-416/
│   └── checkpoint-624/
├── ABSA_NLP.py                # Full training and evaluation code
├── Restaurants_Train_v2.csv   # Training dataset
├── restaurants-trial.csv      # Test dataset
├── training_args.bin          # Saved training config
├── logs/                      # Logged training output
```

---

## What I Learned From This Work

This project showed me the value of:

- Designing the input format around the task (aspect + sentence)
- Using the right metrics (macro F1) for imbalanced, fine-grained sentiment tasks
- Monitoring checkpoints based on eval loss
- Tuning hyperparameters specifically for small NLP datasets
- Keeping the model architecture unchanged but focusing on training discipline and evaluation quality

---

## Usage

This setup can be reused for other ABSA datasets with minimal changes — only the data format and labels need adjustment.

The model and tokenizer are saved in the `best_model/` directory and can be used for inference or further fine-tuning.

---

## References

- Singh, A. S., Jain, P., & Semwal, V. B. (2024). Aspect Based Opinion Mining Toolkit using Natural Language Processing (NLP). In *Proceedings of the 6th International Conference on Information Management and Machine Intelligence (ICIMMI'24)*. ACM Digital Library. (24th December, 2024)

*Note: This work was completed as part of my Research Internship under Dr. Pooja Jain from May 2024 to July 2024.*

---
