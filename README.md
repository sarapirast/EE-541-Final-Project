# ATIS Intent Classification with BiLSTM Architectures

This repository contains the implementation of bidirectional LSTM models for intent classification on the ATIS (Airline Travel Information System) dataset. The project compares four architectures: single-layer BiLSTM, two-layer BiLSTM, BiLSTM with attention mechanism, and Transformer encoder.

## Repository Structure

```
.
├── data/                          # Processed CSV files (train/val/test splits)
├── ATIS_dataset/                  # Raw ATIS dataset in Rasa format
├── paths/                         # Saved model checkpoints (.pt files)
├── plots/                         # Generated figures (loss curves, confusion matrices)
├── preprocessing.py               # Data loading, normalization, and splitting
├── token_and_augment.py          # Tokenization and data augmentation utilities
├── build_vocab.py                # Vocabulary construction from training data
├── evaluation.py                 # Evaluation metrics and confusion matrix generation
├── bilstm_single.py              # Single-layer BiLSTM training script
├── bilstm_double.py              # Two-layer BiLSTM training script
├── bilstm_attention.py           # BiLSTM with attention mechanism training script
├── transformer_encoder.py        # Transformer encoder training script
└── README.md                     # This file
```

## Dependencies

```bash
# Core dependencies
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.65.0
regex>=2022.10.31
```

Install all dependencies:
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn tqdm regex
```

## Setup and Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Download ATIS dataset**:
   - Place the ATIS dataset in Rasa format under `ATIS_dataset/data/standard_format/rasa/`
   - Required files: `train.json` and `test.json`

3. **Create output directories**:
```bash
mkdir -p data paths plots
```

4. **Preprocess the data**:
```bash
python preprocessing.py
```
This generates:
- `data/atis_train.csv` (80% of training data)
- `data/atis_val.csv` (20% of training data for validation)
- `data/atis_test.csv` (held-out test set)

**Important**: You must run `preprocessing.py` before training any models, as the training scripts depend on the generated CSV files.

## Data Format

### Input Format (Rasa JSON)
The raw ATIS dataset uses Rasa format with:
- `text`: User query string
- `intent`: Intent label
- `entities`: Named entities (optional, not used in this project)

### Processed CSV Format
After preprocessing, each CSV contains:
- `text`: Tokenized and normalized query
- `intent`: Normalized intent string
- `label`: Integer label (0 to num_classes-1)
- `entities`: Entity annotations (preserved but unused)

### Data Preprocessing Pipeline
1. **Intent normalization**: Consolidates variant labels (e.g., `airfare+flight` → `flight+airfare`)
2. **Rare intent removal**: Filters intents with ≤5 total examples
3. **Tokenization**: Lowercasing, punctuation handling, month removal
4. **Stratified splitting**: 80/20 train/val split maintaining class proportions

## Utility Modules

The following utility modules are used by the training scripts and must be present in the repository:

### `token_and_augment.py`
Provides tokenization and data augmentation functions:
- **Tokenization**: Cleans and tokenizes ATIS queries (lowercasing, punctuation handling, month removal)
- **Data augmentation**: Applies synonym replacement, random deletion, and random insertion
- **Sequence conversion**: Converts tokenized text to padded integer indices for model input

This module is imported by all training scripts but is **not run standalone**.

### `build_vocab.py`
Constructs vocabulary mappings from training data:
- Builds word→index and index→word dictionaries
- Handles special tokens (`<pad>`, `<unk>`)
- Applies minimum frequency threshold

This module is imported and called within training scripts but is **not run standalone**.

### `evaluation.py`
Computes evaluation metrics and generates confusion matrices:
- Overall accuracy, macro/weighted precision, recall, F1
- Per-class classification reports
- Confusion matrix generation and visualization

This module is imported and called at the end of each training script but is **not run standalone**.

**Note**: These three utility modules (`token_and_augment.py`, `build_vocab.py`, `evaluation.py`) are automatically imported and used by the training scripts. You do not need to run them separately.

## Running the Code

### Complete Workflow

1. **Preprocess the data** (required first step):
```bash
python preprocessing.py
```

2. **Train models** (each script imports and uses the utility modules automatically):

**Single-layer BiLSTM**:
```bash
python bilstm_single.py
```

**Two-layer BiLSTM**:
```bash
python bilstm_double.py
```

**BiLSTM with Attention**:
```bash
python bilstm_attention.py
```

**Transformer Encoder**:
```bash
python transformer_encoder.py
```

### What Each Training Script Does

Each training script automatically:
1. Loads preprocessed CSVs from `data/` (requires `preprocessing.py` to have been run)
2. Imports and uses `build_vocab()` to build vocabulary from training data
3. Imports and uses `augment_dataframe()` from `token_and_augment.py` to augment training data
4. Creates PyTorch DataLoaders with tokenization via `token_and_augment()`
5. Trains model for 20 epochs with class-weighted loss
6. Saves best model checkpoint to `paths/<model_name>.pt`
7. Imports and uses `evaluate()` to compute metrics on test set
8. Generates plots (loss curves, confusion matrix) in `plots/`

### Training Outputs

Each training run produces:
1. **Model checkpoint**: Saved to `paths/<model_name>.pt` containing:
   - Model state dict
   - Vocabulary mappings (word2idx, idx2word)
   
2. **Training plots**: Saved to `plots/<model_name>_loss_accuracy_curves.png`
   - Training/validation loss curves
   - Training/test accuracy curves

3. **Confusion matrix**: Saved to `plots/<model_name>_confusion_matrix.png`

4. **Console output**: Training progress, metrics, and final evaluation results

### Hyperparameters

All models use identical hyperparameters (defined in each training script):

```python
seed = 42                # Random seed for reproducibility
batch_size = 32          # Mini-batch size
max_len = 40            # Maximum sequence length (padding/truncation)
embed_dim = 200         # Word embedding dimension
hidden_dim = 128        # LSTM hidden state dimension (256 bidirectional)
num_epochs = 20         # Training epochs
lr = 1e-3               # Learning rate (Adam optimizer)
dropout = 0.3           # Dropout probability (final layer)
inter_layer_dropout = 0.2  # Dropout between LSTM layers (2-layer models)

# Data augmentation
augment_frac = 1.0      # Fraction of training data to augment (1.0 = doubles dataset)
aug_ins = 0.05          # Token insertion probability
aug_del = 0.05          # Token deletion probability
aug_rep = 0.05          # Synonym replacement probability
```

## Data Augmentation

The training pipeline applies token-level augmentation to address dataset size limitations:

1. **Synonym Replacement** (5% probability per token): Replaces tokens with domain-relevant synonyms (e.g., `flight` ↔ `flights`, `airfare` ↔ `airfares`)

2. **Random Deletion** (5% probability per token): Removes tokens while maintaining minimum sequence length of 3 tokens

3. **Random Insertion** (5% probability per token): Duplicates tokens to simulate speech patterns

Augmentation is applied on-the-fly during training with `augment_frac=1.0`, effectively doubling the training set from ~4k to ~8k examples.

## Model Architectures

### BiLSTM-1Layer (Baseline)
```
Input → Embedding(200) → BiLSTM(128) → Dropout(0.3) → FC(num_classes)
```
- Uses final hidden states (forward + backward concatenation)
- ~300K parameters

### BiLSTM-2Layer
```
Input → Embedding(200) → BiLSTM(128) → Dropout(0.2) → BiLSTM(128) → Dropout(0.3) → FC(num_classes)
```
- Hierarchical feature learning across two LSTM layers
- ~400K parameters

### BiLSTM-Attention
```
Input → Embedding(200) → BiLSTM(128) → Dropout(0.2) → BiLSTM(128) → Attention → Dropout(0.3) → FC(num_classes)
```
- Additive attention mechanism over all hidden states
- Dynamic token weighting instead of final-state aggregation
- ~400K parameters

### Transformer Encoder
```
Input → Embedding(200) + Positional Encoding → TransformerEncoder(2 layers, 4 heads, dim=256) → Mean Pooling → Dropout(0.3) → FC(num_classes)
```
- Multi-head self-attention with feed-forward layers
- ~600-800K parameters

## Evaluation Metrics

All models are evaluated using:

- **Overall Accuracy**: Percentage of correct predictions
- **Macro Precision/Recall/F1**: Unweighted average across all classes (reveals rare class performance)
- **Weighted Precision/Recall/F1**: Weighted by class support
- **Per-Class Metrics**: Precision, recall, F1 for each intent
- **Confusion Matrix**: Visual representation of classification errors

Metrics are computed on the held-out test set using the best model checkpoint (selected by validation accuracy).

## Reproducing Results

To reproduce the main results from the project report:

1. **Preprocess the data**:
```bash
python preprocessing.py
```

2. **Train all four models** (each takes ~5-10 minutes on GPU):
```bash
python bilstm_single.py
python bilstm_double.py
python bilstm_attention.py
python transformer_encoder.py
```

3. **Expected results** (approximate, due to random initialization):

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|---------------|-----------------|--------------|----------|
| BiLSTM-1Layer | 96.30% | 88.50% | 85.36% | 84.23% |
| BiLSTM-2Layer | 95.74% | 89.02% | 86.77% | 85.58% |
| BiLSTM-Attention | 95.96% | 88.72% | 81.02% | 81.73% |
| Transformer | TBD | TBD | TBD | TBD |

**Note**: Results may vary slightly (±0.5-1.0%) due to:
- Random weight initialization
- CUDA non-determinism
- Augmentation randomness
- Hardware differences (GPU vs CPU)

All training uses `seed=42` for reproducibility, but perfect bit-for-bit reproduction is not guaranteed across different hardware/CUDA versions.

## Key Findings

1. **Simpler models generalize better**: Single-layer BiLSTM achieves highest accuracy (96.30%) with fewest parameters
2. **Modest depth improves class balance**: Two-layer BiLSTM provides best macro F1 (85.58%), helping rare classes
3. **Attention underperforms on small data**: Attention mechanism degrades macro recall to 81.02%, likely due to insufficient training data (~4k examples)
4. **Class imbalance challenges**: With 62.5% of data being `flight` intent, macro metrics are critical for evaluation

## Code Organization Notes

- **Modular design**: Separate files for preprocessing, tokenization, vocabulary, evaluation, and model training
- **Shared utilities**: `token_and_augment.py`, `build_vocab.py`, and `evaluation.py` are reused across all models via imports
- **Consistent interface**: All training scripts follow the same structure (data loading → model definition → training loop → evaluation)
- **Class weighting**: Inverse frequency weighting applied in loss function to handle class imbalance

## Special Requirements

- **GPU recommended**: Training on CPU is significantly slower
- **PyTorch version**: Tested on PyTorch 2.0+; earlier versions may have compatibility issues

