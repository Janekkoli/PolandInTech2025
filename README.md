# PolandInTech2025

A compact PyTorch project for learning a sentence similarity / closeness score from paired sentence embeddings.

This repository contains a simple training pipeline for a model that:

- takes two sentence representations as input,
- encodes each sentence with a shared embedding network,
- combines the two embeddings,
- predicts a scalar closeness score for the pair.

The project is intentionally lightweight and is organized around a small dataset pipeline and a single training script.

## Project structure

```text
PolandInTech2025/
├── README.md
├── makefile
├── requirments.txt
├── code/
│   ├── dataset.py
│   ├── dataset_conversion.py
│   ├── model.py
│   └── train.py
├── models/
└── data/                  # expected runtime data directory
```

## Overview of the model

The main model is defined in `code/model.py`.

- `MySentenceEmbedding`:
  - takes a token sequence of shape `(batch, seq_len, 768)`,
  - pools over the sequence dimension with mean pooling,
  - projects it through a small MLP to an embedding vector of size 30.

- `SentenceEmbedingClosnesnessModel`:
  - embeds each sentence independently with the same network,
  - concatenates the two resulting vectors,
  - feeds the combined vector through a linear layer,
  - returns a single scalar score representing sentence similarity/closeness.

The model is trained with MSE loss and Adam optimization.

## Requirements

The project depends on PyTorch.

```bash
pip install -r requirments.txt
```

> Note: the existing file is named `requirments.txt` in this repository. If you want to standardize it, you can rename it to `requirements.txt`.

The repository comment suggests Python 3.12 compatibility, but the actual code is standard PyTorch Python code and should work in a compatible Python 3.10+ environment.

## Data pipeline

The raw sentence data is stored in `.dat` files and converted into PyTorch tensors before training.

The conversion workflow is handled by `code/dataset_conversion.py`, which expects files like:

- `data/sentence_pack_1.dat`
- `data/sentence_pack_2.dat`
- `data/label_pack.dat`

This script:

1. reads the raw token data,
2. pads each sequence to a fixed length,
3. reshapes the data into a tensor of shape `(count, 2, seq_len, token_dim)`,
4. saves the pair data as `data/sentence_pack.pt`,
5. saves the labels as `data/label_pack.pt`.

The dataset class in `code/dataset.py` loads those `.pt` files and yields `(sample, label)` pairs for training.

## Training

Run the training script:

```bash
python code/train.py
```

This script:

1. builds the model,
2. creates an `Adam` optimizer,
3. creates a `DataLoader` from `SentencePackDataset`,
4. trains for a fixed number of epochs,
5. logs training and validation loss,
6. saves the sentence embedding weights to `models/sentence_embedding_model.pth`.

A typical training loop looks like this:

```python
model = SentenceEmbedingClosnesnessModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

## Model packaging / export

The included `makefile` contains a packaging step that zips the model artifact together with the model definition.

```bash
make
```

This creates a zip archive containing the saved model and the model code.

## Notes and caveats

- The project currently assumes a `data/` folder exists before running conversion or training.
- The model expects sentence features shaped as 768-dimensional token embeddings.
- The repository uses a few naming conventions that may need cleanup over time, including the `requirments.txt` filename and some model artifact naming differences (`.pth` vs `.bin`).
- The exported `get_model()` function in `code/model.py` intentionally does not load weights directly; the platform or training process is expected to handle model loading.

## Suggested workflow

```bash
pip install -r requirments.txt
mkdir -p data
# place your raw sentence data in data/
python code/dataset_conversion.py
python code/train.py
```

This is the minimal flow for preparing data and training the sentence similarity model.
