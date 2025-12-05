import pandas as pd
from collections import Counter
import json
from token_and_augment import simple_tokenize


def build_vocab(csv_path, min_freq=1, save_json=False, json_path="vocab.json"):
    # Load CSV containing at least a "text" column
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise RuntimeError("CSV must contain 'text' column")

    # Counter to accumulate token frequencies across all rows
    counter = Counter()

    for text in df["text"]:
        # Basic tokenization for each text entry
        tokens = simple_tokenize(text)
        counter.update(tokens)

    # Start vocab with special tokens
    vocab = ["<pad>", "<unk>"]

    # Add tokens meeting minimum frequency threshold
    for w, freq in counter.items():
        if freq >= min_freq:
            vocab.append(w)

    # Create lookup tables
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    print(f"[build_vocab] vocab size = {len(vocab)} (min_freq={min_freq})")

    # Optionally save the vocab as JSON
    if save_json:
        with open(json_path, "w") as f:
            json.dump({"word2idx": word2idx, "idx2word": idx2word}, f, indent=2)

    return word2idx, idx2word