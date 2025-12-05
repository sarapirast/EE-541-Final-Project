import re
import random
import pandas as pd

# Tokenization
MONTHS = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december"
]

def tokenize(text):
    """
    Cleans and tokenizes ATIS queries following your rubric:
    - lowercase
    - normalize punctuation (keep ? ! .)
    - keep airport codes (ORD, BOS)
    - keep numbers (flight numbers matter)
    - remove standalone months
    - remove weird punctuation except <> tokens
    """
    text = text.lower().strip()

    # remove months as useless tokens
    for m in MONTHS:
        text = re.sub(rf"\b{m}\b", "", text)

    # normalize apostrophes inside words
    text = re.sub(r"(\w)'(\w)", r"\1\2", text)

    # keep punctuation that affects meaning, remove the rest
    text = re.sub(r"[^\w\s<>?!\.]", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text.split()


# Synonym dictionary for augmentation
SYNONYMS = {
    "airfare": ["airfares"],
    "airfares": ["airfare"],
    "airline": ["airlines"],
    "airlines": ["airline"],
    "arrive": ["arrives", "arriving"],
    "depart": ["departing", "departs"],
    "flight": ["flights"],
    "trip": ["trips"],
    "fare": ["fares"],
    "price": ["prices"],
    "meal": ["meals"],
}


# Token-level augmentation
def augment_tokens(tokens, ins_p=0.05, del_p=0.05, rep_p=0.05):
    """
    Applies 3 augmentations:
    - deletion
    - synonym replacement
    - insertion (duplicate)
    """
    new_tokens = []

    for tok in tokens:

        # delete
        if len(tokens) > 3 and random.random() < del_p:
            continue

        # synonym replace
        if tok in SYNONYMS and random.random() < rep_p:
            tok = random.choice(SYNONYMS[tok])

        new_tokens.append(tok)

        # insertion
        if random.random() < ins_p:
            new_tokens.append(tok)

    return new_tokens


def augment_text(text, ins_p=0.05, del_p=0.05, rep_p=0.05):
    tokens = tokenize(text)
    tokens = augment_tokens(tokens, ins_p, del_p, rep_p)
    return " ".join(tokens)


# DataFrame augmentation
def augment_dataframe(df, frac=1.0, ins_p=0.05, del_p=0.05, rep_p=0.05):
    """
    df must contain columns: ['text', 'intent', 'label']
    """
    random.seed(42)

    n = int(len(df) * frac)
    sampled = df.sample(n, random_state=42)

    augmented_rows = []

    for _, row in sampled.iterrows():
        text = row["text"]
        new_text = augment_text(text, ins_p, del_p, rep_p)
        augmented_rows.append({
            "text": new_text,
            "intent": row["intent"],
            "label": row["label"]
        })

    return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)


# Convert one sentence → padded indices
def token_and_augment(sentence, word2idx, augment=False, max_len=50):
    tokens = tokenize(sentence)
    if augment:
        tokens = augment_tokens(tokens)

    ids = [word2idx.get(tok, word2idx["<unk>"]) for tok in tokens]

    # pad or truncate
    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


# Wrapper for vocab building
def simple_tokenize(text):
    return tokenize(text)
