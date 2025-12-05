from numpy import full
import json
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Load ATIS (Rasa format)
TRAIN_PATH = "ATIS_dataset/data/standard_format/rasa/train.json"
TEST_PATH  = "ATIS_dataset/data/standard_format/rasa/test.json"

with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_j = json.load(f)
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_j = json.load(f)

train_examples = train_j["rasa_nlu_data"]["common_examples"]
test_examples  = test_j["rasa_nlu_data"]["common_examples"]

def rasa_examples_to_df(examples):
    rows = []
    for ex in examples:
        text = ex.get("text", "")
        intent = ex.get("intent", None)
        entities = ex.get("entities", [])
        entities_str = ";".join(f"{e.get('entity')}={e.get('value')}" for e in entities)
        rows.append({"text": text, "intent": intent, "entities": entities_str})
    return pd.DataFrame(rows)

train_df = rasa_examples_to_df(train_examples)
test_df  = rasa_examples_to_df(test_examples)

# 3. Normalize intents
def normalize_intent(intent: str) -> str:
    mapping = {
        "airfare+flight": "flight+airfare",
        "flight_no+airline": "airline+flight_no",
        "cheapest": "airfare",
        "day_name": "flight_time",
    }
    intent = mapping.get(intent, intent)
    return intent

train_df["intent"] = train_df["intent"].apply(normalize_intent)
test_df["intent"]  = test_df["intent"].apply(normalize_intent)

full_df= pd.concat([train_df,test_df], ignore_index=True)
counts= full_df['intent'].value_counts()
rare_intents = counts[counts <= 5]
rare = counts[counts <= 5].index.tolist()
print(rare_intents)
print(rare)

train_df = train_df[~train_df['intent'].isin(rare)]
test_df = test_df[~test_df['intent'].isin(rare)]

# 4. Build intent → index mapping
all_intents = sorted(set(train_df["intent"]) | set(test_df["intent"]))
intent2idx = {intent: i for i, intent in enumerate(all_intents)}
idx2intent = {i: intent for intent, i in intent2idx.items()}


train_df["label"] = train_df["intent"].map(intent2idx)
test_df["label"]  = test_df["intent"].map(intent2idx)

# 5. Train/validation split
train_sub, val_sub = train_test_split(train_df, test_size=0.2, shuffle=True,stratify=train_df["intent"],random_state=42)


# 6. Save to CSV
OUTPUT_DIR = "data/"
train_sub.to_csv(OUTPUT_DIR + "atis_train.csv", index=False)
val_sub.to_csv(OUTPUT_DIR + "atis_val.csv", index=False)
test_df.to_csv(OUTPUT_DIR + "atis_test.csv", index=False)


# 7. Print essential info
def print_dataset_info(name, df):
    print(f"{name} examples: {len(df)}")
    print(f"Unique intents: {df['intent'].nunique()}\n")

print_dataset_info("Train", train_sub)
print_dataset_info("Validation", val_sub)
print_dataset_info("Test", test_df)

print("Sample training rows:")
print(train_sub[["text", "intent", "label"]].head(3).to_string(index=False), "\n")

print("Intent → label mapping (example):")
for intent, idx in list(intent2idx.items()):
    print(f"{intent:25s} {idx}")

