# extract_bert_embeddings.py

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel
from preprocessing import lanczosinterp2D, make_delayed

# === Configurations ===
RAW_TEXT_PATH = "../../../shared/data/raw_text.pkl"
OUTPUT_DIR = "../results/embeddings/bert_XY"
SUBJECT_DIR = "../../../shared/data/subject2"
CHUNK_SIZE = 128
STRIDE = 64
HIDDEN_SIZE = 768

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load tokenizer and model ===
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# === Embedding extraction function ===
def extract_bert_embeddings(seq, model, tokenizer, chunk_size=CHUNK_SIZE, stride=STRIDE, hidden_size=HIDDEN_SIZE):
    device = next(model.parameters()).device
    text_words = seq.data
    total_words = len(text_words)
    word_embeddings = [None] * total_words

    for start in range(0, total_words, stride):
        chunk_words = text_words[start:start + chunk_size]
        tokens = tokenizer(
            chunk_words,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=chunk_size,
            return_attention_mask=True,
            return_token_type_ids=True
        )

        input_ids = tokens["input_ids"]
        token_type_ids = tokens["token_type_ids"]
        attention_mask = tokens["attention_mask"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state.squeeze(0).cpu()

        word_ids = tokens.word_ids(batch_index=0)
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            abs_word_idx = start + word_idx
            if abs_word_idx >= total_words:
                continue
            if word_embeddings[abs_word_idx] is None:
                word_embeddings[abs_word_idx] = []
            word_embeddings[abs_word_idx].append(hidden_states[token_idx])

    for i in range(total_words):
        if word_embeddings[i] is None:
            word_embeddings[i] = torch.zeros(hidden_size)
        else:
            word_embeddings[i] = torch.stack(word_embeddings[i]).mean(0)

    return torch.stack(word_embeddings).numpy()

# === Run embedding extraction ===
with open(RAW_TEXT_PATH, "rb") as f:
    raw_texts = pickle.load(f)

for story_id, seq in tqdm(raw_texts.items(), desc="Extracting BERT embeddings"):
    try:
        emb = extract_bert_embeddings(seq, model, tokenizer)
        X_interp = lanczosinterp2D(emb, seq.data_times, seq.tr_times)

        # Trim the beginning and end
        TR = np.mean(np.diff(seq.tr_times))
        n_skip_start = int(np.ceil(5 / TR))
        n_skip_end = int(np.ceil(10 / TR))
        X_interp = X_interp[n_skip_start:-n_skip_end]

        # Create delayed features
        X_delayed = make_delayed(X_interp, [1, 2, 3, 4])

        # Load and delay-align response
        y_path = os.path.join(SUBJECT_DIR, f"{story_id}.npy")
        Y = np.load(y_path)
        Y = Y[n_skip_start:-n_skip_end]
        Y_delayed = make_delayed(Y, [1, 2, 3, 4])

        # Save
        np.save(os.path.join(OUTPUT_DIR, f"{story_id}_X.npy"), X_delayed)
        np.save(os.path.join(OUTPUT_DIR, f"{story_id}_Y.npy"), Y_delayed)

    except Exception as e:
        print(f"⚠️ Skipping {story_id}: {e}")

print("Done: Embeddings saved to", OUTPUT_DIR)
