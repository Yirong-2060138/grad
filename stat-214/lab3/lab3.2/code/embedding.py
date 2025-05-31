## python3 embedding_encoder.py
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast

from encoder import Encoder
from preprocessing import lanczosinterp2D, make_delayed


def extract_embeddings(seq, model, tokenizer, chunk_size=128, stride=64, hidden_size=512):
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

        input_ids = tokens["input_ids"].to(device)
        token_type_ids = tokens["token_type_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            hidden_states = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_hidden=True
            )[0]

        hidden_states = hidden_states.squeeze(0).cpu()
        word_ids = tokens.word_ids(batch_index=0)

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            absolute_word_idx = start + word_idx
            if absolute_word_idx >= total_words:
                continue
            if word_embeddings[absolute_word_idx] is None:
                word_embeddings[absolute_word_idx] = []
            word_embeddings[absolute_word_idx].append(hidden_states[token_idx])

    for i in range(total_words):
        if word_embeddings[i] is None:
            word_embeddings[i] = torch.zeros(hidden_size)
        else:
            word_embeddings[i] = torch.stack(word_embeddings[i]).mean(0)

    return torch.stack(word_embeddings).numpy()


if __name__ == "__main__":

    # Paths
    model_path = "saved_models/encoder_lr0.0001_layers6_hs512.pt"
    raw_text_path = "../data/raw_text.pkl"
    output_dir = "../results/embeddings/encoder"
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = Encoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=512,
        num_heads=4,
        num_layers=6,
        intermediate_size=1024,
        max_len=128
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Model token embedding shape:", model.token_emb.weight.shape)

    # Load raw text
    with open(raw_text_path, "rb") as f:
        raw_texts = pickle.load(f)

    # Extract and save embeddings
    for story_id, seq in tqdm(raw_texts.items(), desc="Extracting encoder embeddings"):
        try:
            emb = extract_embeddings(seq, model, tokenizer)
            X_interp = lanczosinterp2D(emb, oldtime=seq.data_times, newtime=seq.tr_times)
            TR = np.mean(np.diff(seq.tr_times))
            n_skip_start = int(np.ceil(5 / TR))
            n_skip_end = int(np.ceil(10 / TR))
            X_interp = X_interp[n_skip_start:-n_skip_end]
            X_delayed = make_delayed(X_interp, [1, 2, 3, 4])
            np.save(os.path.join(output_dir, f"{story_id}.npy"), X_delayed)
        except Exception as e:
            print(f"⚠️ Skipping {story_id}: {e}")

    print("Encoder embedding extraction done. Saved to", output_dir)
