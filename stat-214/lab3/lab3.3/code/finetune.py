
# Imports & Configuration
import os, sys, pickle, random, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType

# Paths
RAW_TEXT = "../data/raw_text.pkl"
SUBJ_DIR  = "../data/subject2"
SAVE_DIR  = "./bert-fmri-finetuned"

# Hyperparameters
BATCH_SIZE = 8
ACCUM_STEPS = 2
EPOCHS = 5
LEARNING_RATE = 1e-6
MAX_LEN = 128
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3
USE_L1_LOSS = True
USE_BOTTLENECK = True
BOTTLENECK_SIZE = 512
GRAD_CLIP_VAL = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

# Load Data
repo_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
code_path = os.path.join(repo_root, "code")
if os.path.isdir(code_path):
    sys.path.insert(0, code_path)

with open(RAW_TEXT, "rb") as f:
    raw_texts = pickle.load(f)

stories = [s for s in raw_texts if os.path.exists(f"{SUBJ_DIR}/{s}.npy")]
print("Total stories:", len(stories))

# Compute y_mean, y_std
def get_trimmed_Y(seq, subj_dir, sid):
    Y = np.load(f"{subj_dir}/{sid}.npy", mmap_mode="r")
    TR = float(np.mean(np.diff(seq.tr_times)))
    start = int(math.ceil(5.0 / TR))
    end = -int(math.ceil(10.0/ TR))
    return Y[start:end]

train_ids, test_ids = train_test_split(stories, test_size=0.2, random_state=42)

sum_y = sum_y2 = count = None
for sid in train_ids:
    seq = raw_texts[sid]
    Y_trim = get_trimmed_Y(seq, SUBJ_DIR, sid)
    Y_trim = np.nan_to_num(Y_trim)
    if sum_y is None:
        V = Y_trim.shape[1]
        sum_y = np.zeros(V)
        sum_y2 = np.zeros(V)
    sum_y += Y_trim.sum(axis=0)
    sum_y2 += (Y_trim**2).sum(axis=0)
    count += Y_trim.shape[0]

y_mean = sum_y / count
y_std = np.sqrt(sum_y2/count - y_mean**2)
y_std = np.maximum(y_std, 0.1)

# IterableDataset 
class FMRIIterableDataset(IterableDataset):
    def __init__(self, raw_texts, subj_dir, story_ids, tokenizer, max_len, y_mean, y_std):
        self.raw_texts = raw_texts
        self.subj_dir = subj_dir
        self.story_ids = story_ids
        self.tok = tokenizer
        self.max_len = max_len
        self.y_mean = torch.tensor(y_mean, dtype=torch.float)
        self.y_std = torch.tensor(y_std,  dtype=torch.float)

    def __iter__(self):
        random.shuffle(self.story_ids)
        for sid in self.story_ids:
            seq = self.raw_texts[sid]
            Y = np.load(f"{self.subj_dir}/{sid}.npy", mmap_mode="r")
            Y = np.nan_to_num(Y)
            TR = float(np.mean(np.diff(seq.tr_times)))
            start = int(math.ceil(5.0 / TR))
            end = -int(math.ceil(10.0 / TR))
            for t, y_row in zip(seq.tr_times[start:end], Y[start:end]):
                mask = (seq.data_times >= t-TR/2) & (seq.data_times < t+TR/2)
                words = [seq.data[i] for i in np.where(mask)[0]]
                text = " ".join(words) or "[PAD]"
                enc = self.tok(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
                y = (torch.tensor(y_row, dtype=torch.float) - self.y_mean) / self.y_std
                yield {
                    "input_ids": enc.input_ids.squeeze(0),
                    "attention_mask": enc.attention_mask.squeeze(0),
                    "y": y
                }

# DataLoaders 
tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
train_loader = DataLoader(FMRIIterableDataset(raw_texts, SUBJ_DIR, train_ids, tokenizer, MAX_LEN, y_mean, y_std), batch_size=BATCH_SIZE)
test_loader = DataLoader(FMRIIterableDataset(raw_texts, SUBJ_DIR, test_ids, tokenizer, MAX_LEN, y_mean, y_std), batch_size=BATCH_SIZE)

class BertVoxel(nn.Module):
    def __init__(self, V, dropout=DROPOUT_RATE):
        super().__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        H = self.bert.config.hidden_size
        self.bottleneck = nn.Linear(H, BOTTLENECK_SIZE)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(BOTTLENECK_SIZE, V)

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask, return_dict=True)
        x = self.bottleneck(out.pooler_output)
        x = self.activation(x)
        x = self.drop(x)
        return self.head(x)

batch0 = next(iter(train_loader))
V = batch0["y"].shape[1]
model = BertVoxel(V).to(DEVICE)

# === Optimizer ===
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "lr": LEARNING_RATE, "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped, betas=(0.9, 0.999), eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=500)
scaler = amp.GradScaler()
loss_fn = nn.L1Loss() if USE_L1_LOSS else nn.MSELoss()

print(" Starting training...")
best_loss = float("inf")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    step = 0
    for batch in train_loader:
        step += 1
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        y = batch["y"].to(DEVICE)

        if torch.isnan(y).any():
            continue

        with amp.autocast(device_type=DEVICE):
            preds = model(ids, mask)
            loss = loss_fn(preds, y) / ACCUM_STEPS
            if torch.isnan(loss).any():
                continue

        scaler.scale(loss).backward()

        if step % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VAL)
            if math.isinf(grad_norm) or math.isnan(grad_norm):
                optimizer.zero_grad()
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * ACCUM_STEPS

    avg_loss = total_loss / step
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, val_batch in enumerate(test_loader):
            if i > 100: break
            ids = val_batch["input_ids"].to(DEVICE)
            mask = val_batch["attention_mask"].to(DEVICE)
            y = val_batch["y"].to(DEVICE)
            preds = model(ids, mask)
            val_loss += loss_fn(preds, y).item()
    val_loss /= (i+1)
    print(f"Validation Loss = {val_loss:.6f}")
    if val_loss < best_loss:
        best_loss = val_loss
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.bert.save_pretrained(SAVE_DIR)
        torch.save(model.head.state_dict(), os.path.join(SAVE_DIR, "head_state.pt"))
        if hasattr(model, 'bottleneck'):
            torch.save(model.bottleneck.state_dict(), os.path.join(SAVE_DIR, "bottleneck_state.pt"))
