import torch
import torch.nn as nn
from torch.optim import AdamW


# --- MLM MASKING ---
# --- MLM MASKING ---
def mask_tokens(input_ids, vocab_size, mask_token_id, pad_token_id, mlm_prob=0.15 ):
    '''
    TODO: Implement MLM masking
    Args:
        input_ids: Input IDs
        vocab_size: Vocabulary size
        mask_token_id: Mask token ID
        pad_token_id: Pad token ID
        mlm_prob: Probability of masking
    '''
    labels = input_ids.clone()
    # Create mask for padding tokens
    padding_mask = input_ids == pad_token_id
    # Sample tokens for MLM
    probability_matrix = torch.full(labels.shape, mlm_prob, device=input_ids.device)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    # Only compute loss on masked tokens
    labels[~masked_indices] = -100
    # 80% replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=input_ids.device)).bool() & masked_indices
    input_ids[indices_replaced] = mask_token_id
    # 10% replace with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
    input_ids[indices_random] = random_tokens[indices_random]
    # The rest 10%: keep original
    return input_ids, labels

def train_bert(
    model,
    train_dataloader,
    tokenizer,
    val_dataloader=None,
    epochs=3,
    lr=5e-4,
    device='cuda'
):
    '''
    Train BERT-style encoder with MLM.
    Returns the trained model, list of train losses, list of validation losses.
    '''
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs+1):
        # --- Training ---
        model.train()
        total_train = 0.0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device).squeeze(1).squeeze(1)

            # Mask tokens and prepare labels
            masked_input_ids, labels = mask_tokens(
                input_ids.clone(),
                vocab_size=tokenizer.vocab_size,
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id,
                mlm_prob=0.15
            )

            optimizer.zero_grad()
            logits = model(masked_input_ids, token_type_ids, attention_mask)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss.backward()
            optimizer.step()

            total_train += loss.item()

        avg_train = total_train / len(train_dataloader)
        train_losses.append(avg_train)

        # --- Validation ---
        if val_dataloader is not None:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device).squeeze(1).squeeze(1)

                    masked_input_ids, labels = mask_tokens(
                        input_ids.clone(),
                        vocab_size=tokenizer.vocab_size,
                        mask_token_id=tokenizer.mask_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        mlm_prob=0.15
                    )

                    logits = model(masked_input_ids, token_type_ids, attention_mask)
                    vloss = loss_fn(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                    total_val += vloss.item()

            avg_val = total_val / len(val_dataloader)
            val_losses.append(avg_val)
            print(f"Epoch {epoch}/{epochs} — train: {avg_train:.4f}, val: {avg_val:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} — train: {avg_train:.4f}")

    return model, train_losses, val_losses
