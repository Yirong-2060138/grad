# fine_tuning.py
# Example usage:
# python fine_tuning.py configs/finetuning.yaml checkpoints/exp_0315_all/exp-epoch=034-val_loss=0.1046.ckpt

import numpy as np
import sys
import os
import yaml
import gc
import torch
import lightning as L


from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from autoencoder import Autoencoder
from patchdataset import LabeledPatchDataset
from data import make_data_label 

# AEClassifier 

class AEClassifier(L.LightningModule):
    def __init__(self, encoder, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(8, 1)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x).squeeze()
        y = (y == 1).float()
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x).squeeze()
        y = (y == 1).float()
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Finetuning Begins

if len(sys.argv) < 3:
    print("Usage: python fine_tuning.py <config_path> <checkpoint_path>")
    sys.exit(1)

config_path = sys.argv[1]
checkpoint_path = sys.argv[2]

assert os.path.exists(config_path), f"Config file {config_path} not found"
assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} not found"

config = yaml.safe_load(open(config_path, "r"))
print(f"Using checkpoint: {checkpoint_path} for Fine-Tuning...")


gc.collect()
torch.cuda.empty_cache()

#  patch dataset
print("Generating training patches with labels...")
patches, labels = make_data_label(patch_size=config["data"]["patch_size"])

# train and loss
train_bool = np.random.rand(len(patches)) < 0.8
train_idx = np.where(train_bool)[0]
val_idx = np.where(~train_bool)[0]

train_patches = [patches[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]
val_patches = [patches[i] for i in val_idx]
val_labels = [labels[i] for i in val_idx]

train_dataset = LabeledPatchDataset(train_patches, train_labels)
val_dataset = LabeledPatchDataset(val_patches, val_labels)

dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

# Encoder 
print("Loading pretrained encoder from checkpoint...")
autoencoder = Autoencoder.load_from_checkpoint(checkpoint_path)
encoder = autoencoder.encoder

# Classification Model
print("Creating AEClassifier...")
model = AEClassifier(encoder=encoder, lr=config["optimizer"]["lr"])

checkpoint_callback = ModelCheckpoint(**config["checkpoint"])
early_stopping = EarlyStopping(**config["early_stopping"])
csv_logger = CSVLogger("logs/", name="finetune")

trainer = L.Trainer(
    logger=csv_logger,
    callbacks=[early_stopping, checkpoint_callback],
    **config["trainer"]
)

print("Fine-tuning...")
trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

gc.collect()
torch.cuda.empty_cache()
