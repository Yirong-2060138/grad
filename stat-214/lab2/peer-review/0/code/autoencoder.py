import lightning as L
import torch
import torch.nn as nn


class Autoencoder(L.LightningModule):
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config

        input_size = int(n_input_channels * (patch_size**2))
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, embedding_size, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(embedding_size * patch_size * patch_size, embedding_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * patch_size * patch_size),
            nn.ReLU(),
            nn.Unflatten(1, (embedding_size, patch_size, patch_size)),
            nn.ConvTranspose2d(embedding_size, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(16, n_input_channels, kernel_size=3, stride=1, padding=1),  
        )

    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """

        # all the autencoder does is encode then decode the input tensor
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put is the MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.l1_loss(batch, decoded)
        # you can consider other possible loss functions, or add additional terms
        # to this loss.
        # for instance, could it be good to add a term that encourages sparsity
        # in the embedding?

        # log the training loss for experiment tracking purposes
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """

        # we encode then decode.
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        # the loss I have put for validation is the MSE
        # between the input tensor and the decoded tensor
        loss = torch.nn.functional.l1_loss(batch, decoded)
        # log the validation loss for experiment tracking purposes
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self.encoder(x)
