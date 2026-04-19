import lightning as L
import torch
import torch.nn as nn
from monai.networks.nets import UNet


class SegmentationModel(L.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, learning_rate=1e-3):
        super().__init__()

        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        self.bottleneck = lambda model: model.model[1].submodule[1].submodule[1].submodule[1].submodule

        self.loss = nn.BCEWithLogitsLoss()
        self.lr = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, mask = batch
        mask = mask.unsqueeze(1).float()

        pred = self(x)
        loss = self.loss(pred, mask)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        mask = mask.unsqueeze(1).float()

        pred = self(x)
        loss = self.loss(pred, mask)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,  # halve LR when stuck
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
