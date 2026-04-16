import lightning as L
import torch
import torch.nn as nn
from monai.networks.nets import UNet

class SegmentationModel(L.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, learning_rate=1e-3):
        super().__init__()
        self.model = UNet(
            spatial_dims=3,          # 3D convolutions
            in_channels=in_channels,  # number of channels in your image
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),  # encoder/decoder channels
            strides=(2, 2, 2, 2),    # downsampling
            num_res_units=2,
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, mask = batch  # x: (B, C, Z, H, W), mask: (B, 1, Z, H, W)
        pred = self(x).squeeze(1)
        assert pred.shape == mask.shape, f"{pred.shape} != {mask.shape}"
        loss = self.loss(pred, mask)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)