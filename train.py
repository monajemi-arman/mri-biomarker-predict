from model import SegmentationModel
from dataset import Dataset
from config import *
import lightning as L
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from transforms import train_transforms


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=False,
        help="For continuing an interrupted training session, set it to a checkpoint file path.",
    )
    parsed = parser.parse_args()

    train_dataset = Dataset(*dataset_paths, split="train", transforms=train_transforms)
    val_dataset = Dataset(*dataset_paths, split="val")

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=2)

    if parsed.checkpoint:
        print("Loading from checkpoint...")
        model = SegmentationModel.load_from_checkpoint(parsed.checkpoint)
    else:
        model = SegmentationModel()

    trainer = L.Trainer(
        max_epochs=100,
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                save_top_k=3,
                mode="min",
                filename="segment-{epoch:02d}-{val_loss:.4f}",
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, train_loader, val_loader)

    print("Training has finished.")
    print(
        "Please go to 'lightning_logs' folder and choose a model checkpoint from the 'checkpoints' model."
    )
    print(
        "Copy that file into the project root, which is the folder where this file (train.py) is located,"
        " and rename it to checkpoint.ckpt"
    )


if __name__ == "__main__":
    main()
