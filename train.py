from model import SegmentationModel
from dataset import Dataset
from config import *
import lightning as L
from torch.utils.data import DataLoader

def main():
    train_dataset = Dataset(*dataset_paths, split="train")
    val_dataset = Dataset(*dataset_paths, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=2
    )

    model = SegmentationModel()

    trainer = L.Trainer(
        max_epochs=50,
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="train_loss",
                save_top_k=3,
                mode="min",
                filename="segment-{epoch:02d}-{train_loss:.4f}"
            )
        ]
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()