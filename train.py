import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from data import DataModule
from model import HateModel


def main():
    data = DataModule()
    model = HateModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",filename="hate_model",
        save_top_k=1,monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(accelerator="gpu",
        default_root_dir="logs",
        max_epochs=5,
        fast_dev_run=False,
        logger=TensorBoardLogger("logs/", name="hate", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
