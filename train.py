import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from data import DataModule
from model import HateModel
from pytorch_lightning.loggers import WandbLogger
import pandas as pd


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["comment_text"]

        outputs = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(outputs.logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"comment_text": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


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
    
    wandb_logger = WandbLogger(project="mlops", entity="shyamsundar007")

    # trainer = pl.Trainer(accelerator="gpu",
    #     default_root_dir="logs",
    #     max_epochs=5,
    #     fast_dev_run=False,
    #     logger=TensorBoardLogger("logs/", name="hate", version=1),
    #     callbacks=[checkpoint_callback, early_stopping_callback],
    # )
    
    
    trainer = pl.Trainer(
        max_epochs=2,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(data), early_stopping_callback],
        log_every_n_steps=10,
        deterministic=True,
        # limit_train_batches=0.25,
        # limit_val_batches=0.25
    )
    
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
