from src import *
from pathlib import Path
import pytorch_lightning as pl
import yaml
import sys
from pytorch_lightning.callbacks import ModelCheckpoint

config = {
    'datamodule': {
        'path': Path('dataset'),
        'batch_size': 25
    },
    'trainer': {
        'max_epochs': 10,
        'logger': None,
        'enable_checkpointing': False,
        'overfit_batches': 0
    }
}


def train(config):
    dm = MNISTDataModule(**config['datamodule'])
    module = MNISTModule(config)
    trainer = pl.Trainer(
        **config['trainer'],
        # callbacks=[
        #     ModelCheckpoint(
        #         monitor="val_loss",
        #         dirpath="checkpoints",
        #         filename="mnist-{epoch:02d}-{val_loss:.2f}",
        #         save_top_k=1,
        #         mode="min",
        #     )
        # ]
    )
    trainer.fit(module, dm)
    trainer.save_checkpoint('final.ckpt')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if config_file:
            with open(config_file, 'r') as stream:
                loaded_config = yaml.safe_load(stream)
            deep_update(config, loaded_config)
    print(config)
    train(config)
