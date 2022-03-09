import wandb
from pytorch_lightning.callbacks import Callback
import torch
import collections


def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


class WandBCallback(Callback):

    def __init__(self, dl, labels):
        self.dl = dl
        self.labels = labels

    def get_preds(self, pl_module, batch):
        img, label = batch
        with torch.no_grad():
            y_hat = pl_module(img.to(pl_module.device))
        return y_hat, label

    def on_epoch_end(self, trainer, pl_module):
        preds = torch.tensor([], device=pl_module.device)
        y = torch.tensor([])
        pl_module.eval()
        for batch in self.dl:
            y_hat, label = self.get_preds(pl_module, batch)
            preds = torch.cat([preds, y_hat])
            y = torch.cat([y, label])
        probas = torch.sigmoid(preds)
        preds = probas > 0.5
        preds = preds.long()
        preds = torch.nn.functional.one_hot(
            preds, num_classes=len(self.labels))
        trainer.logger.experiment.log({
            "roc_curve": wandb.plot.roc_curve(y.numpy(), preds.cpu().numpy(), labels=self.labels)})
        trainer.logger.experiment.log({"conf_mat": wandb.plot.confusion_matrix(
            probs=preds.cpu().numpy(),
            y_true=y.numpy(),
            class_names=self.labels
        )
        }
        )
        pl_module.train()
