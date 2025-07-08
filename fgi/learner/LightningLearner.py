from abc import ABC
from . import Learner
from lightning import LightningModule

# Dùng Lightning để hỗ trợ học
class LightningLearner(Learner, LightningModule, ABC):
    """
    Triển khai bản dựng học dùng thư viện hỗ trợ lightning.
    (Có hỗ trợ callback)
    """
    def __init__(self, problem, *args, **kwargs):
        super().__init__(problem, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return self._problem.forward(x, *args, **kwargs)
    
    def configure_optimizers(self):
        return self._optimizer
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        y_hat = self(x)
        overall = self._aggerate_loss(y_hat, y)
        self.log_dict(overall, *args, **kwargs)
        self.log("train_loss", overall["total_loss"], on_step=True, on_epoch=True, prog_bar=True)
        return overall["total_loss"]
    
    def validation_step(self, batch, batch_idx ,*args, **kwargs):
        x, y = batch
        y_hat = self(x)
        overall = self._aggerate_loss(y_hat, y)
        self.log_dict(overall, *args, **kwargs)
        self.log("val_loss", overall["total_loss"], on_step=True, on_epoch=True, prog_bar=True)
        return overall["total_loss"]