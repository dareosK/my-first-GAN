from pytorch_lightning.callbacks.base import Callback

class WriteCheckpointLogs(Callback):
    """
    Write final logs to neptune.
    """
    def on_epoch_end(self, trainer, pl_module):
        if isinstance(trainer.logger, list):
            logger = trainer.logger[0]
        else:
            logger = trainer.logger
        if torch.is_tensor(trainer.checkpoint_callback.best_model_score):
            if isinstance(trainer.logger, list):
                trainer.logger[0].run["checkpoint_metric"] = trainer.checkpoint_callback.monitor
                trainer.logger[0].run["checkpoint_value"] = str(trainer.checkpoint_callback.best_model_score.item())
                trainer.logger[0].run["checkpoint_path"] = trainer.checkpoint_callback.best_model_path
            else:
                trainer.logger.run["checkpoint_metric"] = trainer.checkpoint_callback.monitor
                trainer.logger.run["checkpoint_value"] = str(trainer.checkpoint_callback.best_model_score.item())
                trainer.logger.run["checkpoint_path"] = trainer.checkpoint_callback.best_model_path