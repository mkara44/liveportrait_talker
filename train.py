import os
import argparse
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.modules.voxceleb2_prep import VoxCeleb2Preprocess
from src.modules.threedmm_prediction import ThreeDMMPrediction
from src.modules.mappingnet_train import MappingNetTrain


def train(cfg, args):
    os.makedirs(os.path.join(cfg.checkpoint_save_path, args.experiment_name), exist_ok=True)
    os.makedirs(cfg.logging_save_path, exist_ok=True)

    monitor_loss = "basic_loss"
    model = MappingNetTrain(**cfg.model_cfg)


    callbacks = [ModelCheckpoint(dirpath=os.path.join(cfg.checkpoint_save_path, args.experiment_name),
                                 filename="epoch{epoch:05d}-val_loss{val/loss:.2f}",
                                 monitor=f"val/{monitor_loss}",
                                 mode="min",
                                 verbose=True,
                                 auto_insert_metric_name=False),
                EarlyStopping(monitor=f"val/{monitor_loss}",
                              mode="min",
                              patience=10,
                              verbose=True)]
    
    logger = CSVLogger(cfg.logging_save_path, args.experiment_name)

    trainer = Trainer(max_epochs=-1,
                      callbacks=callbacks,
                      logger=logger)
    
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--mode", choices=["download", "thredmm", "train"], required=True)
    parser.add_argument("--experiment_name", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    if args.mode == "download":
        download_prep = VoxCeleb2Preprocess(**cfg.download)
        download_prep.download()
        print("Download Completed!")

    elif args.mode == "thredmm":
        threedmm_pred = ThreeDMMPrediction(**cfg.threedmm)
        threedmm_pred.predict()
        print("ThreedMM Prediction Completed!")

    elif args.mode == "train":
        if args.experiment_name is not None:
            train(cfg.train, args)
        else:
            print("experiment_name must be specified for training!")