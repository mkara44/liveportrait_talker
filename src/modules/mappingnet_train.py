import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.mappingnet import MappingNet
from src.utils.train.dataset import VoxCeleb2Dataset
from src.utils.train.motion_prediction import MotionExtractionPrediction
from src.utils.train.basic_loss import L1Loss


class MappingNetTrain(pl.LightningModule):
    def __init__(self, device, batch_size, num_workers, learning_rate,
                 motionextraction_cfg, mappingnet_cfg, loss_cfg,
                 train_dataset_folder_path, val_dataset_folder_path, dataset_cfg):
        super().__init__()

        self.target_device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.train_dataset_folder_path = train_dataset_folder_path
        self.val_dataset_folder_path = val_dataset_folder_path
        self.dataset_cfg = dataset_cfg

        self.motionextraction_prediction = MotionExtractionPrediction(device=self.target_device,
                                                                      **motionextraction_cfg)

        self.mappingnet = MappingNet(**mappingnet_cfg)
        self.loss = L1Loss(**loss_cfg)

    def forward(self, semantics):
        predicted_motions = self.mappingnet(semantics)
        return predicted_motions

    def training_step(self, batch):
        motions = self.motionextraction_prediction(batch["cropped_face"])
        predicted_motions = self(batch["threedmm_list"])
        loss, log = self.loss(motions, predicted_motions, split_name="train")
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        motions = self.motionextraction_prediction(batch["cropped_face"])
        predicted_motions = self(batch["threedmm_list"])
        _, log = self.loss(motions, predicted_motions, split_name="val")
        self.log_dict(log)

    def configure_optimizers(self):
        opt = Adam(self.mappingnet.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self):
        dataset = VoxCeleb2Dataset(dataset_folder_path=self.train_dataset_folder_path,
                                   **self.dataset_cfg)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        dataset = VoxCeleb2Dataset(dataset_folder_path=self.val_dataset_folder_path,
                                   **self.dataset_cfg)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
