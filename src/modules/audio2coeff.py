import torch
import numpy as np
from scipy.signal import savgol_filter

from src.models.audio2exp import Audio2Exp
from src.models.audio2pose import Audio2Pose


class SadtalkerAudio2Coeff():
    def __init__(self, device, sadtalker_checkpoint_path, flag_exp, flag_pose, audio2pose_cfg):
        self.device = device
        self.flag_exp = flag_exp
        self.flag_pose = flag_pose

        if self.flag_exp:
            self.audio2exp_model = Audio2Exp(device=self.device,
                                             sadtalker_checkpoint_path=sadtalker_checkpoint_path)
            self.audio2exp_model = self.audio2exp_model.to(self.device)
            for param in self.audio2exp_model.parameters():
                param.requires_grad = False
            self.audio2exp_model.eval()

        if self.flag_pose:
            self.audio2pose_model = Audio2Pose(device=self.device,
                                               sadtalker_checkpoint_path=sadtalker_checkpoint_path,
                                               **audio2pose_cfg)
            self.audio2pose_model = self.audio2pose_model.to(device)
            for param in self.audio2pose_model.parameters():
                param.requires_grad = False 
            self.audio2pose_model.eval()

    def __call__(self, batch):
        with torch.no_grad():
            exp_pred = self.audio2exp_model(batch)

            if batch["ref_head_pose_path"] is None or batch["source_type"] != "video":
                pose_pred = self.audio2pose_model(batch)

                pose_len = pose_pred.shape[1]
                if pose_len<13: 
                    pose_len = int((pose_len-1)/2)*2+1
                    pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), pose_len, 2, axis=1)).to(self.device)
                else:
                    pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), 13, 2, axis=1)).to(self.device) 
         
            elif batch["source_type"] == "video":
                pose_pred = batch["source_coeff"][:, :, : 64]
            
            else:
                pose_pred = batch["ref_head_pose_coeff"].repeat(batch["num_frames"], 1).unsqueeze(0)

        pose_pred = torch.zeros_like(pose_pred)
        coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)            #bs T 70
        batch["predicted_coeffs"] = coeffs_pred[0]
        return batch
