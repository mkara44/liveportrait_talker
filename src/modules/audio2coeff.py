import torch
from src.models.audio2exp import Audio2Exp

class Audio2Coeff():
    def __init__(self, device, sadtalker_checkpoint_path):
        self.device = device

        self.audio2exp_model = Audio2Exp(device=self.device,
                                         sadtalker_checkpoint_path=sadtalker_checkpoint_path)
        self.audio2exp_model = self.audio2exp_model.to(self.device)
        for param in self.audio2exp_model.parameters():
            param.requires_grad = False
        self.audio2exp_model.eval()

    def __call__(self, batch):
        with torch.no_grad():
            results_dict_exp= self.audio2exp_model(batch)

        batch["predicted_expressions"] = results_dict_exp[0]
        return batch
