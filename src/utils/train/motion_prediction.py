import yaml
import torch
import contextlib

from src.utils.train.motion_extractor import MotionExtractor


class MotionExtractionPrediction:
    def __init__(self, device, models_config_path, motion_extractor_checkpoint_path):
        self.device = device

        model_config = yaml.load(open(models_config_path, 'r'), Loader=yaml.SafeLoader)
        model_params = model_config["model_params"]["motion_extractor_params"]
        self.motion_extractor = MotionExtractor(**model_params).to(self.device)
        self.motion_extractor.load_state_dict(torch.load(motion_extractor_checkpoint_path, map_location=self.device))
        self.motion_extractor.eval()

    def __call__(self, frame):
        with torch.no_grad(), self.inference_ctx():
            kp_info = self.motion_extractor(frame)

        return kp_info

    def inference_ctx(self):
        if self.device == "mps":
            ctx = contextlib.nullcontext()
        else:
            ctx = torch.autocast(device_type=self.device[:4], dtype=torch.float16, enabled=True)
        return ctx