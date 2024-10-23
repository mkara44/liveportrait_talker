import torch

from src.models.mappingnet import MappingNet
from src.utils.helper import transform_semantic_target


class Map2LivePortrait:
    def __init__(self, device, semantic_radius, mappingnet_model_path, mappingnet_cfg):
        self.device = device
        self.semantic_radius = semantic_radius

        self.mapping = MappingNet(**mappingnet_cfg)
        self.mapping.load_state_dict(torch.load(mappingnet_model_path, map_location=self.device))
        self.mapping.to(self.device)

    def __call__(self, batch):
        target_semantics_list = self.preprocess(num_frames=batch["num_frames"],
                                                predicted_coeffs=batch["predicted_coeffs"])
        
        mapped_semantics = self.mapping(target_semantics_list)
        batch["mapped_semantics"] = mapped_semantics
        return batch

    def preprocess(self, num_frames, predicted_coeffs):
        target_semantics_list = []
        for frame_idx in range(num_frames):
            target_semantics = transform_semantic_target(predicted_coeffs, frame_idx, self.semantic_radius)
            target_semantics_list.append(target_semantics)
        return torch.stack(target_semantics_list, dim=0)
