import torch
import random
import pickle as pkl
from collections import OrderedDict
from typing import Union
import numpy as np
from scipy.spatial import ConvexHull
from pykalman import KalmanFilter

from src.models.appearance_feature_extractor import AppearanceFeatureExtractor
from src.models.motion_extractor import MotionExtractor
from src.models.warping_network import WarpingNetwork
from src.models.spade_generator import SPADEDecoder
from src.models.stitching_retargeting_network import StitchingRetargetingNetwork


def retarget_pupil(delta_new, pupil_x, pupil_y):
    # open for improvements
    if 0 < pupil_x:
        delta_new[:, 11, 0] += pupil_x * 0.0007
        delta_new[:, 15, 0] += pupil_x * 0.001
    else:
        delta_new[:, 11, 0] += pupil_x * 0.001
        delta_new[:, 15, 0] += pupil_x * 0.0007

    delta_new[:, 11, 1] += pupil_y * -0.001
    delta_new[:, 15, 1] += pupil_y * -0.001
    return delta_new


def get_reference_frames(ref_R_list, n_frames, ref_frames_from_zero):
    if ref_frames_from_zero:
        start_frame_number = 0
    else:
        start_frame_number = random.randint(0, len(ref_R_list)-n_frames-1)
    return ref_R_list[start_frame_number:start_frame_number+n_frames]

def smooth(x_d_lst, shape, device, observation_variance=3e-7, process_variance=1e-5):
    x_d_lst_reshape = [x.reshape(-1) for x in x_d_lst]
    x_d_stacked = np.vstack(x_d_lst_reshape)
    kf = KalmanFilter(
        initial_state_mean=x_d_stacked[0],
        n_dim_obs=x_d_stacked.shape[1],
        transition_covariance=process_variance * np.eye(x_d_stacked.shape[1]),
        observation_covariance=observation_variance * np.eye(x_d_stacked.shape[1])
    )
    smoothed_state_means, _ = kf.smooth(x_d_stacked)
    x_d_lst_smooth = [torch.tensor(state_mean.reshape(shape[-2:]), dtype=torch.float32, device=device) for state_mean in smoothed_state_means]
    return x_d_lst_smooth

def tensor_to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """transform torch.Tensor into numpy.ndarray"""
    if isinstance(data, torch.Tensor):
        return data.data.cpu().numpy()
    return data

def calc_motion_multiplier(
    kp_source: Union[np.ndarray, torch.Tensor],
    kp_driving_initial: Union[np.ndarray, torch.Tensor]
) -> float:
    """calculate motion_multiplier based on the source image and the first driving frame"""
    kp_source_np = tensor_to_numpy(kp_source)
    kp_driving_initial_np = tensor_to_numpy(kp_driving_initial)

    source_area = ConvexHull(kp_source_np.squeeze(0)).volume
    driving_area = ConvexHull(kp_driving_initial_np.squeeze(0)).volume
    motion_multiplier = np.sqrt(source_area) / np.sqrt(driving_area)
    # motion_multiplier = np.cbrt(source_area) / np.cbrt(driving_area)

    return motion_multiplier


def remove_ddp_dumplicate_key(state_dict):
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new

def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':
        model = AppearanceFeatureExtractor(**model_params).to(device)
    elif model_type == 'motion_extractor':
        model = MotionExtractor(**model_params).to(device)
    elif model_type == 'warping_module':
        model = WarpingNetwork(**model_params).to(device)
    elif model_type == 'spade_generator':
        model = SPADEDecoder(**model_params).to(device)
    elif model_type == 'stitching_retargeting_module':
        # Special handling for stitching and retargeting module
        config = model_config['model_params']['stitching_retargeting_module_params']
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        stitcher.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
        stitcher = stitcher.to(device)
        stitcher.eval()

        retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        retargetor_lip.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_mouth']))
        retargetor_lip = retargetor_lip.to(device)
        retargetor_lip.eval()

        retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))
        retargetor_eye.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_eye']))
        retargetor_eye = retargetor_eye.to(device)
        retargetor_eye.eval()

        return {
            'stitching': stitcher,
            'lip': retargetor_lip,
            'eye': retargetor_eye
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
    model.eval()
    return model

def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat

def load_lip_array(lip_array_pkl):
    with open(lip_array_pkl, 'rb') as f:
        return pkl.load(f)