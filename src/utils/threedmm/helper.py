import torch
import numpy as np
from facexlib.utils import load_file_from_url

from src.utils.threedmm.arch import FAN


def voxceleb_crop_frame(frame, coords, scale_crop=1.):
    x1_hat = int(float(coords[0]) * frame.shape[1])
    y1_hat = int(float(coords[1]) * frame.shape[0])
    x2_hat = int(float(coords[2]) * frame.shape[1]) + x1_hat
    y2_hat = int(float(coords[3]) * frame.shape[0]) + y1_hat

    w = x2_hat - x1_hat
    h = y2_hat - y1_hat		
    cx = int(x1_hat + w/2)
    cy = int(y1_hat + h/2)

    w_hat = int(w * scale_crop) 
    h_hat = int(h * scale_crop) 

    x1_hat = cx - int(w_hat/2)
    if x1_hat < 0:
        x1_hat = 0

    y1_hat = cy - int(h_hat/2)
    if y1_hat < 0:
        y1_hat = 0

    x2_hat = x1_hat + w_hat
    y2_hat = y1_hat + h_hat

    if x2_hat > frame.shape[1]:
        x2_hat = frame.shape[1]
    if y2_hat > frame.shape[0]:
        y2_hat = frame.shape[0]

    if (y2_hat - y1_hat) > 20 and (x2_hat - x1_hat) > 20:
        cropped_frame = frame[y1_hat:y2_hat, x1_hat:x2_hat, :]	
    else:
        cropped_frame = frame

    return cropped_frame, [x1_hat, y1_hat, x2_hat, y2_hat]

def calculate_angle(eye_corner, nose_tip):
    diff = eye_corner - nose_tip

    dot_product = np.dot(diff, np.array([1, 0]))
    magnitude = np.linalg.norm(diff)

    if np.isclose(magnitude, 0):
        return 0

    angle_radians = np.arccos(dot_product / magnitude)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def check_frontality_by_angle(landmarks, threshold):
    left_eye_angle = calculate_angle(landmarks[64, :2],
                                     landmarks[54, :2])
    right_eye_angle = calculate_angle(landmarks[68, :2],
                                      landmarks[54, :2])
    
    angle = left_eye_angle + right_eye_angle
    return True if abs(180 - angle) < threshold else False

def init_alignment_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'awing_fan':
        model = FAN(num_modules=4, num_landmarks=98, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'], strict=True)
    model.eval()
    model = model.to(device)
    return model

def split_coeff(coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

def load_x_from_safetensor(checkpoint, key):
    x_generator = {}
    for k,v in checkpoint.items():
        if key in k:
            x_generator[k.replace(key+'.', '')] = v
    return x_generator