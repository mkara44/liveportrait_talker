import torch
import random
import numpy as np
from facexlib.utils import load_file_from_url

from src.utils.preprocess.arch import FAN


def calculate_distance_ratio(lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int, eps: float = 1e-6) -> np.ndarray:
    return (np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) /
            (np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps))

def calc_eye_close_ratio(lmk, target_eye_ratio=None):
    lefteye_close_ratio = calculate_distance_ratio(lmk, 62, 66, 60, 64)
    righteye_close_ratio = calculate_distance_ratio(lmk, 70, 74, 68, 72)

    if target_eye_ratio is not None:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio, target_eye_ratio], axis=1)
    else:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio], axis=1)

def check_source_type(input_path):
    input_type = None

    if input_path.endswith("jpg") or input_path.endswith("png"):
        input_type = "image"
    elif input_path.endswith("mp4"):
        input_type = "video"

    return input_type

def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames

def generate_blink_seq(num_frames):
    ratio = np.zeros((num_frames,1))
    frame_id = 0
    while frame_id in range(num_frames):
        start = 80
        if frame_id+start+9<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+9, 0] = [0.5,0.6,0.7,0.9,1, 0.9, 0.7,0.6,0.5]
            frame_id = frame_id+start+9
        else:
            break
    return ratio 

def generate_blink_seq_randomly(num_frames, left_eye_max, right_eye_max):
    sd_ratio = np.zeros((num_frames, 1))
    left_lp_ratio = np.ones((num_frames, 1)) * left_eye_max
    right_lp_ratio = np.ones((num_frames, 1)) * right_eye_max

    if num_frames<=20:
        return sd_ratio, np.concatenate((left_lp_ratio, right_lp_ratio), axis=1)
    
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10,num_frames), min(int(num_frames/2), 70))) 
        if frame_id+start+5<=num_frames - 1:
            sd_ratio[frame_id+start:frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            left_lp_ratio[frame_id+start:frame_id+start+5, 0] = [left_eye_max, left_eye_max*0.5, 0., left_eye_max*0.5, left_eye_max]
            right_lp_ratio[frame_id+start:frame_id+start+5, 0] = [right_eye_max, right_eye_max*0.5, 0., right_eye_max*0.5, right_eye_max]
            frame_id = frame_id+start+5
        else:
            break
    return sd_ratio, np.repeat(np.concatenate((left_lp_ratio, right_lp_ratio), axis=1), 2, axis=1)

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