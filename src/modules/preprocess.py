import cv2
import torch
import safetensors
import safetensors.torch 
from tqdm import tqdm
import numpy as np

import src.utils.preprocess.audio as audio
from src.utils.preprocess import networks
from src.utils.preprocess.helper import load_x_from_safetensor, check_source_type, split_coeff, parse_audio_length, crop_pad_audio, generate_blink_seq_randomly
from src.utils.preprocess.sadtalker_preprocess import SadTalkerPreprocess


class Preprocess:
    def __init__(self, device, fps, sadtalker_checkpoint_path, use_blink, speech_rate, syncnet_mel_step_size, sadtalker_preprocesser_cfg):
        self.device = device
        self.fps = fps
        self.use_blink = use_blink
        self.speech_rate = speech_rate
        self.syncnet_mel_step_size = syncnet_mel_step_size

        self.sd_prep = SadTalkerPreprocess(device=device,
                                           **sadtalker_preprocesser_cfg)
        
        self.__load_3dmm_coeff_model(sadtalker_checkpoint_path=sadtalker_checkpoint_path)
        
    def __load_3dmm_coeff_model(self, sadtalker_checkpoint_path):
        checkpoint = safetensors.torch.load_file(sadtalker_checkpoint_path)    
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(self.device)        
        self.net_recon.load_state_dict(load_x_from_safetensor(checkpoint, 'face_3drecon'))
        self.net_recon.eval()

    def __call__(self, source_path, audio_path):
        source_type = check_source_type(source_path)
        if source_type == "image":
            torch_inp_face, face, exp_coeffs, angle_coeffs = self.__image_source_call(source_img_path=source_path)
            data = {"source_type": source_type,
                    "torch_inp_face" : torch_inp_face,
                    "face" : face,
                    "exp_coeffs" : exp_coeffs,
                    "angle_coeffs" : angle_coeffs}
            
        indiv_mels, num_frames = self.__load_audio(audio_path=audio_path)
        blink_ratio = self.__get_blink(num_frames=num_frames)

        data["indiv_mels"] = indiv_mels
        data["num_frame"] = num_frames
        data["blink_ratio"] = blink_ratio
            
        return data

    def __image_source_call(self, source_img_path):
        img = cv2.imread(source_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        torch_inp_face, face, exp_coeffs, angle_coeffs = self.__get_3dmm_coeff(img)
        return torch_inp_face, face, exp_coeffs, angle_coeffs

    def __get_3dmm_coeff(self, frame):
        torch_inp_face, face, _ = self.sd_prep(frame)
        torch_inp_face = torch_inp_face.to(self.device)

        full_coeff = self.net_recon(torch_inp_face)
        coeffs = split_coeff(full_coeff)
        return torch_inp_face, face, coeffs["exp"], coeffs["angle"]

    def __load_audio(self, audio_path):
        wav = audio.load_wav(audio_path, self.speech_rate) 
        wav_length, num_frames = parse_audio_length(len(wav), self.speech_rate, self.fps)
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio.melspectrogram(wav).T
        spec = orig_mel.copy()         # nframes 80
        indiv_mels = []

        for i in tqdm(range(num_frames), 'Generating Mel Spectrograms...'):
            start_frame_num = i-2
            start_idx = int(80. * (start_frame_num / float(self.fps)))
            end_idx = start_idx + self.syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
            m = spec[seq, :]
            indiv_mels.append(m.T)

        indiv_mels = np.asarray(indiv_mels)
        indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0)
        return indiv_mels.to(self.device), num_frames
    
    def __get_blink(self, num_frames):
        ratio = generate_blink_seq_randomly(num_frames)
        if self.use_blink:
            ratio = torch.FloatTensor(ratio).unsqueeze(0)                       # bs T
        else:
            ratio = torch.FloatTensor(ratio).unsqueeze(0).fill_(0.) 
            
        return ratio.to(self.device)
