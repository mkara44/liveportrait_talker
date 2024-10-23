import cv2
import torch
import safetensors
import safetensors.torch 
from tqdm import tqdm
import numpy as np

import src.utils.preprocess.audio as audio
from src.utils.preprocess import networks
from src.utils.helper import load_x_from_safetensor
from src.utils.preprocess.helper import check_source_type, split_coeff, parse_audio_length, crop_pad_audio, generate_blink_seq_randomly
from src.utils.preprocess.sadtalker_preprocess import SadTalkerPreprocess


class Preprocess:
    def __init__(self, device, fps, sadtalker_checkpoint_path,
                 preprocessed_inputs_exist, ref_head_pose_inputs_exist, 
                 use_blink, speech_rate, syncnet_mel_step_size, liveportrait_input_shape, sadtalker_preprocesser_cfg):
        self.device = device
        self.fps = fps
        self.use_blink = use_blink
        self.speech_rate = speech_rate
        self.syncnet_mel_step_size = syncnet_mel_step_size
        self.liveportrait_input_shape = liveportrait_input_shape
        self.preprocessed_inputs_exist = preprocessed_inputs_exist
        self.ref_head_pose_inputs_exist = ref_head_pose_inputs_exist

        self.sd_prep = SadTalkerPreprocess(device=device,
                                           **sadtalker_preprocesser_cfg)
        
        self.__load_3dmm_coeff_model(sadtalker_checkpoint_path=sadtalker_checkpoint_path)
        
    def __load_3dmm_coeff_model(self, sadtalker_checkpoint_path):
        checkpoint = safetensors.torch.load_file(sadtalker_checkpoint_path)    
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(self.device)        
        self.net_recon.load_state_dict(load_x_from_safetensor(checkpoint, 'face_3drecon'))
        self.net_recon.eval()

    def __call__(self, batch):
        indiv_mels, num_frames = self.__load_audio(audio_path=batch["audio_path"])
        batch["indiv_mels"] = indiv_mels
        batch["num_frames"] = num_frames

        source_type = check_source_type(batch["source_path"])
        do_pred_coeff = True #if not self.preprocessed_inputs_exist else False
        if source_type == "image":
            original_frame, face_for_rendering, pred_coeff, face_crop_coords, eye_close_ratio = self.__image_source_call(inp_path=batch["source_path"],
                                                                                                                         num_frames=num_frames,
                                                                                                                         do_pred_coeff=do_pred_coeff,
                                                                                                                         no_crop=batch["no_crop"])
            
            eye_close_ratio = torch.tensor(eye_close_ratio, dtype=torch.float32).repeat(num_frames, 1).to(self.device)
            sd_ratio, lp_ratio = self.__get_blink(num_frames=num_frames, eye_close_ratio=eye_close_ratio.detach().cpu().numpy()) #max_point=eye_close_ratio.max().detach().cpu().item())

            batch["source_eye_close_ratio"] = eye_close_ratio
            batch["liveportrait_blink_ratio"] = lp_ratio

        elif source_type == "video":
            original_frame, face_for_rendering, pred_coeff, face_crop_coords = self.__video_source_call(inp_path=batch["source_path"],
                                                                                                        num_frames=num_frames,
                                                                                                        do_pred_coeff=do_pred_coeff,
                                                                                                        no_crop=batch["no_crop"])
            sd_ratio, _ = self.__get_blink(num_frames=num_frames)


        pred_coeff = pred_coeff if pred_coeff is not None else batch["source_coeff"]
        pred_coeff = pred_coeff.repeat(num_frames, 1).unsqueeze(0)

        batch["source_type"] = source_type
        batch["rendering_input_face"] = face_for_rendering
        batch["face_crop_coords"] = face_crop_coords
        batch["original_frame"] = original_frame
        batch["source_coeff"] = pred_coeff
        batch["sadtalker_blink_ratio"] = sd_ratio


        if batch["ref_head_pose_path"] is not None and not self.ref_head_pose_inputs_exist:
            reference_source_type = check_source_type(batch["ref_head_pose_path"])
            if reference_source_type == "image":
                _, face_for_rendering, _, _, _ = self.__image_source_call(inp_path=batch["ref_head_pose_path"],
                                                                          num_frames=num_frames,
                                                                          do_pred_coeff=False)
                
            elif reference_source_type == "video":
                _, face_for_rendering, _, _ = self.__video_source_call(inp_path=batch["ref_head_pose_path"],
                                                                          num_frames=num_frames,
                                                                          do_pred_coeff=False,
                                                                          source=False)
            batch["ref_source_type"] = reference_source_type
            batch["ref_rendering_input_face"] = face_for_rendering

        return batch

    def __image_source_call(self, inp_path, num_frames, do_pred_coeff, no_crop=False):
        img = cv2.imread(inp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_for_rendering, pred_coeff, crop_for_rendering, eye_close_ratio = self.__get_3dmm_coeff(img, do_pred=do_pred_coeff, no_crop=no_crop)
        return [img]*num_frames, face_for_rendering, pred_coeff, [crop_for_rendering]*num_frames, eye_close_ratio
    
    def __video_source_call(self, inp_path, num_frames, do_pred_coeff, source=True, no_crop=False):
        cap = cv2.VideoCapture(inp_path)
        video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not source else num_frames

        frame_list = []
        pred_coeff = None
        crop_for_rendering_list = []
        face_for_rendering_list = []
        for _ in tqdm(range(video_num_frames), "Preprocessing video.."):
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            do_pred = True if pred_coeff is None and do_pred_coeff else False
            face_for_rendering, _pred_coeff, crop_for_rendering, _ = self.__get_3dmm_coeff(frame, do_pred=do_pred, no_crop=no_crop)

            if pred_coeff is None:
                pred_coeff = _pred_coeff

            frame_list.append(frame)
            crop_for_rendering_list.append(crop_for_rendering)
            face_for_rendering_list.append(face_for_rendering)

        #while len(frame_list) > num_frames:
        #    frame_list += frame_list[-video_num_frames:][::-1]
        #    crop_for_rendering_list += crop_for_rendering_list[-video_num_frames:][::-1]
        #    face_for_rendering_list += face_for_rendering_list[-video_num_frames:][::-1]

        #    print("Number of video frames are smaller than expected frame number, video frames are reversed and added!")

        face_for_rendering_list = torch.cat(face_for_rendering_list, dim=0)
        return frame_list, face_for_rendering_list, pred_coeff, crop_for_rendering_list

    def __get_3dmm_coeff(self, frame, do_pred=True, no_crop=False):
        torch_inp_face, face_for_rendering, _, crop_for_rendering, _, eye_close_ratio = self.sd_prep(frame, no_crop=no_crop)
        torch_inp_face = torch_inp_face.to(self.device)

        pred_coeff = None
        if do_pred:
            full_coeff = self.net_recon(torch_inp_face)
            pred_coeff = split_coeff(full_coeff)
            pred_coeff = torch.cat([pred_coeff["exp"][0],
                                    pred_coeff["angle"][0],
                                    pred_coeff["trans"][0]])
        
        return face_for_rendering, pred_coeff, crop_for_rendering, eye_close_ratio

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
    
    def __get_blink(self, num_frames, eye_close_ratio=None):
        left_eye_max = eye_close_ratio[:, :1].max() if eye_close_ratio is not None else .5
        right_eye_max = eye_close_ratio[: 1:].max() if eye_close_ratio is not None else .5

        sd_ratio, lp_ratio = generate_blink_seq_randomly(num_frames, left_eye_max=left_eye_max, right_eye_max=right_eye_max)

        if self.use_blink:
            sd_ratio = torch.FloatTensor(sd_ratio).unsqueeze(0).fill_(0.).to(self.device)
            lp_ratio = torch.FloatTensor(lp_ratio).to(self.device)
        else:
            sd_ratio = torch.FloatTensor(sd_ratio).unsqueeze(0).fill_(0.).to(self.device) 
            lp_ratio = torch.FloatTensor(lp_ratio).fill_(left_eye_max).to(self.device) 

        return sd_ratio, lp_ratio
