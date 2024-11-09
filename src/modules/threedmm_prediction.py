import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import safetensors
import safetensors.torch 
from scipy.io import savemat
import threading
from joblib import Parallel, delayed

from src.utils.download.utils import read_file
from src.utils.threedmm import networks
from src.utils.threedmm.helper import voxceleb_crop_frame, load_x_from_safetensor, split_coeff
from src.utils.threedmm.preprocesser import SadTalkerPreprocess


class ThreeDMMPrediction:
    def __init__(self, device, fps, batch_size, dataset_folder_path, raw_folder_path, sadtalker_checkpoint_path, preprocesser_cfg):
        self.device = device
        self.fps = fps
        self.batch_size = batch_size
        self.dataset_folder_path = dataset_folder_path
        self.raw_folder_path = raw_folder_path

        self.ytid_list = self.__load_dataset()

        self.sd_prep = SadTalkerPreprocess(device=device, **preprocesser_cfg)

        checkpoint = safetensors.torch.load_file(sadtalker_checkpoint_path)    
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(self.device)        
        self.net_recon.load_state_dict(load_x_from_safetensor(checkpoint, 'face_3drecon'))
        self.net_recon.eval()

        self.preprocess_lock = threading.Lock()
        self.net_recon_lock = threading.Lock()
    
    def __load_dataset(self):
        ytid_list = []
        for id_name in os.listdir(self.dataset_folder_path):
            if id_name.startswith("."): continue

            for ytid_name in os.listdir(os.path.join(self.dataset_folder_path, id_name)):
                if ytid_name.startswith("."): continue
                ytid_list.append(os.path.join(self.dataset_folder_path, id_name, ytid_name))

        return ytid_list
    
    def __read_video(self, selected_video_path):
        cap = cv2.VideoCapture(selected_video_path)
        total_frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, total_frame_number, total_frame_number/self.fps

    def __interpolate_coords(self, video_duration, total_frame_number, coord_list):
        original_timeline = np.linspace(0, video_duration, len(coord_list))
        new_timeline = np.linspace(0, video_duration, total_frame_number)

        coord_list = np.array(coord_list)
        coord_list = np.array([np.interp(new_timeline,
                                         original_timeline,
                                         coord_list[:, i]) for i in range(coord_list.shape[1])]).T
        return coord_list
    
    def __get_threedmm_info(self, cap, coord_list):
        threedmm_list = []
        face_crop_list = []
        frame_number_list = []      
        landmarks_list = []      

        batch_hash = {"torch_face": [], "sd_crop": [], "landmarks": [], "voxceleb_crop": [], "frame_number": []}
        for frame_number, coord in enumerate(coord_list):
            cap.set(1, frame_number)
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cropped_frame, voxceleb_crop = voxceleb_crop_frame(frame, coord, scale_crop=2)

            with self.preprocess_lock:
                torch_face, sd_crop, landmarks = self.sd_prep(cropped_frame)

                if torch_face is None:
                    print("Torch face is None, skipping...")
                    return None, None, None, None

            batch_hash["sd_crop"].append(sd_crop)
            batch_hash["torch_face"].append(torch_face)
            batch_hash["frame_number"].append(frame_number)
            batch_hash["voxceleb_crop"].append(voxceleb_crop)
            batch_hash["landmarks"].append(landmarks)

        full_coeff_list = []
        for batch_idx in range(0, len(batch_hash["torch_face"]), self.batch_size):
            with torch.no_grad():
                with self.net_recon_lock:
                    full_coeff = self.net_recon(torch.cat(batch_hash["torch_face"][batch_idx:batch_idx+self.batch_size], dim=0).to(self.device))
                    full_coeff_list += torch.split(full_coeff, 1, dim=0)
                
        for full_coeff, frame_number, voxceleb_crop, sd_crop, landmarks in zip(full_coeff_list, batch_hash["frame_number"],
                                                                    batch_hash["voxceleb_crop"], batch_hash["sd_crop"], batch_hash["landmarks"]):
            coeffs = split_coeff(full_coeff)

            pred_coeff = {key:coeffs[key].detach().cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff["exp"][0],
                                         pred_coeff["angle"][0],
                                         pred_coeff["trans"][0]])
            
            frame_number_list.append(frame_number)
            threedmm_list.append(pred_coeff)
            landmarks_list.append(landmarks)
            face_crop_list.append([voxceleb_crop[0]+sd_crop[0],
                                   voxceleb_crop[1]+sd_crop[1],
                                   voxceleb_crop[0]+sd_crop[2],
                                   voxceleb_crop[1]+sd_crop[3]])


        if len(threedmm_list) > 0:
            return np.stack(threedmm_list), face_crop_list, frame_number_list, landmarks_list
        else:
            return None, None, None, None
    
    def predict(self):
        def process_pipe(ytid):
            for video_name in tqdm(os.listdir(ytid), desc=f"Working on {'/'.join(ytid.split('/')[-2:])} splits..."):
                if not video_name.endswith("mp4"):
                    continue

                if os.path.exists(os.path.join(ytid, video_name.replace("mp4", "mat"))) or os.path.exists(os.path.join(ytid.replace("train", "val"), video_name.replace("mp4", "mat"))):
                    print(f"({'/'.join(ytid.split('/')[-2:])}/{video_name.replace('mp4', 'mat')}) - Exists, skipping...")
                    continue

                _, _, coord_list = read_file(os.path.join(self.raw_folder_path,
                                                          "/".join(ytid.split("/")[-2:]),
                                                          video_name.replace("mp4", "txt")))

                try:
                    cap, total_frame_number, video_duration = self.__read_video(os.path.join(ytid, video_name))

                except Exception as err:
                    print(f"({'/'.join(ytid.split('/')[-2:])}/{video_name}) - Could not read. Error: {err}. Skipping...")
                    continue

                coord_list = self.__interpolate_coords(video_duration, total_frame_number, coord_list)

                try:
                    threedmm_stack, face_crop_list, frame_number_list, landmarks_list = self.__get_threedmm_info(cap, coord_list)

                except Exception as err:
                    print(f"({'/'.join(ytid.split('/')[-2:])}/{video_name}) - Could not predict threedmm. Error: {err}. Skipping...")
                    continue

                if threedmm_stack is None:
                    print(f"({'/'.join(ytid.split('/')[-2:])}/{video_name}) - No threedmm skipping...")
                    continue

                data = {"threedmm": threedmm_stack, "face_crop_list": face_crop_list,
                        "landmarks_list": landmarks_list, "frame_number_list": frame_number_list}
                savemat(os.path.join(ytid, video_name.replace("mp4", "mat")), data)

        _ = Parallel(n_jobs=-1, backend="threading")(delayed(process_pipe)(ytid) for ytid in tqdm(self.ytid_list, desc="Preprocessing videos..."))
