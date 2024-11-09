import os
import cv2
import torch
import random
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import Dataset


class VoxCeleb2Dataset(Dataset):
    def __init__(self, dataset_folder_path, input_size, semantic_radius, transform=None):
        super().__init__()

        self.dataset_folder_path = dataset_folder_path
        self.semantic_radius = semantic_radius
        self.input_size = input_size

        self.mat_path_list = self.__load_dataset()

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

    def __load_dataset(self):
        mat_path_list = []
        for id_name in os.listdir(self.dataset_folder_path):
            if id_name.startswith("."): continue

            for ytid_name in os.listdir(os.path.join(self.dataset_folder_path, id_name)):
                if ytid_name.startswith("."): continue

                ytid_full_path = os.path.join(self.dataset_folder_path, id_name, ytid_name)
                mat_files = [os.path.join(ytid_full_path, split_name) for split_name in os.listdir(ytid_full_path) if split_name.endswith("mat")]
                if len(mat_files) > 0:
                    mat_path_list += mat_files

        return mat_path_list
    
    def __len__(self):
        return len(self.mat_path_list)
    
    def __read_video(self, selected_video_path):
        return cv2.VideoCapture(selected_video_path)
    
    def __read_video_hash(self, selected_video_hash):
        mat = loadmat(selected_video_hash)
        return mat["threedmm"], mat["face_crop_list"], mat["frame_number_list"].reshape(-1), mat["landmarks_list"]
    
    def __transform_semantic_target(self, coeff_3dmm, frame_index):
        num_frames = coeff_3dmm.shape[0]
        seq = list(range(frame_index- self.semantic_radius, frame_index + self.semantic_radius+1))
        index = [ min(max(item, 0), num_frames-1) for item in seq ] 
        coeff_3dmm_g = coeff_3dmm[index, :]
        return coeff_3dmm_g.transpose(1,0)
    
    def __get_cropped_face(self, cap, selected_frame_number, selected_face_crop_list):
        cap.set(1, selected_frame_number)
        _, frame = cap.read()
        
        cropped_face = frame[selected_face_crop_list[1]:selected_face_crop_list[3], selected_face_crop_list[0]:selected_face_crop_list[2]]
        cropped_face = cv2.resize(cropped_face, (self.input_size, self.input_size))
        return cropped_face
    
    def __getitem__(self, idx):
        selected_mat_path = self.mat_path_list[idx]
        selected_video_path = selected_mat_path.replace("mat", "mp4")

        cap = self.__read_video(selected_video_path)
        threedmm_list, face_crop_list, frame_number_list, landmarks_list = self.__read_video_hash(selected_mat_path)
        
        selected_idx = random.randint(0, threedmm_list.shape[0]-1)
        selected_face_crop_list = face_crop_list[selected_idx]
        selected_frame_number = frame_number_list[selected_idx]
        selected_landmark = landmarks_list[selected_idx]

        threedmm_list = self.__transform_semantic_target(threedmm_list, selected_frame_number)
        cropped_face = self.__get_cropped_face(cap, selected_frame_number, selected_face_crop_list)

        return {"threedmm_list": torch.FloatTensor(threedmm_list)[:64, :],
                "cropped_face": self.transform(cropped_face),
                "selected_frame_number": selected_frame_number,
                "face_crop_coordinates": selected_face_crop_list,
                "landmarks": selected_landmark}
