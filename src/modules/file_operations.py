import os
import cv2
import torch
import imageio
import numpy as np
from scipy.io import savemat, loadmat


class FileOperations:
    def __init__(self, device, fps, save_path, source_path, audio_path, ref_head_pose_path):
        self.device = device
        self.fps = fps
        self.save_path = save_path
        self.source_name = source_path.split(os.sep)[-1].split(".")[0]
        self.audio_name = audio_path.split(os.sep)[-1].split(".")[0]
        self.ref_head_pose_path = ref_head_pose_path

        self.source_folder_path = self.create_folders_if_not_exist()

        self.ref_head_pose_inputs_exist = False
        if self.ref_head_pose_path is not None:
            self.ref_head_pose_name = self.ref_head_pose_path.split(os.sep)[-1].split(".")[0]

            if os.path.exists(os.path.join(self.save_path, "references", self.ref_head_pose_name, "batch.mat")):
                self.ref_head_pose_inputs_exist = True

    def save(self, batch):
        self.save_output(still=batch["still"],
                         source_type=batch["source_type"],
                         rendered_frame_list=batch["rendered_frame_list"],
                         num_frames=batch["num_frames"],
                         time=batch["time"],
                         audio_path=batch["audio_path"],
                         original_frame=batch["original_frame"],
                         face_crop_coords=batch["face_crop_coords"])
        
        if self.ref_head_pose_path is not None:
            self.save_references(source_type=batch["ref_source_type"],
                                 ref_R_list=batch["ref_R_list"])

    def save_references(self, source_type, ref_R_list):
        input_folder_path = os.path.join(self.save_path, "references", self.ref_head_pose_name)
        os.makedirs(input_folder_path, exist_ok=True)

        batch_to_save = {"source_type": source_type,
                         "ref_R_list": [R.detach().cpu().numpy() for R in ref_R_list]}
        savemat(os.path.join(input_folder_path, "batch.mat"), batch_to_save)

    def load_inputs(self):
        batch_to_load = {}
        
        if self.ref_head_pose_inputs_exist:
            ref_batch = loadmat(os.path.join(self.save_path, "references", self.ref_head_pose_name, "batch.mat"))
            batch_to_load["ref_source_type"] = ref_batch["source_type"][0]
            batch_to_load["ref_R_list"] = [torch.tensor(R).to(self.device) for R in ref_batch["ref_R_list"]]
            print("Using existing inputs for this reference input!")

        return batch_to_load

    def save_output(self, still, source_type, rendered_frame_list, num_frames, time, audio_path, original_frame, face_crop_coords):
        frame_list = []
        video_name = f"{self.audio_name}_{time}.mp4"
        for idx, rendered_frame in enumerate(rendered_frame_list):
            rendered_frame = (rendered_frame[0].permute(1,2,0).detach().cpu().numpy()*255).astype("uint8")

            if source_type == "video" or (source_type == "image" and still):
                resized_rendered_frame = cv2.resize(rendered_frame,
                                                    (face_crop_coords[idx][2]-face_crop_coords[idx][0], face_crop_coords[idx][3]-face_crop_coords[idx][1]))
                original_frame[idx][face_crop_coords[idx][1]:face_crop_coords[idx][3],
                                    face_crop_coords[idx][0]:face_crop_coords[idx][2]] = resized_rendered_frame
            
            else:
                original_frame[idx] = rendered_frame
            
            h, w = original_frame[idx].shape[:2]
            w -= w % 2
            h -= h % 2
            original_frame[idx] = cv2.resize(original_frame[idx], (w, h))
            frame_list.append(original_frame[idx])

        imageio.mimsave(os.path.join(self.source_folder_path, video_name.replace('.mp4', '_novoice.mp4')), frame_list,  fps=float(self.fps))

        os.system(f"ffmpeg -hide_banner -loglevel error -i {os.path.join(self.source_folder_path, video_name.replace('.mp4', '_novoice.mp4'))} -i {audio_path} -map 0:v -map 1:a -c:v copy -shortest {os.path.join(self.source_folder_path, video_name)}")
        os.remove(os.path.join(self.source_folder_path, video_name.replace('.mp4', '_novoice.mp4')))

    def create_folders_if_not_exist(self):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "references"), exist_ok=True)

        source_folder_path = os.path.join(self.save_path, self.source_name)
        os.makedirs(source_folder_path, exist_ok=True)
        return source_folder_path
