import os
import cv2
import torch
from scipy.io import savemat, loadmat


class FileOperations:
    def __init__(self, device, save_path, source_path, audio_path, ref_head_pose_path):
        self.device = device
        self.save_path = save_path
        self.source_name = source_path.split("/")[-1].split(".")[0]
        self.audio_name = audio_path.split("/")[-1].split(".")[0]
        self.ref_head_pose_path = ref_head_pose_path

        self.source_folder_path = self.create_folders_if_not_exist()

        #self.ref_head_pose_inputs_exist = False
        #if self.ref_head_pose_path is not None:
        #    self.ref_head_pose_name = self.ref_head_pose_path.split("/")[-1].split(".")[0]

        #    if os.path.exists(os.path.join(self.save_path, "references", self.ref_head_pose_name, "batch.mat")):
        #        self.ref_head_pose_inputs_exist = True

        if os.path.exists(os.path.join(self.source_folder_path, "preprocessed_inputs", "batch.mat")):
            print(f"Preproseed inputs exists for {self.source_name}!")
            self.preprocessed_inputs_exist = True
        else:
            self.preprocessed_inputs_exist = False

    def save(self, batch):
        self.save_output(source_type=batch["source_type"],
                         ref_head_pose_path=batch["ref_head_pose_path"],
                         rendered_frame_list=batch["rendered_frame_list"],
                         num_frames=batch["num_frames"],
                         time=batch["time"],
                         audio_path=batch["audio_path"],
                         original_frame=batch["original_frame"],
                         face_crop_coords=batch["face_crop_coords"])
        
        self.save_inputs(source_type=batch["source_type"],
                        rendering_input_face=batch["rendering_input_face"],
                        face_crop_coords=batch["face_crop_coords"],
                        original_frame=batch["original_frame"],
                        source_coeff=batch["source_coeff"])
            
        #if self.ref_head_pose_path is not None and not self.ref_head_pose_inputs_exist:
        #    self.save_references(ref_head_pose_coeff=batch["ref_head_pose_coeff"])
        
    def save_inputs(self, source_type, rendering_input_face, face_crop_coords, original_frame, source_coeff):
        input_folder_path = os.path.join(self.source_folder_path, "preprocessed_inputs")
        os.makedirs(input_folder_path, exist_ok=True)

        batch_to_save = {"source_type": source_type,
                         "source_coeff": source_coeff[0, 0, :].unsqueeze(0).detach().cpu().numpy()}
        
        savemat(os.path.join(input_folder_path, "batch.mat"), batch_to_save)

    def save_references(self, ref_head_pose_coeff):
        input_folder_path = os.path.join(self.save_path, "references", self.ref_head_pose_name)
        os.makedirs(input_folder_path, exist_ok=True)

        batch_to_save = {"pose_coeff": ref_head_pose_coeff[0, 0].detach().cpu().numpy()}
        savemat(os.path.join(input_folder_path, "batch.mat"), batch_to_save)

    def load_inputs(self):
        batch_to_load = {}
        if self.preprocessed_inputs_exist:
            batch_inputs = loadmat(os.path.join(self.source_folder_path, "preprocessed_inputs", "batch.mat"))
            batch_to_load["source_type"] = batch_inputs["source_type"][0]
            batch_to_load["source_coeff"] = torch.tensor(batch_inputs["source_coeff"]).to(self.device)
        
        #if self.ref_head_pose_inputs_exist:
        #    ref_batch = loadmat(os.path.join(self.save_path, "references", self.ref_head_pose_name, "batch.mat"))
        #    batch_to_load["ref_head_pose_coeff"] = torch.tensor(ref_batch["pose_coeff"]).to(self.device)

        return batch_to_load

    def save_output(self, source_type, ref_head_pose_path, rendered_frame_list, num_frames, time, audio_path, original_frame, face_crop_coords):
        tmp_folder_path = os.path.join(self.source_folder_path, "tmp")
        os.makedirs(tmp_folder_path)

        video_name = f"{self.audio_name}_{time}.mp4"
        for idx, rendered_frame in enumerate(rendered_frame_list):
            rendered_frame = (rendered_frame[0].permute(1,2,0).detach().cpu().numpy()*255).astype("uint8")

            if source_type == "video" or (source_type == "image" and ref_head_pose_path is None):
                resized_rendered_frame = cv2.resize(rendered_frame,
                                                    (face_crop_coords[idx][2]-face_crop_coords[idx][0], face_crop_coords[idx][3]-face_crop_coords[idx][1]))
                original_frame[idx][face_crop_coords[idx][1]:face_crop_coords[idx][3],
                                    face_crop_coords[idx][0]:face_crop_coords[idx][2]] = resized_rendered_frame
            
            else:
                original_frame[idx] = rendered_frame
            
            cv2.imwrite(f"{tmp_folder_path}/{str(idx).zfill(len(str(num_frames)))}.png",
                        cv2.cvtColor(original_frame[idx], cv2.COLOR_RGB2BGR))

        os.system(f"ffmpeg -y -hide_banner -loglevel error -framerate 25 -pattern_type glob -i '{tmp_folder_path}/*.png' -c:v libx264 -pix_fmt yuv420p {os.path.join(self.source_folder_path, video_name.replace('.mp4', '_novoice.mp4'))}")
        os.system(f"rm -rf {tmp_folder_path}")
        os.system(f"ffmpeg -hide_banner -loglevel error -i {os.path.join(self.source_folder_path, video_name.replace('.mp4', '_novoice.mp4'))} -i {audio_path} -map 0:v -map 1:a -c:v copy -shortest {os.path.join(self.source_folder_path, video_name)}")
        os.system(f"rm -rf {os.path.join(self.source_folder_path, video_name.replace('.mp4', '_novoice.mp4'))}")

    def create_folders_if_not_exist(self):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "references"), exist_ok=True)

        source_folder_path = os.path.join(self.save_path, self.source_name)
        os.makedirs(source_folder_path, exist_ok=True)
        return source_folder_path
