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

        self.ref_head_pose_inputs_exist = False
        if self.ref_head_pose_path is not None:
            self.ref_head_pose_name = self.ref_head_pose_path.split("/")[-1].split(".")[0]

            if os.path.exists(os.path.join(self.save_path, "references", self.ref_head_pose_name, "batch.mat")):
                self.ref_head_pose_inputs_exist = True

        if os.path.exists(os.path.join(self.source_folder_path, "preprocessed_inputs", "batch.mat")):
            self.preprocessed_inputs_exist = True
        else:
            self.preprocessed_inputs_exist = False

    def save(self, batch):
        self.save_output(still=batch["still"],
                         source_type=batch["source_type"],
                         rendered_frame_list=batch["rendered_frame_list"],
                         num_frames=batch["num_frames"],
                         time=batch["time"],
                         audio_path=batch["audio_path"],
                         original_frame=batch["original_frame"],
                         face_crop_coords=batch["face_crop_coords"])
        
        #self.save_inputs(source_type=batch["source_type"],
        #                rendering_input_face=batch["rendering_input_face"],
        #                face_crop_coords=batch["face_crop_coords"],
        #                original_frame=batch["original_frame"],
        #                source_coeff=batch["source_coeff"])
            
        if self.ref_head_pose_path is not None:
            self.save_references(source_type=batch["ref_source_type"],
                                 ref_R_list=batch["ref_R_list"])
        
    def instant_save_input(self, source_type, source_coeff, source_eye_close_ratio, x_s_i_info, R_s_i, f_s_i, x_s_i):
        input_folder_path = os.path.join(self.source_folder_path, "preprocessed_inputs")
        os.makedirs(input_folder_path, exist_ok=True)

        batch_to_save = {"source_type": source_type,
                         "source_coeff": source_coeff[0, 0, :].unsqueeze(0).detach().cpu().numpy(),
                         "source_eye_close_ratio": source_eye_close_ratio[0].unsqueeze(0).detach().cpu().numpy(),
                         "x_s_i_info": {k: [] for k in x_s_i_info.keys()},
                         "R_s_i": [],
                         "f_s_i": [],
                         "x_s_i": []}

        file_save_path = os.path.join(input_folder_path, "batch.mat")
        if os.path.exists(file_save_path):
            batch_to_save = loadmat(file_save_path)

        batch_to_save["R_s_i"].append(R_s_i.detach().cpu().numpy())
        batch_to_save["f_s_i"].append(f_s_i.detach().cpu().numpy())
        batch_to_save["x_s_i"].append(x_s_i.detach().cpu().numpy())
        for k, v in x_s_i_info.items():
            batch_to_save["x_s_i_info"][k].append(v.detach().cpu().numpy())
        
        savemat(file_save_path, batch_to_save)

    def save_references(self, source_type, ref_R_list):
        input_folder_path = os.path.join(self.save_path, "references", self.ref_head_pose_name)
        os.makedirs(input_folder_path, exist_ok=True)

        batch_to_save = {"source_type": source_type,
                         "ref_R_list": [R.detach().cpu().numpy() for R in ref_R_list]}
        savemat(os.path.join(input_folder_path, "batch.mat"), batch_to_save)

    def load_inputs(self):
        batch_to_load = {}
        if False: #self.preprocessed_inputs_exist:
            batch_inputs = loadmat(os.path.join(self.source_folder_path, "preprocessed_inputs", "batch.mat"))
            batch_to_load["source_type"] = batch_inputs["source_type"][0]
            batch_to_load["source_coeff"] = torch.tensor(batch_inputs["source_coeff"]).to(self.device)
            batch_to_load["source_eye_close_ratio"] = torch.tensor(batch_inputs["source_eye_close_ratio"]).to(self.device)
            batch_to_load["x_s_i_info"] = {k: torch.tensor(v).to(self.device) for k, v in batch_inputs["x_s_i_info"]}
            batch_to_load["R_s_i"] = [torch.tensor(i).to(self.device) for i in batch_inputs["R_s_i"]]
            batch_to_load["f_s_i"] = [torch.tensor(i).to(self.device) for i in batch_inputs["f_s_i"]]
            batch_to_load["x_s_i"] = [torch.tensor(i).to(self.device) for i in batch_inputs["x_s_i"]]
            print("Using existing inputs for this source input!")

        
        if self.ref_head_pose_inputs_exist:
            ref_batch = loadmat(os.path.join(self.save_path, "references", self.ref_head_pose_name, "batch.mat"))
            batch_to_load["ref_source_type"] = ref_batch["source_type"][0]
            batch_to_load["ref_R_list"] = [torch.tensor(R).to(self.device) for R in ref_batch["ref_R_list"]]
            print("Using existing inputs for this reference input!")

        return batch_to_load

    def save_output(self, still, source_type, rendered_frame_list, num_frames, time, audio_path, original_frame, face_crop_coords):
        tmp_folder_path = os.path.join(self.source_folder_path, "tmp")
        os.makedirs(tmp_folder_path)

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
