import os
import cv2
import numpy as np


class SaveResult:
    def __init__(self, save_path):
        self.save_path = save_path

    def __call__(self, batch):
        os.makedirs(self.save_path, exist_ok=True)

        video_path = os.path.join(self.save_path, batch["time"])
        os.makedirs(video_path)

        tmp_folder_path = os.path.join(self.save_path, batch["time"], "tmp")
        os.makedirs(tmp_folder_path)
        
        original_frame = batch["original_frame"]
        face_crop_coords = batch["face_crop_coords"]
        original_width = face_crop_coords[2]-face_crop_coords[0]
        original_height = face_crop_coords[3]-face_crop_coords[1]
        for idx, rendered_frame in enumerate(batch["rendered_frame_list"]):
            rendered_frame = (rendered_frame[0].permute(1,2,0).detach().cpu().numpy()*255).astype("uint8")
            original_frame[face_crop_coords[1]:face_crop_coords[3], face_crop_coords[0]:face_crop_coords[2]] = cv2.resize(rendered_frame, (original_width, original_height))
            cv2.imwrite(f"{tmp_folder_path}/{str(idx).zfill(len(batch['rendered_frame_list']))}.png", cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))

        os.system(f"cd {tmp_folder_path} && ffmpeg -y -hide_banner -loglevel error -framerate 25 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p ../rendered_output_novoice.mp4")
        os.system(f"rm -rf {tmp_folder_path}")
        os.system(f"ffmpeg -hide_banner -loglevel error -i {os.path.join(video_path, 'rendered_output_novoice.mp4')} -i {batch['audio_path']} -map 0:v -map 1:a -c:v copy -shortest {os.path.join(video_path, 'rendered_output.mp4')}")
        os.system(f"rm -rf {os.path.join(video_path, 'rendered_output_novoice.mp4')}")