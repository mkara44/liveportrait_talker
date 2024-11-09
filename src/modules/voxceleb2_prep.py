import os
from tqdm import tqdm
from joblib import Parallel, delayed

from src.utils.download.utils import read_file, download_video, split_video


class VoxCeleb2Preprocess:
    def __init__(self, dataset_folder_path, raw_folder_path, target_fps):
        
        self.dataset_folder_path = dataset_folder_path
        self.raw_folder_path = raw_folder_path
        self.target_fps = target_fps

    def __get_video_hash(self, folder_id):
        video_hash_list = []
        for yt_id in os.listdir(os.path.join(self.raw_folder_path, folder_id)):
            if yt_id.startswith("."): continue

            info_files = os.listdir(os.path.join(self.raw_folder_path, folder_id, yt_id))
            info_files = [os.path.join(self.raw_folder_path, folder_id, yt_id, inff) for inff in info_files if not inff.startswith(".")]                

            video_hash =  {"name": f"{folder_id}-{yt_id}", 
                           "folder_id": folder_id,
                           "youtube_id": yt_id,
                           "url": f"https://www.youtube.com/watch?v={yt_id}",
                           "frame_crop_info_list": [read_file(inff) for inff in info_files]}
            
            video_hash_list.append(video_hash)
        return video_hash_list
            
    def download(self):
        def preprocess_pipe(video_hash):
            download_path = os.path.join(self.dataset_folder_path, video_hash["folder_id"], video_hash["youtube_id"])
            os.makedirs(download_path, exist_ok=True)

            video_path = download_video(name=video_hash["name"],
                                        url=video_hash["url"],
                                        download_path=download_path)
            
            if video_path is not None:
                for file_path, frame_number_list, _ in video_hash["frame_crop_info_list"]:
                    split_video(video_path=video_path,
                                frame_number_list=frame_number_list,
                                file_name=file_path.split("/")[-1],
                                save_path=download_path,
                                fps=self.target_fps)
                    
                os.system(f"rm {video_path}")

        count = 0
        total_folder = len(self.raw_folder_path)
        for folder_id in os.listdir(self.raw_folder_path):
            if folder_id.startswith("."): continue
            count += 1
            print(f"({count}/{total_folder}) Preprocessing started for folder id: {folder_id}...")

            os.makedirs(os.path.join(self.dataset_folder_path, folder_id), exist_ok=True)

            video_hash_list = self.__get_video_hash(folder_id)
            _ = Parallel(n_jobs=-1, backend="threading")(delayed(preprocess_pipe)(video_hash) for video_hash in tqdm(video_hash_list, desc="Preprocessing videos..."))