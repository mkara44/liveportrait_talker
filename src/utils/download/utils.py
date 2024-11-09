import os
import numpy as np
import pandas as pd
from pytubefix import YouTube
from pytubefix.exceptions import LoginRequired


def read_file(file_path):
    df = pd.read_csv(file_path, sep="\t", skiprows=6)
    
    df["merged"] = df.apply(lambda x: [x["X "], x["Y "], x["W "], x["H "]], axis=1)
    frames = df["FRAME "].tolist()
    coords = df["merged"].tolist()

    return file_path, frames, coords

def download_video(name, url, download_path):
    try:
        yt = YouTube(url)
        yt = yt.streams.filter(mime_type='video/mp4').order_by("resolution").last()

        file_name=f"{name}.mp4"
        yt.download(download_path, filename=file_name)

    except LoginRequired:
        print(f"({name}) Login Required error, skipping...")
        os.system(f"rm -rf {download_path}")
        return None

    except Exception as err:
        print(f"({name}) Error: {err}, skipping...")
        os.system(f"rm -rf {download_path}")
        return None

    video_path = os.path.join(download_path, file_name)
    return video_path

def split_video(video_path, frame_number_list, file_name, save_path, fps):
    first_frame = round(frame_number_list[0]/ fps, 3)
    last_frame = round(frame_number_list[-1]/ fps, 3)
    
    chunk_name = file_name.replace(".txt", ".mp4")
    ffmpeg_cmd = f"ffmpeg -y -i {video_path} -qscale:v 5 -r {fps} -threads 1 -ss {first_frame} -to {last_frame} -strict -2 {os.path.join(save_path, chunk_name)} -loglevel quiet"
    os.system(ffmpeg_cmd)

def calculate_angle(eye_corner, nose_tip):
    diff = eye_corner - nose_tip

    dot_product = np.dot(diff, np.array([1, 0]))
    magnitude = np.linalg.norm(diff)

    if np.isclose(magnitude, 0):
        return 0

    angle_radians = np.arccos(dot_product / magnitude)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def check_frontality_by_angle(threed_landmarks, threshold):
    left_eye_angle = calculate_angle(threed_landmarks[39, :2],
                                     threed_landmarks[30, :2])
    right_eye_angle = calculate_angle(threed_landmarks[42, :2],
                                      threed_landmarks[30, :2])
    
    angle = left_eye_angle + right_eye_angle
    
    return True if abs(180 - angle) < threshold else False

def extract_landmarks(cropped_frame, landmark_detector, check_frontality=True, frontality_threshold=50):
    threed_landmarks = landmark_detector(cropped_frame)

    if (threed_landmarks is None or len(threed_landmarks) == 0):
        return None

    if check_frontality:
        frontal = check_frontality_by_angle(threed_landmarks=threed_landmarks[0],
                                            threshold=frontality_threshold)
        
        if not frontal:
            return None
    
    return threed_landmarks

