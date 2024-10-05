import cv2
import copy
import torch
import numpy as np
from PIL import Image
from facexlib.detection import init_detection_model
from facexlib.alignment import landmark_98_to_68

from src.utils.preprocess.load_mats import load_lm3d
from src.utils.preprocess.preprocess import align_img
from src.utils.preprocess.helper import init_alignment_model, calc_eye_close_ratio


class SadTalkerPreprocess:
    def __init__(self, device, pic_size, model_path, lm3d_mat_path):
        self.device = device
        self.pic_size = pic_size

        self.lm3d_std = load_lm3d(lm3d_mat_path)
        self.detector = init_alignment_model('awing_fan', device=device, model_rootpath=model_path)   
        self.det_net = init_detection_model('retinaface_resnet50', half=False, device=device, model_rootpath=model_path) 

    def __call__(self, frame):
        face, crop = self.crop(frame)
        if face is None or crop is None:
            return None, None, None

        face = cv2.resize(face, (self.pic_size, self.pic_size))
        face_for_rendering = face.copy()
        crop_for_rendering = copy.deepcopy(crop)

        landmarks, eye_close_ratio = self.extract_landmarks(face, calc_eye_ratio=True)
        landmark_for_rendering = copy.deepcopy(landmarks)
        if landmarks is None:
            print("No landmark is detected on cropped face!")
            return None, None, None

        landmarks_ret = copy.deepcopy(landmarks) / self.pic_size
        face = Image.fromarray(face)
        W, H = face.size
        landmarks = landmarks.reshape([-1, 2])
        if np.mean(landmarks) == -1:
            landmarks = (self.lm3d_std[:, :2]+1)/2.
            landmarks = np.concatenate([landmarks[:, :1]*W, landmarks[:, 1:2]*H], 1)
        else:
            landmarks[:, -1] = H - 1 - landmarks[:, -1]

        _, face, landmarks, _ = align_img(face, landmarks, self.lm3d_std)
        torch_face = torch.tensor(np.array(face)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        face_for_rendering = torch.tensor(face_for_rendering/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return torch_face, face_for_rendering, crop, crop_for_rendering, landmarks_ret, eye_close_ratio
    
    def extract_landmarks(self, face, calc_eye_ratio=False):
        with torch.no_grad():
            bboxes = self.det_net.detect_faces(face, 0.97)
            if len(bboxes) == 0:
                print("No faces is detected!")
                return None

            bboxes = bboxes[0]
            face = face[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]

            landmarks = self.detector.get_landmarks(face)

            eye_close_ratio = None
            if calc_eye_ratio:
                _landmarks = copy.deepcopy(landmarks)
                _landmarks[:, 0] += int(bboxes[0])
                _landmarks[:, 1] += int(bboxes[1])
                eye_close_ratio = calc_eye_close_ratio(_landmarks[None])

            landmarks = landmark_98_to_68(landmarks)
            landmarks[:,0] += int(bboxes[0])
            landmarks[:,1] += int(bboxes[1])
        
        return landmarks, eye_close_ratio

    def crop(self, frame):
        landmarks, _ = self.extract_landmarks(frame)
        if landmarks is None:
            return None, None

        rsize, crop = self.align_face(img=Image.fromarray(frame),
                                      lm=landmarks,
                                      output_size=512)
        face = cv2.resize(frame, (rsize[0], rsize[1]))
        face = face[crop[1]:crop[3], crop[0]:crop[2]]
        return face, crop
         
    def align_face(self, img, lm, output_size=1024):
        """
        :param filepath: str
        :return: PIL Image
        """
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]  # Addition of binocular difference and double mouth difference
        x /= np.hypot(*x)   # hypot函数计算直角三角形的斜边长，用斜边长对三角形两条直边做归一化
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)    # 双眼差和眼嘴差，选较大的作为基准尺度
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])   # 定义四边形，以面部基准位置为中心上下左右平移得到四个顶点
        qsize = np.hypot(*x) * 2    # 定义四边形的大小（边长），为基准尺度的2倍

        # Shrink.
        # 如果计算出的四边形太大了，就按比例缩小它
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
        else:
            rsize = (int(np.rint(float(img.size[0]))), int(np.rint(float(img.size[1]))))

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))

        # Save aligned image.
        return rsize, crop
