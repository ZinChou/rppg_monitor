from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import h5py
from torchvision import transforms
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

img_size = 128
clip_frames = 160   
    
class Data_Processor:
    def __init__(self, image_size=128, device='cuda'):
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=0,
            keep_all=True,
            post_process=False,
            device=device
        )
        self.image_size = image_size
        self.device = device

    
    def video2tensor(self, video):
        

        processed_frames = []
        last_roi = None  # 保存上一帧检测到的人脸区域

        for item in video:
            # 兼容 frame 或 (ret, frame)
            if isinstance(item, tuple) and len(item) == 2:
                ret, frame = item
                if not ret or frame is None:
                    continue
            else:
                frame = item
                if frame is None:
                    continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            boxes, probs = self.mtcnn.detect(img)

            if boxes is not None and len(boxes) > 0:
                # 取面积最大的人脸
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                best_idx = np.argmax(areas)
                x1, y1, x2, y2 = boxes[best_idx]

                # 防越界
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(frame_rgb.shape[1], int(x2))
                y2 = min(frame_rgb.shape[0], int(y2))

                face = frame_rgb[y1:y2, x1:x2]

                if face.size != 0:
                    last_roi = face.copy()
                else:
                    face = last_roi if last_roi is not None else frame_rgb

            else:
                # 当前帧没检测到，用上一帧 ROI
                face = last_roi if last_roi is not None else frame_rgb

            face = cv2.resize(face, (self.image_size, self.image_size))
            processed_frames.append(face)

        if len(processed_frames) == 0:
            return torch.empty(0)
        
        video= torch.stack(processed_frames, dim=0)
        video = (video - 127.5) / 128
        video = video.transpose((3, 0, 1, 2))
        video = video.unsqueeze(0)
        
        return video

