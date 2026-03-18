import cv2
import time
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, welch

class POS():
    """
    使用摄像头实时进行 rPPG 和心率监测（POS算法）
    - 人脸检测：OpenCV Haar Cascade
    - ROI：基于人脸框的额头 + 双脸颊
    - 信号提取：平均RGB
    - rPPG算法：POS
    - HR估计：Welch频谱峰值
    - 可视化：OpenCV 产品级 HUD
    """

    def __init__(
        self,
        pos_window_seconds=1.6,
        bpm_low=42,
        bpm_high=180,
    ):
        self.pos_window_seconds = pos_window_seconds
        self.bpm_low = bpm_low
        self.bpm_high = bpm_high

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)

        self.last_face_box = None
        self.last_face_time = 0.0
        self.face_hold_seconds = 0.8

        self.fs = 30

    def detect_face(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100),
        )

        if len(faces) == 0:
            now = time.time()
            if self.last_face_box is not None and (now - self.last_face_time) < self.face_hold_seconds:
                return self.last_face_box
            return None

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        face = tuple(faces[0])

        self.last_face_box = face
        self.last_face_time = time.time()
        return face

    def build_roi_mask_from_face_box(self, frame_shape, face_box):
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if face_box is None:
            return mask

        x, y, fw, fh = face_box

        forehead = (
            int(x + 0.22 * fw),
            int(y + 0.10 * fh),
            int(0.56 * fw),
            int(0.18 * fh),
        )

        left_cheek = (
            int(x + 0.12 * fw),
            int(y + 0.45 * fh),
            int(0.22 * fw),
            int(0.20 * fh),
        )

        right_cheek = (
            int(x + 0.66 * fw),
            int(y + 0.45 * fh),
            int(0.22 * fw),
            int(0.20 * fh),
        )

        for rx, ry, rw, rh in [forehead, left_cheek, right_cheek]:
            x1 = max(0, rx)
            y1 = max(0, ry)
            x2 = min(w, rx + rw)
            y2 = min(h, ry + rh)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        return mask

    def get_roi_regions(self, frame_shape, face_box):
        if face_box is None:
            return []

        h, w = frame_shape[:2]
        x, y, fw, fh = face_box
        regions = [
            ("Forehead", (int(x + 0.22 * fw), int(y + 0.10 * fh), int(0.56 * fw), int(0.18 * fh))),
            ("Left Cheek", (int(x + 0.12 * fw), int(y + 0.45 * fh), int(0.22 * fw), int(0.20 * fh))),
            ("Right Cheek", (int(x + 0.66 * fw), int(y + 0.45 * fh), int(0.22 * fw), int(0.20 * fh))),
        ]

        clipped = []
        for label, (rx, ry, rw, rh) in regions:
            x1 = max(0, rx)
            y1 = max(0, ry)
            x2 = min(w, rx + rw)
            y2 = min(h, ry + rh)
            if x2 > x1 and y2 > y1:
                clipped.append((label, (x1, y1, x2 - x1, y2 - y1)))
        return clipped

    def extract_mean_rgb(self, frame_bgr, mask):
        if mask is None or np.count_nonzero(mask) < 50:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pixels = frame_rgb[mask > 0]

        if pixels.size == 0:
            return None

        mean_rgb = np.mean(pixels, axis=0)
        return mean_rgb

    def compute_pos_signal(self, rgb_window):
        X = np.asarray(rgb_window, dtype=np.float64).T
        if X.shape[1] < 2:
            return None

        mean_color = np.mean(X, axis=1, keepdims=True)
        mean_color[mean_color == 0] = 1e-8
        Xn = X / mean_color - 1.0

        projection = np.array([
            [0, 1, -1],
            [-2, 1, 1]
        ], dtype=np.float64)

        S = projection @ Xn

        std0 = np.std(S[0]) + 1e-8
        std1 = np.std(S[1]) + 1e-8
        alpha = std0 / std1

        h = S[0] + alpha * S[1]
        h = h - np.mean(h)
        return h

    def forward(self, video):
        """
        video: numpy array (B, T, H, W, C)  # 注意一般是 H, W
        return: rPPG (B, T)
        """
        B, T, H, W, C = video.shape
        outputs = []

        for b in range(B):
            frames = video[b]

            rgb_buffer = []
            rppg_signal = []

            for t in range(T):
                frame = frames[t].astype(np.uint8)

                # 1️⃣ 人脸检测
                face_box = self.detect_face(frame)

                if face_box is None:
                    rppg_signal.append(0.0)
                    continue

                # 2️⃣ ROI mask
                mask = self.build_roi_mask_from_face_box(frame.shape, face_box)

                # 3️⃣ mean RGB
                mean_rgb = self.extract_mean_rgb(frame, mask)

                if mean_rgb is None:
                    rppg_signal.append(0.0)
                    continue

                rgb_buffer.append(mean_rgb)

                # 4️⃣ POS window
                win_size = int(self.pos_window_seconds * self.fs)  # 默认30fps
                if len(rgb_buffer) < win_size:
                    rppg_signal.append(0.0)
                    continue

                rgb_window = np.array(rgb_buffer[-win_size:])

                # 5️⃣ POS计算
                h = self.compute_pos_signal(rgb_window)

                if h is None or len(h) == 0:
                    rppg_signal.append(0.0)
                else:
                    rppg_signal.append(float(h[-1]))

            outputs.append(rppg_signal)

        return np.array(outputs)