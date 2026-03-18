import time
from collections import deque
from model.POS import POS
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch
from display import ProUI
from utils import estimate_hr_from_rppg, bandpass_filter, normalize_signal

class Monitor:
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
        model=None,
        camera_id=0,
        buffer_seconds=10,
        pos_window_seconds=1.6,
        display_scale=1.0,
    ):
        self.model = model
        self.camera_id = camera_id
        self.buffer_seconds = buffer_seconds
        self.pos_window_seconds = pos_window_seconds
        self.display_scale = display_scale

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头，请检查 camera_id 或摄像头权限。")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError("无法加载 OpenCV Haar Cascade 人脸检测器。")

        self.timestamps = deque()
        self.frame_buffer = deque()
        self.rppg_buffer = deque()
        self.bpm_buffer = deque()
        self.bpm_timestamps = deque()

        self.frame_times = deque(maxlen=120)

        self.bpm_low = 42
        self.bpm_high = 180


        self.last_face_box = None
        self.last_face_time = 0.0
        self.face_hold_seconds = 0.8

        self.ui = ProUI()

    def get_fps(self):
        if len(self.frame_times) < 2:
            return 30.0
        diffs = np.diff(np.array(self.frame_times))
        mean_diff = np.mean(diffs)
        if mean_diff <= 1e-6:
            return 30.0
        return 1.0 / mean_diff

    
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

    def overlap_add_latest(self):
        fps = self.get_fps()
        win_size = max(10, int(self.pos_window_seconds * fps))

        if len(self.rgb_buffer) < win_size:
            return None

        rgb_arr = np.array(self.rgb_buffer, dtype=np.float64)
        rgb_window = rgb_arr[-win_size:]

        h = self.compute_pos_signal(rgb_window)
        if h is None or len(h) == 0:
            return None

        return float(h[-1])

    def trim_buffers(self):
        if len(self.timestamps) == 0:
            return

        current_t = self.timestamps[-1]
        min_t = current_t - self.buffer_seconds

        while len(self.timestamps) > 0 and self.timestamps[0] < min_t:
            self.timestamps.popleft()
            if len(self.frame_buffer) > 0:
                self.frame_buffer.popleft()
            if len(self.rppg_buffer) > 0:
                self.rppg_buffer.popleft()

        bpm_min_t = current_t - max(30, self.buffer_seconds * 3)
        while len(self.bpm_timestamps) > 0 and self.bpm_timestamps[0] < bpm_min_t:
            self.bpm_timestamps.popleft()
            self.bpm_buffer.popleft()

    def compute_signal_quality(self):
        if len(self.rppg_buffer) < 30:
            return 0.0

        sig = np.array(list(self.rppg_buffer)[-90:], dtype=np.float64)
        sig = sig - np.mean(sig)
        return float(np.std(sig))
   
    def get_filtered_rppg_for_display(self):
        if len(self.rppg_buffer) < 8:
            return list(self.rppg_buffer)

        y = np.array(self.rppg_buffer, dtype=np.float64)
        fs = self.get_fps()
        y = bandpass_filter(y, fs, self.bpm_low, self.bpm_high)
        y = normalize_signal(y)
        return y.tolist()

    def draw_roi_overlay(self, frame, face_box, mask):
        display = frame.copy()

        if mask is not None:
            colored_mask = np.zeros_like(display)
            colored_mask[:, :, 1] = mask
            colored_mask[:, :, 0] = (mask * 0.35).astype(np.uint8)
            display = cv2.addWeighted(display, 1.0, colored_mask, 0.24, 0)

        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 210, 80), 2, cv2.LINE_AA)

            for label, (rx, ry, rw, rh) in self.get_roi_regions(frame.shape, face_box):
                overlay = display.copy()
                cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (40, 180, 255), -1)
                display = cv2.addWeighted(overlay, 0.12, display, 0.88, 0)
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (80, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    display, label, (rx, max(18, ry - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 240, 255), 1, cv2.LINE_AA
                )

        return display

    def draw_info(self, frame, face_box, mask, bpm):
        video_view = self.draw_roi_overlay(frame, face_box, mask)

        fps = self.get_fps()
        quality = self.compute_signal_quality()

        h, w = video_view.shape[:2]
        margin = 14
        gap = 16
        panel_w = max(320, min(400, int(w * 0.38)))
        dashboard = np.zeros((h, panel_w, 3), dtype=np.uint8)
        dashboard = self.ui.draw_panel_background(dashboard)

        cv2.putText(
            video_view, "Camera View", (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA
        )

        filtered_rppg = self.get_filtered_rppg_for_display()
        card_gap = 12
        card_w = panel_w - margin * 2
        card_h = 84
        top_y = 18

        bpm_text = "--" if bpm is None else f"{int(round(bpm))}"
        dashboard = self.ui.draw_metric_card(
            dashboard, margin, top_y, card_w, card_h,
            "Heart Rate", bpm_text, "bpm", accent=(0, 210, 255)
        )
        dashboard = self.ui.draw_metric_card(
            dashboard, margin, top_y + card_h + card_gap, card_w, card_h,
            "Frame Rate", f"{fps:.1f}", "fps", accent=(255, 190, 0)
        )
        dashboard = self.ui.draw_quality_card(
            dashboard, margin, top_y + (card_h + card_gap) * 2, card_w, card_h + 12, quality
        )

        wave_y = top_y + (card_h + card_gap) * 3 + 12
        wave_h = h - wave_y - 56
        dashboard = self.ui.draw_simple_waveform(dashboard, filtered_rppg, margin, wave_y, card_w, wave_h)

        dashboard = self.ui.draw_solid_panel(dashboard, margin, h - 42, panel_w - margin * 2, 28, fill_color=(24, 32, 32), border_color=(56, 76, 76))
        cv2.putText(
            dashboard,
            "Press 'q' to quit",
            (margin + 12, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.ui.panel_muted,
            1,
            cv2.LINE_AA,
        )

        separator = np.full((h, gap, 3), 20, dtype=np.uint8)
        cv2.line(separator, (gap // 2, 0), (gap // 2, h), (46, 66, 60), 1, cv2.LINE_AA)
        display = np.hstack([video_view, separator, dashboard])

        if self.display_scale != 1.0:
            hh, ww = display.shape[:2]
            display = cv2.resize(display, (int(ww * self.display_scale), int(hh * self.display_scale)))

        return display

    def run(self):
        current_bpm = None
        last_bpm_update = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                now = time.time()
                self.frame_times.append(now)

                face_box = self.detect_face(frame)

                if frame is not None:
                    self.timestamps.append(now)
                    self.frame_buffer.append(frame)

                    self.trim_buffers()

                    # ⭐ 构造模型输入 (1, T, H, W, C)
                    video_clip = np.array(self.frame_buffer)
                    video_clip = video_clip[None, ...]  # add batch dim

                    # ⭐ 调用模型
                    rppg = self.model.forward(video_clip)

                    if rppg is not None:
                        latest = float(rppg[0, -1])
                        self.rppg_buffer.append(latest)

                    # ⭐ BPM计算
                    if now - last_bpm_update > 1.0 and len(self.rppg_buffer) > 30:
                        fs = self.get_fps()
                        bpm = estimate_hr_from_rppg(
                            list(self.rppg_buffer), fs
                        )

                        if bpm is not None:
                            if current_bpm is None:
                                current_bpm = bpm
                            else:
                                current_bpm = 0.85 * current_bpm + 0.15 * bpm

                        last_bpm_update = now

                display = self.draw_info(frame, face_box, None, current_bpm)
                cv2.imshow("rPPG Monitor", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()