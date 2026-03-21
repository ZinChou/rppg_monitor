import multiprocessing as mp
import os
import queue
import time
from collections import deque

import cv2
import numpy as np
import torch

from display import ProUI
from model.Physformer.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp
from utils import estimate_hr_from_rppg, bandpass_filter, normalize_signal


DEFAULT_MODEL_WINDOW = 160
DEFAULT_IMAGE_SIZE = 128
DEFAULT_INFERENCE_STRIDE = 5


def _put_latest(mp_queue, item):
    while True:
        try:
            mp_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                mp_queue.get_nowait()
            except queue.Empty:
                return


def _drain_latest(mp_queue):
    latest = None
    while True:
        try:
            latest = mp_queue.get_nowait()
        except queue.Empty:
            break
    return latest


def _resolve_device(device):
    if device is not None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _load_physformer_model(device, weights_path=None, model_window=DEFAULT_MODEL_WINDOW):
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(model_window, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        patches=(4, 4, 4),
        dim=96,
        ff_dim=144,
        num_heads=4,
        num_layers=12,
        dropout_rate=0.1,
        theta=0.7,
    ).to(device)
    model.eval()

    warning_message = None
    if weights_path:
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device)
            state_dict = _extract_state_dict(checkpoint)
            clean_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key.replace("module.", "")
                clean_state_dict[clean_key] = value
            missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
            if missing_keys or unexpected_keys:
                warning_message = (
                    "PhysFormer weights loaded with key mismatch. "
                    f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
                )
        else:
            warning_message = f"PhysFormer weights not found: {weights_path}"
    else:
        warning_message = "PhysFormer weights were not provided. Model will run with random initialization."

    return model, warning_message


def _build_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Unable to load OpenCV Haar Cascade face detector.")
    return detector


def _detect_face(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def _expand_face_box(frame_shape, face_box, scale=1.25):
    if face_box is None:
        return None
    h, w = frame_shape[:2]
    x, y, fw, fh = face_box
    cx = x + fw / 2.0
    cy = y + fh / 2.0
    size = int(max(fw, fh) * scale)
    x1 = max(0, int(round(cx - size / 2.0)))
    y1 = max(0, int(round(cy - size / 2.0)))
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    return x1, y1, x2, y2


def _crop_face_frame(frame, face_box, image_size, last_face_frame):
    if face_box is not None:
        expanded = _expand_face_box(frame.shape, face_box)
        if expanded is not None:
            x1, y1, x2, y2 = expanded
            face = frame[y1:y2, x1:x2]
            if face.size != 0:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(face_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                return resized, resized

    if last_face_frame is not None:
        return last_face_frame.copy(), last_face_frame

    fallback = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fallback = cv2.resize(fallback, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return fallback, fallback


def _frames_to_model_input(face_frames, device):
    clip = np.stack(face_frames, axis=0).astype(np.float32)
    clip = (clip - 127.5) / 128.0
    clip = np.transpose(clip, (3, 0, 1, 2))
    tensor = torch.from_numpy(clip).unsqueeze(0).to(device)
    return tensor


def camera_capture_worker(camera_id, target_fps, frame_queue, processing_queue, stop_event):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    if not cap.isOpened():
        _put_latest(
            frame_queue,
            {
                "type": "error",
                "message": f"Unable to open camera: {camera_id}",
                "timestamp": time.time(),
            },
        )
        return

    frame_interval = 1.0 / max(target_fps, 1e-6)
    next_frame_time = time.perf_counter()

    try:
        while not stop_event.is_set():
            now = time.perf_counter()
            if now < next_frame_time:
                time.sleep(min(frame_interval / 2.0, next_frame_time - now))
                continue

            ret, frame = cap.read()
            timestamp = time.time()
            if not ret:
                _put_latest(
                    frame_queue,
                    {
                        "type": "error",
                        "message": "Camera read failed.",
                        "timestamp": timestamp,
                    },
                )
                time.sleep(frame_interval)
                next_frame_time += frame_interval
                continue

            packet = {
                "type": "frame",
                "frame": frame,
                "timestamp": timestamp,
            }
            _put_latest(frame_queue, packet)
            _put_latest(processing_queue, packet)

            next_frame_time += frame_interval
            behind = time.perf_counter() - next_frame_time
            if behind > frame_interval:
                next_frame_time = time.perf_counter()
    finally:
        cap.release()


def rppg_worker(
    processing_queue,
    result_queue,
    stop_event,
    buffer_seconds,
    target_fps,
    bpm_low,
    bpm_high,
    model_weights,
    model_device,
    model_window,
    inference_stride,
):
    device = _resolve_device(model_device)

    try:
        model, model_warning = _load_physformer_model(
            device=device,
            weights_path=model_weights,
            model_window=model_window,
        )
        face_detector = _build_face_detector()
    except Exception as exc:
        _put_latest(
            result_queue,
            {
                "type": "error",
                "message": f"Failed to initialize deep rPPG worker: {exc}",
                "timestamp": time.time(),
            },
        )
        return

    face_frames = deque(maxlen=model_window)
    clip_timestamps = deque(maxlen=model_window)
    signal_timestamps = deque()
    rppg_buffer = deque()
    bpm_buffer = deque()
    bpm_timestamps = deque()
    frame_times = deque(maxlen=120)

    current_bpm = None
    last_bpm_update = 0.0
    latest_face_box = None
    last_face_frame = None
    total_frames_seen = 0
    infer_count = 0
    last_inference_at = 0.0

    def trim_buffers(current_t):
        min_t = current_t - buffer_seconds
        while signal_timestamps and signal_timestamps[0] < min_t:
            signal_timestamps.popleft()
            if rppg_buffer:
                rppg_buffer.popleft()

        bpm_min_t = current_t - max(30, buffer_seconds * 3)
        while bpm_timestamps and bpm_timestamps[0] < bpm_min_t:
            bpm_timestamps.popleft()
            bpm_buffer.popleft()

    def compute_fps():
        if len(frame_times) < 2:
            return float(target_fps)
        diffs = np.diff(np.array(frame_times, dtype=np.float64))
        mean_diff = np.mean(diffs)
        if mean_diff <= 1e-6:
            return float(target_fps)
        return float(1.0 / mean_diff)

    def compute_quality(fps=30.0):
        """
        基于周期性评定 rPPG 信号质量
        返回值: 0.0 ~ 1.0，越大表示周期性越强、质量越好
        """
        min_len = int(fps * 3)   # 至少3秒
        if len(rppg_buffer) < min_len:
            return 0.0

        # 取最近 8 秒，过短不稳定，过长对实时性不好
        win_len = min(len(rppg_buffer), int(fps * 8))
        sig = np.array(list(rppg_buffer)[-win_len:], dtype=np.float64)

        # 去均值
        sig = sig - np.mean(sig)

        # 振幅太小，直接认为质量差
        sig_std = np.std(sig)
        if sig_std < 1e-6:
            return 0.0

        # 归一化
        sig = sig / sig_std

        # 自相关
        acf = np.correlate(sig, sig, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / (acf[0] + 1e-8)

        # 心率范围: 40~180 BPM
        # 对应周期范围
        min_bpm = 40
        max_bpm = 180
        min_lag = int(fps * 60 / max_bpm)  # 最短周期
        max_lag = int(fps * 60 / min_bpm)  # 最长周期

        if max_lag >= len(acf):
            max_lag = len(acf) - 1
        if min_lag >= max_lag:
            return 0.0

        search_region = acf[min_lag:max_lag + 1]
        peak = np.max(search_region)

        # 峰值位置对应主周期
        peak_idx = np.argmax(search_region) + min_lag

        # 再检查该周期的倍周期是否也有一定一致性
        harmonic_score = 0.0
        if 2 * peak_idx < len(acf):
            harmonic_score = max(0.0, acf[2 * peak_idx])

        # 组合评分
        # peak 越高说明周期性越明显
        # harmonic_score 越高说明重复节律更稳定
        quality = 0.7 * peak + 0.3 * harmonic_score

        # 限制到 0~1
        quality = float(np.clip(quality, 0.0, 1.0))
        return quality

    while not stop_event.is_set():
        try:
            packet = processing_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if packet.get("type") != "frame":
            continue

        frame = packet["frame"]
        timestamp = packet["timestamp"]
        frame_times.append(timestamp)
        total_frames_seen += 1

        latest_face_box = _detect_face(frame, face_detector)
        start = time.time()
        face_frame, last_face_frame = _crop_face_frame(
            frame,
            latest_face_box,
            DEFAULT_IMAGE_SIZE,
            last_face_frame,
        )
        print(time.time() - start)
        face_frames.append(face_frame)
        clip_timestamps.append(timestamp)

        should_infer = (
            len(face_frames) == model_window
            and (infer_count == 0 or total_frames_seen % max(1, inference_stride) == 0)
        )

        if should_infer:
            try:
                inputs = _frames_to_model_input(list(face_frames), device)
                with torch.no_grad():
                    rppg_pred, _, _, _ = model(inputs, gra_sharp=2.0)
                pred = rppg_pred.detach().float().cpu().numpy().reshape(-1)
                pred = normalize_signal(pred)

                if infer_count == 0:
                    new_count = len(pred)
                    new_values = pred
                    new_timestamps = list(clip_timestamps)
                else:
                    new_count = min(max(1, inference_stride), len(pred), len(clip_timestamps))
                    new_values = pred[-new_count:]
                    new_timestamps = list(clip_timestamps)[-new_count:]

                for sample_t, sample_v in zip(new_timestamps, new_values):
                    signal_timestamps.append(sample_t)
                    rppg_buffer.append(float(sample_v))

                infer_count += 1
                last_inference_at = timestamp
            except Exception as exc:
                _put_latest(
                    result_queue,
                    {
                        "type": "error",
                        "message": f"Deep rPPG inference failed: {exc}",
                        "timestamp": timestamp,
                    },
                )
                return

        trim_buffers(timestamp)

        fs = compute_fps()
        if timestamp - last_bpm_update > 1.0 and len(rppg_buffer) > max(30, int(fs * 4)):
            bpm = estimate_hr_from_rppg(list(rppg_buffer), fs, bpm_low=bpm_low, bpm_high=bpm_high)
            if bpm is not None:
                if current_bpm is None:
                    current_bpm = bpm
                else:
                    current_bpm = 0.85 * current_bpm + 0.15 * bpm
                bpm_timestamps.append(timestamp)
                bpm_buffer.append(current_bpm)
            last_bpm_update = timestamp

        _put_latest(
            result_queue,
            {
                "type": "result",
                "timestamp": timestamp,
                "face_box": latest_face_box,
                "rppg_values": list(rppg_buffer),
                "bpm_values": list(bpm_buffer),
                "current_bpm": current_bpm,
                "quality": compute_quality(),
                "fps": fs,
                "device": device,
                "model_window": model_window,
                "inference_stride": inference_stride,
                "model_warning": model_warning,
                "warmup_progress": min(1.0, len(face_frames) / float(model_window)),
                "last_inference_at": last_inference_at,
            },
        )


class Monitor:
    def __init__(
        self,
        model=None,
        camera_id=0,
        buffer_seconds=10,
        pos_window_seconds=1.6,
        display_scale=1.0,
        target_fps=30.0,
        model_weights=None,
        model_device=None,
        model_window=DEFAULT_MODEL_WINDOW,
        inference_stride=DEFAULT_INFERENCE_STRIDE,
    ):
        self.model = model
        self.camera_id = camera_id
        self.buffer_seconds = buffer_seconds
        self.pos_window_seconds = pos_window_seconds
        self.display_scale = display_scale
        self.target_fps = float(target_fps)
        self.model_weights = model_weights
        self.model_device = model_device
        self.model_window = int(model_window)
        self.inference_stride = int(inference_stride)

        self.bpm_low = 42
        self.bpm_high = 180
        self.ui = ProUI()

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError("Unable to load OpenCV Haar Cascade face detector.")

    def build_roi_mask_from_face_box(self, frame_shape, face_box):
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if face_box is None:
            return mask

        x, y, fw, fh = face_box
        regions = [
            (int(x + 0.22 * fw), int(y + 0.10 * fh), int(0.56 * fw), int(0.18 * fh)),
            (int(x + 0.12 * fw), int(y + 0.45 * fh), int(0.22 * fw), int(0.20 * fh)),
            (int(x + 0.66 * fw), int(y + 0.45 * fh), int(0.22 * fw), int(0.20 * fh)),
        ]

        for rx, ry, rw, rh in regions:
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

    def get_filtered_rppg_for_display(self, rppg_values, fps):
        if len(rppg_values) < 8:
            return list(rppg_values)

        y = np.array(rppg_values, dtype=np.float64)
        y = bandpass_filter(y, fps, self.bpm_low, self.bpm_high)
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
                    display,
                    label,
                    (rx, max(18, ry - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (210, 240, 255),
                    1,
                    cv2.LINE_AA,
                )

        return display

    def draw_info(self, frame, face_box, mask, bpm, rppg_values, fps, quality):
        video_view = self.draw_roi_overlay(frame, face_box, mask)

        h, w = video_view.shape[:2]
        margin = 14
        gap = 16
        panel_w = max(320, min(400, int(w * 0.38)))
        dashboard = np.zeros((h, panel_w, 3), dtype=np.uint8)
        dashboard = self.ui.draw_panel_background(dashboard)

        cv2.putText(
            video_view,
            "Camera View",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

        filtered_rppg = self.get_filtered_rppg_for_display(rppg_values, fps)
        card_gap = 12
        card_w = panel_w - margin * 2
        card_h = 84
        top_y = 18

        bpm_text = "--" if bpm is None else f"{int(round(bpm))}"
        dashboard = self.ui.draw_metric_card(
            dashboard, margin, top_y, card_w, card_h, "Heart Rate", bpm_text, "bpm", accent=(0, 210, 255)
        )
        dashboard = self.ui.draw_metric_card(
            dashboard,
            margin,
            top_y + card_h + card_gap,
            card_w,
            card_h,
            "Frame Rate",
            f"{fps:.1f}",
            "fps",
            accent=(255, 190, 0),
        )
        dashboard = self.ui.draw_quality_card(
            dashboard, margin, top_y + (card_h + card_gap) * 2, card_w, card_h + 12, quality
        )

        wave_y = top_y + (card_h + card_gap) * 3 + 12
        wave_h = h - wave_y - 56
        dashboard = self.ui.draw_simple_waveform(dashboard, filtered_rppg, margin, wave_y, card_w, wave_h)

        dashboard = self.ui.draw_solid_panel(
            dashboard,
            margin,
            h - 42,
            panel_w - margin * 2,
            28,
            fill_color=(24, 32, 32),
            border_color=(56, 76, 76),
        )
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
        ctx = mp.get_context("spawn")
        frame_queue = ctx.Queue(maxsize=2)
        processing_queue = ctx.Queue(maxsize=2)
        result_queue = ctx.Queue(maxsize=2)
        stop_event = ctx.Event()

        camera_process = ctx.Process(
            target=camera_capture_worker,
            args=(self.camera_id, self.target_fps, frame_queue, processing_queue, stop_event),
            daemon=True,
        )
        compute_process = ctx.Process(
            target=rppg_worker,
            args=(
                processing_queue,
                result_queue,
                stop_event,
                self.buffer_seconds,
                self.target_fps,
                self.bpm_low,
                self.bpm_high,
                self.model_weights,
                self.model_device,
                self.model_window,
                self.inference_stride,
            ),
            daemon=True,
        )

        camera_process.start()
        compute_process.start()

        latest_frame = None
        latest_result = {
            "face_box": None,
            "rppg_values": [],
            "current_bpm": None,
            "quality": 0.0,
            "fps": self.target_fps,
            "model_warning": None,
        }

        try:
            while True:
                frame_packet = _drain_latest(frame_queue)
                if frame_packet is not None:
                    if frame_packet.get("type") == "error":
                        raise RuntimeError(frame_packet.get("message", "Camera worker failed."))
                    latest_frame = frame_packet["frame"]

                result_packet = _drain_latest(result_queue)
                if result_packet is not None:
                    if result_packet.get("type") == "error":
                        raise RuntimeError(result_packet.get("message", "rPPG worker failed."))
                    latest_result = result_packet

                if latest_frame is None:
                    if not camera_process.is_alive():
                        raise RuntimeError("Camera process exited unexpectedly.")
                    time.sleep(0.01)
                    continue

                face_box = latest_result.get("face_box")
                mask = self.build_roi_mask_from_face_box(latest_frame.shape, face_box) if face_box is not None else None

                display = self.draw_info(
                    latest_frame,
                    face_box,
                    mask,
                    latest_result.get("current_bpm"),
                    latest_result.get("rppg_values", []),
                    latest_result.get("fps", self.target_fps),
                    latest_result.get("quality", 0.0),
                )
                cv2.imshow("rPPG Monitor (Deep)", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if not camera_process.is_alive():
                    raise RuntimeError("Camera process exited unexpectedly.")
                if not compute_process.is_alive():
                    raise RuntimeError("rPPG process exited unexpectedly.")
        finally:
            stop_event.set()
            for proc in (camera_process, compute_process):
                proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
            cv2.destroyAllWindows()
