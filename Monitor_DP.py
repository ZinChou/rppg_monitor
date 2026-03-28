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
from utils import (
    bandpass_filter,
    estimate_hr_from_rppg,
    estimate_hrv_from_rppg,
    estimate_resp_rate_from_rppg,
    normalize_signal,
    should_accept_metric_update,
)


DEFAULT_MODEL_WINDOW = 160
DEFAULT_IMAGE_SIZE = 128
DEFAULT_INFERENCE_STRIDE = 80


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
                # face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
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
            # ret = True
            # frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 黑色图像（480x640）
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
    import time
    import threading
    import queue as std_queue
    from queue import Queue
    from collections import deque

    import numpy as np
    import torch

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

    # -----------------------------
    # 共享状态
    # -----------------------------
    state_lock = threading.Lock()

    face_frames = deque(maxlen=model_window)
    clip_timestamps = deque(maxlen=model_window)

    signal_timestamps = deque()
    rppg_buffer = deque()

    bpm_timestamps = deque()
    bpm_buffer = deque()

    pending_rppg_points = deque()   # [(sample_t, sample_v), ...]
    frame_times = deque(maxlen=120)

    latest_face_box = None
    last_face_frame = None

    next_rppg_emit_time = None
    current_bpm = None
    current_hrv = None
    current_resp_rate = None
    latest_quality = 0.0
    last_bpm_update = 0.0
    last_inference_at = 0.0

    total_frames_seen = 0
    infer_count = 0

    # -----------------------------
    # 线程内队列
    # -----------------------------
    infer_queue = Queue(maxsize=2)   # 主线程 -> 推理线程
    stats_queue = Queue(maxsize=4)   # 信号线程 -> 统计线程

    # -----------------------------
    # 工具函数
    # -----------------------------
    def trim_buffers_locked(current_t):
        min_t = current_t - buffer_seconds

        while signal_timestamps and signal_timestamps[0] < min_t:
            signal_timestamps.popleft()
            if rppg_buffer:
                rppg_buffer.popleft()

        while pending_rppg_points and pending_rppg_points[0][0] < min_t:
            pending_rppg_points.popleft()

        bpm_min_t = current_t - max(30, buffer_seconds * 3)
        while bpm_timestamps and bpm_timestamps[0] < bpm_min_t:
            bpm_timestamps.popleft()
            if bpm_buffer:
                bpm_buffer.popleft()

    def compute_fps_locked():
        if len(frame_times) < 2:
            return float(target_fps)
        diffs = np.diff(np.array(frame_times, dtype=np.float64))
        mean_diff = np.mean(diffs)
        if mean_diff <= 1e-6:
            return float(target_fps)
        return float(1.0 / mean_diff)

    def compute_quality_from_signal(sig_values, fps=30.0):
        """
        基于周期性评定 rPPG 信号质量
        返回值: 0.0 ~ 1.0，越大表示周期性越强、质量越好
        """
        min_len = int(fps * 3)
        if len(sig_values) < min_len:
            return 0.0

        win_len = min(len(sig_values), int(fps * 8))
        sig = np.array(sig_values[-win_len:], dtype=np.float64)

        sig = sig - np.mean(sig)
        sig_std = np.std(sig)
        if sig_std < 1e-6:
            return 0.0

        sig = sig / sig_std

        acf = np.correlate(sig, sig, mode="full")
        acf = acf[len(acf) // 2 :]
        acf = acf / (acf[0] + 1e-8)

        min_bpm_local = 40
        max_bpm_local = 180
        min_lag = int(fps * 60 / max_bpm_local)
        max_lag = int(fps * 60 / min_bpm_local)

        if max_lag >= len(acf):
            max_lag = len(acf) - 1
        if min_lag >= max_lag:
            return 0.0

        search_region = acf[min_lag : max_lag + 1]
        peak = np.max(search_region)
        peak_idx = np.argmax(search_region) + min_lag

        harmonic_score = 0.0
        if 2 * peak_idx < len(acf):
            harmonic_score = max(0.0, acf[2 * peak_idx])

        quality = 0.7 * peak + 0.3 * harmonic_score
        return float(np.clip(quality, 0.0, 1.0))

    # -----------------------------
    # 推理线程
    # -----------------------------
    def inference_loop():
        nonlocal infer_count, last_inference_at, next_rppg_emit_time

        while not stop_event.is_set():
            try:
                item = infer_queue.get(timeout=0.1)
            except std_queue.Empty:
                continue

            if item is None:
                break

            frames_list, timestamps_list, packet_ts = item

            try:
                inputs = _frames_to_model_input(frames_list, device)

                with torch.no_grad():
                    rppg_pred, _, _, _ = model(inputs, gra_sharp=2.0)

                pred = rppg_pred.detach().float().cpu().numpy().reshape(-1)
                pred = normalize_signal(pred)

                with state_lock:
                    local_infer_count = infer_count

                if local_infer_count == 0:
                    new_count = len(pred)
                    new_values = pred
                    new_timestamps = timestamps_list
                else:
                    new_count = min(
                        max(1, inference_stride),
                        len(pred),
                        len(timestamps_list),
                    )
                    new_values = pred[-new_count:]
                    new_timestamps = timestamps_list[-new_count:]

                with state_lock:
                    for sample_t, sample_v in zip(new_timestamps, new_values):
                        pending_rppg_points.append((sample_t, float(sample_v)))

                    if next_rppg_emit_time is None and pending_rppg_points:
                        # 第一次开始释放时，用当前真实时间作为节拍基准
                        next_rppg_emit_time = time.time()

                    infer_count += 1
                    last_inference_at = packet_ts

            except Exception as exc:
                _put_latest(
                    result_queue,
                    {
                        "type": "error",
                        "message": f"Deep rPPG inference failed: {exc}",
                        "timestamp": packet_ts,
                    },
                )
                stop_event.set()
                return

    # -----------------------------
    # 信号释放线程
    # pending_rppg_points -> rppg_buffer
    # -----------------------------
    def signal_emit_loop():
        nonlocal next_rppg_emit_time

        while not stop_event.is_set():
            packet_for_stats = None

            with state_lock:
                fs = compute_fps_locked()
                emit_interval = 1.0 / max(fs, 1e-6)
                now_ts = time.time()

                if (
                    pending_rppg_points
                    and next_rppg_emit_time is not None
                    and now_ts >= next_rppg_emit_time
                ):
                    sample_t, sample_v = pending_rppg_points.popleft()
                    signal_timestamps.append(sample_t)
                    rppg_buffer.append(sample_v)

                    trim_buffers_locked(sample_t)

                    packet_for_stats = {
                        "timestamp": sample_t,
                        "rppg_values": list(rppg_buffer),
                        "fs": fs,
                    }

                    next_rppg_emit_time += emit_interval

                    if not pending_rppg_points:
                        next_rppg_emit_time = None

            if packet_for_stats is not None:
                try:
                    stats_queue.put_nowait(packet_for_stats)
                except std_queue.Full:
                    pass

            time.sleep(0.005)

    # -----------------------------
    # 统计线程
    # -----------------------------
    def stats_loop():
        nonlocal current_bpm, current_hrv, current_resp_rate, latest_quality, last_bpm_update

        while not stop_event.is_set():
            try:
                item = stats_queue.get(timeout=0.1)
            except std_queue.Empty:
                continue

            ts = item["timestamp"]
            fs = item["fs"]
            rppg_values = item["rppg_values"]

            quality = compute_quality_from_signal(rppg_values, fs)

            bpm = None
            if len(rppg_values) > max(30, int(fs * 4)) and (ts - last_bpm_update > 1.0):
                bpm = estimate_hr_from_rppg(
                    rppg_values,
                    fs,
                    bpm_low=bpm_low,
                    bpm_high=bpm_high,
                )

            with state_lock:
                latest_quality = quality

                if should_accept_metric_update(
                    bpm,
                    current=current_bpm,
                    min_value=bpm_low,
                    max_value=bpm_high,
                    max_abs_delta=18.0,
                    max_rel_delta=0.22,
                ):
                    if current_bpm is None:
                        current_bpm = bpm
                    else:
                        current_bpm = 0.85 * current_bpm + 0.15 * bpm

                    bpm_timestamps.append(ts)
                    bpm_buffer.append(current_bpm)
                    last_bpm_update = ts

                hrv = estimate_hrv_from_rppg(rppg_values, fs, bpm_low=bpm_low, bpm_high=bpm_high)
                if should_accept_metric_update(
                    hrv,
                    current=current_hrv,
                    min_value=8.0,
                    max_value=250.0,
                    max_abs_delta=45.0,
                    max_rel_delta=0.6,
                ):
                    if current_hrv is None:
                        current_hrv = hrv
                    else:
                        current_hrv = 0.8 * current_hrv + 0.2 * hrv

                resp_rate = estimate_resp_rate_from_rppg(rppg_values, fs)
                if should_accept_metric_update(
                    resp_rate,
                    current=current_resp_rate,
                    min_value=6.0,
                    max_value=30.0,
                    max_abs_delta=6.0,
                    max_rel_delta=0.35,
                ):
                    if current_resp_rate is None:
                        current_resp_rate = resp_rate
                    else:
                        current_resp_rate = 0.8 * current_resp_rate + 0.2 * resp_rate

    # -----------------------------
    # 启动线程
    # -----------------------------
    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    emit_thread = threading.Thread(target=signal_emit_loop, daemon=True)
    stats_thread = threading.Thread(target=stats_loop, daemon=True)

    infer_thread.start()
    emit_thread.start()
    stats_thread.start()

    # -----------------------------
    # 主循环：收帧 / 检脸 / 裁脸 / 投递推理任务 / 回传结果
    # -----------------------------
    try:
        while not stop_event.is_set():
            try:
                packet = processing_queue.get(timeout=0.1)
            except std_queue.Empty:
                continue

            if packet.get("type") != "frame":
                continue

            frame = packet["frame"]
            timestamp = packet["timestamp"]

            local_face_box = _detect_face(frame, face_detector)

            with state_lock:
                frame_times.append(timestamp)
                total_frames_seen += 1

            face_frame, cropped_last_face_frame = _crop_face_frame(
                frame,
                local_face_box,
                DEFAULT_IMAGE_SIZE,
                last_face_frame,
            )

            should_infer = False
            frames_list = None
            timestamps_list = None

            with state_lock:
                latest_face_box = local_face_box
                last_face_frame = cropped_last_face_frame

                face_frames.append(face_frame)
                clip_timestamps.append(timestamp)

                should_infer = (
                    len(face_frames) == model_window
                    and (
                        infer_count == 0
                        or total_frames_seen % max(1, inference_stride) == 0
                    )
                )

                if should_infer:
                    frames_list = list(face_frames)
                    timestamps_list = list(clip_timestamps)

                fs = compute_fps_locked()

                result_packet = {
                    "type": "result",
                    "timestamp": timestamp,
                    "face_box": latest_face_box,
                    "rppg_values": list(rppg_buffer),
                    "bpm_values": list(bpm_buffer),
                    "current_bpm": current_bpm,
                    "current_hrv": current_hrv,
                    "current_resp_rate": current_resp_rate,
                    "quality": latest_quality,
                    "fps": fs,
                    "device": device,
                    "model_window": model_window,
                    "inference_stride": inference_stride,
                    "model_warning": model_warning,
                    "warmup_progress": min(1.0, len(face_frames) / float(model_window)),
                    "last_inference_at": last_inference_at,
                }

            _put_latest(result_queue, result_packet)

            if should_infer:
                try:
                    infer_queue.put_nowait((frames_list, timestamps_list, timestamp))
                except std_queue.Full:
                    # 丢弃本次推理任务，避免堆积导致时延越来越大
                    pass

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()

        try:
            infer_queue.put_nowait(None)
        except Exception:
            pass

        infer_thread.join(timeout=1.0)
        emit_thread.join(timeout=1.0)
        stats_thread.join(timeout=1.0)


class Monitor_DP:
    def __init__(
        self,
        model=None,
        camera_id=0,
        buffer_seconds=10,
        pos_window_seconds=1.6,
        display_scale=1.0,
        target_fps=30.0,
        model_weights='checkpoints/Physformer.pkl',
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

    def draw_info(self, frame, face_box, mask, bpm, hrv, resp_rate, rppg_values, fps):
        video_view = self.draw_roi_overlay(frame, face_box, mask)

        h, w = video_view.shape[:2]
        margin = 14
        gap = 16
        panel_w = max(320, min(400, int(w * 0.38)))
        dashboard = np.zeros((h, panel_w, 3), dtype=np.uint8)
        dashboard = self.ui.draw_panel_background(dashboard)

        self.ui.draw_text(
            video_view,
            "摄像头画面",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (235, 235, 235),
            2,
        )

        filtered_rppg = self.get_filtered_rppg_for_display(rppg_values, fps)
        card_gap = 12
        card_w = panel_w - margin * 2
        card_h = 84
        top_y = 18

        bpm_text = "--" if bpm is None else f"{int(round(bpm))}"
        hrv_text = "--" if hrv is None else f"{hrv:.0f}"
        resp_text = "--" if resp_rate is None else f"{resp_rate:.1f}"
        dashboard = self.ui.draw_metric_card(
            dashboard, margin, top_y, card_w, card_h, "心率", bpm_text, "bpm", accent=(0, 210, 255)
        )
        dashboard = self.ui.draw_metric_card(
            dashboard,
            margin,
            top_y + card_h + card_gap,
            card_w,
            card_h,
            "心率变异性",
            hrv_text,
            "ms",
            accent=(255, 190, 0),
        )
        dashboard = self.ui.draw_metric_card(
            dashboard, margin, top_y + (card_h + card_gap) * 2, card_w, card_h + 12, "呼吸速率", resp_text, "次/分", accent=(0, 220, 140)
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
        self.ui.draw_text(
            dashboard,
            "按 q 退出",
            (margin + 12, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.ui.panel_muted,
            1,
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
            "current_hrv": None,
            "current_resp_rate": None,
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
                    latest_result.get("current_hrv"),
                    latest_result.get("current_resp_rate"),
                    latest_result.get("rppg_values", []),
                    latest_result.get("fps", self.target_fps),
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
