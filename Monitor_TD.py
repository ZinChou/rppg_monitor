import multiprocessing as mp
import queue
import time
from collections import deque

import cv2
import numpy as np

from display import ProUI
from model.POS import POS
from utils import (
    bandpass_filter,
    estimate_hr_from_rppg,
    estimate_hrv_from_rppg,
    estimate_resp_rate_from_rppg,
    normalize_signal,
    should_accept_metric_update,
)


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
    pos_window_seconds,
    target_fps,
    bpm_low,
    bpm_high,
):
    model = POS(
        pos_window_seconds=pos_window_seconds,
        bpm_low=bpm_low,
        bpm_high=bpm_high,
    )
    model.fs = target_fps

    timestamps = deque()
    rgb_timestamps = deque()
    rgb_buffer = deque()
    rppg_buffer = deque()
    bpm_buffer = deque()
    bpm_timestamps = deque()
    frame_times = deque(maxlen=120)

    current_bpm = None
    current_hrv = None
    current_resp_rate = None
    last_bpm_update = 0.0

    def trim_buffers(current_t):
        min_t = current_t - buffer_seconds
        while timestamps and timestamps[0] < min_t:
            timestamps.popleft()
            if rppg_buffer:
                rppg_buffer.popleft()

        while rgb_timestamps and rgb_timestamps[0] < min_t:
            rgb_timestamps.popleft()
            if rgb_buffer:
                rgb_buffer.popleft()

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

        face_box = model.detect_face(frame)
        latest_rppg = None
        if face_box is not None:
            mask = model.build_roi_mask_from_face_box(frame.shape, face_box)
            mean_rgb = model.extract_mean_rgb(frame, mask)
            
            if mean_rgb is not None:
                timestamps.append(timestamp)
                rgb_timestamps.append(timestamp)
                rgb_buffer.append(mean_rgb)

                win_size = max(10, int(pos_window_seconds * compute_fps()))
                if len(rgb_buffer) >= win_size:
                    rgb_window = np.array(list(rgb_buffer)[-win_size:], dtype=np.float64)
                    h = model.compute_pos_signal(rgb_window)
                    if h is not None and len(h) > 0:
                        latest_rppg = float(h[-1])
                        rppg_buffer.append(latest_rppg)
                    else:
                        rppg_buffer.append(0.0)
                else:
                    rppg_buffer.append(0.0)
            else:
                timestamps.append(timestamp)
                rppg_buffer.append(0.0)
        else:
            timestamps.append(timestamp)
            rppg_buffer.append(0.0)

        trim_buffers(timestamp)

        fs = compute_fps()
        if timestamp - last_bpm_update > 1.0 and len(rppg_buffer) > max(30, int(fs * 4)):
            bpm = estimate_hr_from_rppg(list(rppg_buffer), fs, bpm_low=bpm_low, bpm_high=bpm_high)
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
                bpm_timestamps.append(timestamp)
                bpm_buffer.append(current_bpm)

                hrv = estimate_hrv_from_rppg(list(rppg_buffer), fs, bpm_low=bpm_low, bpm_high=bpm_high)
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

                resp_rate = estimate_resp_rate_from_rppg(list(rppg_buffer), fs)
                if should_accept_metric_update(
                    resp_rate,
                    current=current_resp_rate,
                    min_value=12.0,
                    max_value=30.0,
                    max_abs_delta=6.0,
                    max_rel_delta=0.35,
                ):
                    if current_resp_rate is None:
                        current_resp_rate = resp_rate
                    else:
                        current_resp_rate = 0.8 * current_resp_rate + 0.2 * resp_rate
            last_bpm_update = timestamp
        # print(len(list(rgb_buffer)))
        _put_latest(
            result_queue,
            {
                "timestamp": timestamp,
                "face_box": face_box,
                "rppg_values": list(rppg_buffer),
                "bpm_values": list(bpm_buffer),
                "current_bpm": current_bpm,
                "current_hrv": current_hrv,
                "current_resp_rate": current_resp_rate,
                "quality": compute_quality(),
                "fps": fs,
            },
        )


class Monitor_TD:
    def __init__(
        self,
        model=None,
        camera_id=0,
        buffer_seconds=10,
        pos_window_seconds=1.6,
        display_scale=1.0,
        target_fps=30.0,
    ):
        self.model = model
        self.camera_id = camera_id
        self.buffer_seconds = buffer_seconds
        self.pos_window_seconds = pos_window_seconds
        self.display_scale = display_scale
        self.target_fps = float(target_fps)

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
            dashboard, margin, top_y + card_h + card_gap, card_w, card_h, "心率变异性", hrv_text, "ms", accent=(255, 190, 0)
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
                self.pos_window_seconds,
                self.target_fps,
                self.bpm_low,
                self.bpm_high,
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
                cv2.imshow("rPPG Monitor", display)

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
