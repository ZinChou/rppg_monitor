import cv2
import numpy as np


class ProUI:
    def __init__(self):
        self.heart_phase = 0.0
        self.panel_bg_top = np.array([28, 44, 36], dtype=np.uint8)
        self.panel_bg_bottom = np.array([10, 18, 16], dtype=np.uint8)
        self.panel_border = (78, 108, 96)
        self.panel_text = (235, 240, 238)
        self.panel_muted = (150, 170, 164)
        self.panel_grid = (46, 64, 58)

    def fit_font_scale(self, text, font, max_width, base_scale, thickness):
        scale = base_scale
        while scale > 0.3:
            width = cv2.getTextSize(text, font, scale, thickness)[0][0]
            if width <= max_width:
                break
            scale -= 0.05
        return max(scale, 0.3)

    def draw_rounded_rect(self, img, x, y, w, h, color, radius=16, thickness=-1):
        overlay = img.copy()

        cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, thickness)
        cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, thickness)
        cv2.circle(overlay, (x + radius, y + radius), radius, color, thickness)
        cv2.circle(overlay, (x + w - radius, y + radius), radius, color, thickness)
        cv2.circle(overlay, (x + radius, y + h - radius), radius, color, thickness)
        cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, thickness)

        return overlay

    def draw_glass_panel(self, frame, x, y, w, h, alpha=0.35):
        overlay = self.draw_rounded_rect(frame, x, y, w, h, (25, 25, 25), radius=18, thickness=-1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        border = frame.copy()
        border = self.draw_rounded_rect(border, x, y, w, h, (90, 90, 90), radius=18, thickness=1)
        frame = cv2.addWeighted(border, 0.9, frame, 0.1, 0)
        return frame

    def draw_solid_panel(self, frame, x, y, w, h, fill_color=(24, 36, 34), border_color=None):
        if border_color is None:
            border_color = self.panel_border
        overlay = self.draw_rounded_rect(frame.copy(), x, y, w, h, fill_color, radius=18, thickness=-1)
        frame[:] = overlay
        border = self.draw_rounded_rect(frame.copy(), x, y, w, h, border_color, radius=18, thickness=1)
        frame[:] = border
        return frame

    def draw_panel_background(self, frame):
        frame[:] = (18, 24, 24)
        return frame

    def draw_metric_card(self, frame, x, y, w, h, label, value, unit="", accent=(0, 200, 255)):
        frame = self.draw_solid_panel(frame, x, y, w, h, fill_color=(24, 32, 32), border_color=(56, 76, 76))
        cv2.line(frame, (x + 14, y + 14), (x + 14, y + h - 14), accent, 3, cv2.LINE_AA)
        cv2.putText(
            frame, label, (x + 28, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.panel_muted, 1, cv2.LINE_AA
        )

        text = str(value)
        scale = self.fit_font_scale(text, cv2.FONT_HERSHEY_DUPLEX, w - 48, 1.2, 2)
        value_y = y + h - 22
        cv2.putText(
            frame, text, (x + 28, value_y),
            cv2.FONT_HERSHEY_DUPLEX, scale, self.panel_text, 2, cv2.LINE_AA
        )
        if unit:
            text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0][0]
            cv2.putText(
                frame, unit, (x + 36 + text_w, value_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.panel_muted, 1, cv2.LINE_AA
            )
        return frame

    def draw_quality_card(self, frame, x, y, w, h, quality):
        status, color = self.signal_status(quality)
        frame = self.draw_metric_card(frame, x, y, w, h, "Signal Quality", f"{quality:.3f}", accent=color)
        bar_x = x + 28
        bar_y = y + h - 18
        bar_w = w - 56
        bar_h = 10
        self.draw_signal_bar(frame, bar_x, bar_y, bar_w, bar_h, quality)
        cv2.putText(
            frame, status, (x + w - 74, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
        return frame

    def draw_simple_waveform(self, frame, values, x, y, w, h):
        frame = self.draw_solid_panel(frame, x, y, w, h, fill_color=(24, 32, 32), border_color=(56, 76, 76))
        cv2.putText(
            frame, "rPPG Signal", (x + 16, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.panel_text, 1, cv2.LINE_AA
        )

        graph_x = x + 12
        graph_y = y + 36
        graph_w = w - 24
        graph_h = h - 52

        for ratio in [0.25, 0.5, 0.75]:
            yy = graph_y + int(graph_h * ratio)
            cv2.line(frame, (graph_x, yy), (graph_x + graph_w, yy), self.panel_grid, 1, cv2.LINE_AA)

        if len(values) < 8:
            cv2.putText(
                frame, "waiting for signal...", (x + 16, y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.panel_muted, 1, cv2.LINE_AA
            )
            return frame

        vals = np.array(values[-graph_w:], dtype=np.float32)
        vals = vals - np.mean(vals)
        std = np.std(vals)
        if std > 1e-6:
            vals = vals / std
        vals = np.clip(vals, -2.5, 2.5)
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)

        pts = []
        for i, v in enumerate(vals):
            px = graph_x + i
            py = graph_y + int((1.0 - v) * graph_h)
            pts.append((px, py))

        mid_y = graph_y + graph_h // 2
        cv2.line(frame, (graph_x, mid_y), (graph_x + graph_w, mid_y), self.panel_border, 1, cv2.LINE_AA)

        if len(pts) > 1:
            fill_pts = np.array([(graph_x, graph_y + graph_h)] + pts + [(pts[-1][0], graph_y + graph_h)], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [fill_pts], (16, 72, 54), lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 235, 150), 2, cv2.LINE_AA)

        cv2.circle(frame, pts[-1], 4, (130, 255, 190), -1, cv2.LINE_AA)
        cv2.putText(
            frame, f"{values[-1]:+.2f}", (x + w - 80, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 230, 200), 1, cv2.LINE_AA
        )
        return frame

    def draw_dashboard_header(self, frame, x, y, w):
        frame = self.draw_solid_panel(frame, x, y, w, 62, fill_color=(24, 38, 36))
        cv2.putText(
            frame, "Vital Signs Monitor", (x + 16, y + 26),
            cv2.FONT_HERSHEY_DUPLEX, 0.78, self.panel_text, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Remote PPG Dashboard", (x + 16, y + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.panel_muted, 1, cv2.LINE_AA
        )
        cv2.line(frame, (x + w - 76, y + 18), (x + w - 22, y + 18), (90, 200, 150), 2, cv2.LINE_AA)
        cv2.line(frame, (x + w - 76, y + 30), (x + w - 36, y + 30), (0, 200, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x + w - 76, y + 42), (x + w - 52, y + 42), (255, 190, 0), 2, cv2.LINE_AA)
        return frame

    def draw_chip(self, frame, text, x, y, bg_color, text_color=(245, 245, 245), pad_x=10, pad_y=7):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thickness = 1
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        w = text_size[0] + pad_x * 2
        h = text_size[1] + pad_y * 2
        frame = self.draw_glass_panel(frame, x, y, w, h, alpha=0.22)
        overlay = self.draw_rounded_rect(frame.copy(), x, y, w, h, bg_color, radius=14, thickness=-1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        cv2.putText(
            frame, text, (x + pad_x, y + h - pad_y - 2),
            font, scale, text_color, thickness, cv2.LINE_AA
        )
        return frame

    def draw_stat_tile(self, frame, x, y, w, h, label, value, accent, value_color=(255, 255, 255)):
        frame = self.draw_solid_panel(frame, x, y, w, h, fill_color=(23, 35, 33))
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h // 2), (40, 55, 52), -1)
        frame = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0)
        cv2.line(frame, (x + 12, y + 12), (x + 12, y + h - 12), accent, 3, cv2.LINE_AA)
        cv2.putText(
            frame, label, (x + 24, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.panel_muted, 1, cv2.LINE_AA
        )
        scale = self.fit_font_scale(str(value), cv2.FONT_HERSHEY_DUPLEX, w - 36, 0.95, 2)
        cv2.putText(
            frame, str(value), (x + 24, y + h - 16),
            cv2.FONT_HERSHEY_DUPLEX, scale, value_color, 2, cv2.LINE_AA
        )
        cv2.circle(frame, (x + w - 18, y + 18), 4, accent, -1, cv2.LINE_AA)
        return frame

    def signal_status(self, quality):
        if quality >= 0.18:
            return "GOOD", (0, 220, 0)
        elif quality >= 0.09:
            return "OK", (0, 180, 255)
        else:
            return "POOR", (0, 0, 255)

    def draw_heart(self, frame, center, size, bpm):
        x, y = center

        if bpm is None:
            scale = 1.0
        else:
            self.heart_phase += max(0.03, bpm / 60.0 * 0.045)
            scale = 1.0 + 0.12 * np.sin(self.heart_phase * 2.0 * np.pi)

        s = max(8, int(size * scale))

        cv2.circle(frame, (x - s // 3, y - s // 6), s // 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x + s // 3, y - s // 6), s // 3, (0, 0, 255), -1, cv2.LINE_AA)

        pts = np.array([
            [x - s, y - s // 10],
            [x + s, y - s // 10],
            [x, y + s]
        ], dtype=np.int32)
        cv2.fillConvexPoly(frame, pts, (0, 0, 255), lineType=cv2.LINE_AA)

    def draw_signal_bar(self, frame, x, y, w, h, quality):
        value = min(1.0, quality / 0.25)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
        color = (0, 220, 0) if value > 0.65 else (0, 180, 255) if value > 0.35 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + int(w * value), y + h), color, -1)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (110, 110, 110), 1)

    def draw_waveform(self, frame, values, x, y, w, h, title="rPPG"):
        frame = self.draw_solid_panel(frame, x, y, w, h, fill_color=(22, 34, 32))

        cv2.putText(
            frame, title, (x + 14, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.panel_text, 1, cv2.LINE_AA
        )

        if len(values) < 8:
            cv2.putText(
                frame, "waiting for signal...", (x + 14, y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.panel_muted, 1, cv2.LINE_AA
            )
            return frame

        graph_x = x + 10
        graph_y = y + 35
        graph_w = w - 20
        graph_h = h - 45

        for ratio in [0.25, 0.5, 0.75]:
            yy = graph_y + int(graph_h * ratio)
            cv2.line(frame, (graph_x, yy), (graph_x + graph_w, yy), self.panel_grid, 1, cv2.LINE_AA)

        vals = np.array(values[-graph_w:], dtype=np.float32)
        vals = vals - np.mean(vals)
        std = np.std(vals)
        if std > 1e-6:
            vals = vals / std

        vals = np.clip(vals, -2.5, 2.5)
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)

        pts = []
        for i, v in enumerate(vals):
            px = graph_x + i
            py = graph_y + int((1.0 - v) * graph_h)
            pts.append((px, py))

        mid_y = graph_y + graph_h // 2
        cv2.line(frame, (graph_x, mid_y), (graph_x + graph_w, mid_y), self.panel_border, 1, cv2.LINE_AA)

        if len(pts) > 1:
            fill_pts = np.array([(graph_x, graph_y + graph_h)] + pts + [(pts[-1][0], graph_y + graph_h)], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [fill_pts], (0, 120, 70), lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0)

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 255, 120), 2, cv2.LINE_AA)

        latest = values[-1]
        cv2.circle(frame, pts[-1], 4, (120, 255, 180), -1, cv2.LINE_AA)
        cv2.putText(
            frame, f"latest {latest:+.2f}", (x + w - 132, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 220, 180), 1, cv2.LINE_AA
        )

        return frame

    def draw_bpm_chart(self, frame, values, x, y, w, h, bpm_low, bpm_high):
        frame = self.draw_solid_panel(frame, x, y, w, h, fill_color=(22, 34, 32))

        cv2.putText(
            frame, "BPM Trend", (x + 14, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.panel_text, 1, cv2.LINE_AA
        )

        if len(values) < 2:
            cv2.putText(
                frame, "collecting bpm...", (x + 14, y + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.panel_muted, 1, cv2.LINE_AA
            )
            return frame

        graph_x = x + 10
        graph_y = y + 35
        graph_w = w - 20
        graph_h = h - 45

        vals = np.array(values[-60:], dtype=np.float32)
        vals = np.clip(vals, bpm_low, bpm_high)
        vals = (vals - bpm_low) / (bpm_high - bpm_low + 1e-6)

        pts = []
        n = len(vals)
        for i, v in enumerate(vals):
            px = graph_x + int(i * (graph_w - 1) / max(1, n - 1))
            py = graph_y + int((1.0 - v) * graph_h)
            pts.append((px, py))

        for bpm_mark in [60, 90, 120, 150]:
            if bpm_low <= bpm_mark <= bpm_high:
                yy = graph_y + int((1.0 - (bpm_mark - bpm_low) / (bpm_high - bpm_low)) * graph_h)
                cv2.line(frame, (graph_x, yy), (graph_x + graph_w, yy), self.panel_grid, 1, cv2.LINE_AA)
                cv2.putText(
                    frame, str(bpm_mark), (graph_x + 4, yy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.panel_muted, 1, cv2.LINE_AA
                )

        if len(pts) > 1:
            fill_pts = np.array([(graph_x, graph_y + graph_h)] + pts + [(pts[-1][0], graph_y + graph_h)], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [fill_pts], (0, 110, 140), lineType=cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0)

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 200, 255), 2, cv2.LINE_AA)

        cv2.circle(frame, pts[-1], 4, (130, 230, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame, f"now {values[-1]:.0f} bpm", (x + w - 118, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 220, 235), 1, cv2.LINE_AA
        )

        return frame

    def draw_main_card(self, frame, bpm, quality, fps, face_ok, x=14, y=14, w=290, h=170):
        frame = self.draw_solid_panel(frame, x, y, w, h, fill_color=(23, 35, 33))
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + 70), (36, 52, 48), -1)
        frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)

        self.draw_heart(frame, (x + 38, y + 50), 18, bpm)

        if bpm is not None:
            cv2.putText(
                frame, f"{int(round(bpm))}",
                (x + 74, y + 64), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, "BPM",
                (x + 78, y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.panel_muted, 1, cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame, "--",
                (x + 74, y + 64), cv2.FONT_HERSHEY_DUPLEX, 1.6, (120, 120, 120), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, "estimating",
                (x + 78, y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.panel_muted, 1, cv2.LINE_AA
            )

        status, color = self.signal_status(quality)
        cv2.putText(
            frame, f"Signal: {status}",
            (x + 6, y + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA
        )
        self.draw_signal_bar(frame, x + 6, y + 132, min(180, w - 100), 12, quality)

        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (x + w - 89, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA
        )

        state_text = "TRACKING" if face_ok else "FACE LOST"
        state_color = (0, 220, 0) if face_ok else (0, 0, 255)
        cv2.putText(
            frame, state_text,
            (x + w - 96, y + h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_color, 2, cv2.LINE_AA
        )

        return frame

    def draw_top_status(self, frame, face_ok, quality):
        h, w = frame.shape[:2]

        if not face_ok:
            text = "Face Lost"
            color = (0, 0, 255)
        elif quality < 0.06:
            text = "Hold Still"
            color = (0, 180, 255)
        else:
            text = "Stable"
            color = (0, 220, 0)

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        x = (w - text_size[0]) // 2
        y = 36
        frame = self.draw_chip(frame, text.upper(), max(12, x - 14), 10, color)
        return frame