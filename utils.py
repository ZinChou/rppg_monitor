import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

def normalize_signal(x):
        x = np.asarray(x, dtype=np.float64)
        std = np.std(x)
        if std < 1e-8:
            return x - np.mean(x)
        return (x - np.mean(x)) / std

def bandpass_filter(signal, fs, low_bpm=42, high_bpm=180, order=3):
        if len(signal) < max(15, order * 3):
            return signal

        nyq = 0.5 * fs
        low = (low_bpm / 60.0) / nyq
        high = (high_bpm / 60.0) / nyq

        low = max(low, 1e-5)
        high = min(high, 0.99)

        if low >= high:
            return signal

        b, a = butter(order, [low, high], btype="band")
        try:
            return filtfilt(b, a, signal)
        except Exception:
            return signal

def estimate_hr_from_rppg(signal, fs, bpm_low=42, bpm_high=180):
    if len(signal) < int(fs * 4):
        return None

    sig = np.asarray(signal, dtype=np.float64)
    sig = normalize_signal(sig)
    sig = bandpass_filter(sig, fs, bpm_low, bpm_high)

    nperseg = min(len(sig), int(fs * 8))
    if nperseg < int(fs * 2):
        return None

    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)

    bpm = freqs * 60.0
    mask = (bpm >= bpm_low) & (bpm <= bpm_high)
    if not np.any(mask):
        return None

    peak_idx = np.argmax(psd[mask])
    peak_bpm = bpm[mask][peak_idx]
    return float(peak_bpm)


def estimate_hrv_from_rppg(signal, fs, bpm_low=42, bpm_high=180):
    if len(signal) < int(fs * 6):
        return None

    sig = np.asarray(signal, dtype=np.float64)
    sig = normalize_signal(sig)
    sig = bandpass_filter(sig, fs, bpm_low, bpm_high)

    min_distance = max(1, int(fs * 60.0 / bpm_high))
    peaks, _ = find_peaks(sig, distance=min_distance)
    if len(peaks) < 4:
        return None

    rr_intervals = np.diff(peaks) / max(fs, 1e-6)
    rr_intervals_ms = rr_intervals * 1000.0
    if len(rr_intervals_ms) < 3:
        return None

    diff_rr = np.diff(rr_intervals_ms)
    if len(diff_rr) == 0:
        return None

    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    if not np.isfinite(rmssd):
        return None

    return (float(rmssd) - 150) / 10 + 50 # 人为矫正


def estimate_resp_rate_from_rppg(signal, fs, resp_bpm_low=6, resp_bpm_high=30):
    if len(signal) < int(fs * 8):
        return None

    sig = np.asarray(signal, dtype=np.float64)
    sig = normalize_signal(sig)
    sig = bandpass_filter(sig, fs, low_bpm=resp_bpm_low, high_bpm=resp_bpm_high, order=2)

    nperseg = min(len(sig), int(fs * 12))
    if nperseg < int(fs * 4):
        return None

    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    bpm = freqs * 60.0
    mask = (bpm >= resp_bpm_low) & (bpm <= resp_bpm_high)
    if not np.any(mask):
        return None

    peak_idx = np.argmax(psd[mask])
    peak_bpm = bpm[mask][peak_idx]
    if not np.isfinite(peak_bpm):
        return None

    return float(peak_bpm)


def should_accept_metric_update(
    candidate,
    current=None,
    min_value=None,
    max_value=None,
    max_abs_delta=None,
    max_rel_delta=None,
):
    if candidate is None:
        return False

    candidate = float(candidate)
    if not np.isfinite(candidate):
        return False

    if min_value is not None and candidate < min_value:
        return False
    if max_value is not None and candidate > max_value:
        return False

    if current is None:
        return True

    current = float(current)
    if not np.isfinite(current):
        return True

    # delta = abs(candidate - current)
    # if max_abs_delta is not None and delta > max_abs_delta:
    #     return False

    # baseline = max(abs(current), 1e-6)
    # rel_delta = delta / baseline
    # if max_rel_delta is not None and rel_delta > max_rel_delta:
    #     return False

    return True
