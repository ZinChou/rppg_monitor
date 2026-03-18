import numpy as np
from scipy.signal import butter, filtfilt, welch

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