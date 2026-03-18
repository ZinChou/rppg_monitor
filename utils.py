import numpy as np
from scipy.signal import butter, filtfilt, welch

def estimate_hr_from_rppg(self, signal, fs):
    if len(signal) < int(fs * 4):
        return None

    sig = np.asarray(signal, dtype=np.float64)
    sig = self.normalize_signal(sig)
    sig = self.bandpass_filter(sig, fs, self.bpm_low, self.bpm_high)

    nperseg = min(len(sig), int(fs * 8))
    if nperseg < int(fs * 2):
        return None

    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)

    bpm = freqs * 60.0
    mask = (bpm >= self.bpm_low) & (bpm <= self.bpm_high)
    if not np.any(mask):
        return None

    peak_idx = np.argmax(psd[mask])
    peak_bpm = bpm[mask][peak_idx]
    return float(peak_bpm)