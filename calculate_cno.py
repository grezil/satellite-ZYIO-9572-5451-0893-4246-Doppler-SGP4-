import glob
import numpy as np
from scipy.io import wavfile

# Tự động tìm file .wav đã xuất ở các bước trước
wav_files = glob.glob("satnogs_*.wav")
if not wav_files:
    raise FileNotFoundError("Không tìm thấy file satnogs_*.wav nào trong thư mục!")

wav_filename = wav_files[0]
print(f"--- Đang phân tích chỉ số C/N0 từ: {wav_filename} ---")

fs, data = wavfile.read(wav_filename)
data = data.astype(np.float32)

# Tính công suất tổng và công suất nhiễu nền
fft_data = np.abs(np.fft.rfft(data))**2
freqs = np.fft.rfftfreq(len(data), 1/fs)

# Tín hiệu APT nằm quanh 2.4 kHz (+- 1 kHz)
sig_mask = (freqs >= 1400) & (freqs <= 3400)
noise_mask = (freqs > 4000) & (freqs <= 10000)

p_signal = np.sum(fft_data[sig_mask])
p_noise = np.median(fft_data[noise_mask]) * np.sum(sig_mask)

snr_linear = max(1.0, (p_signal - p_noise) / p_noise)
snr_db = 10 * np.log10(snr_linear)

# Bandwidth B = 3400 - 1400 = 2000 Hz
cno_measured = snr_db + 10 * np.log10(2000)

print(f"\n=== BÁO CÁO CHỈ SỐ C/N_0 WORK PACKAGE 5 ===")
print(f"Measured SNR: {snr_db:.2f} dB")
print(f"Measured C/N0: {cno_measured:.2f} dB-Hz")
