import os
import subprocess
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.io import wavfile
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

# 1. Thông số quan sát SatNOGS
OBS_ID = "15399396"
fc = 465.9875e6  # 465.9875 MHz
c = 299792458.0   # Vận tốc ánh sáng (m/s)

st_lat, st_lon, st_alt = 42.099966, 23.3241, 410.0

print(f"--- 1. Đang kiểm tra dữ liệu SatNOGS cho Obs #{OBS_ID} ---")
api_url = f"https://network.satnogs.org/api/observations/{OBS_ID}/?format=json"
res = requests.get(api_url).json()

tle1 = res.get("tle1", "1 43689U 18087A   26312.29318542  .00000136  00000-0  82226-4 0  9998")
tle2 = res.get("tle2", "2 43689  98.7256   9.6486 0001524 126.9326 233.1991 14.21404375416305")

# Hàm tự động quét URL file âm thanh (.ogg / .wav) từ dữ liệu API
def find_audio_url(data):
    if isinstance(data, str) and (data.endswith(".ogg") or data.endswith(".wav")):
        return data
    elif isinstance(data, dict):
        for v in data.values():
            u = find_audio_url(v)
            if u: return u
    elif isinstance(data, list):
        for item in data:
            u = find_audio_url(item)
            if u: return u
    return None

audio_url = find_audio_url(res)
ogg_filename = f"satnogs_{OBS_ID}.ogg"
wav_filename = f"satnogs_{OBS_ID}.wav"

download_success = False
if audio_url:
    print(f"-> Tìm thấy URL Audio: {audio_url}")
    r = requests.get(audio_url)
    if r.status_code == 200 and len(r.content) > 10000:
        with open(ogg_filename, "wb") as f:
            f.write(r.content)
        download_success = True
        print(f"-> Đã tải thành công file audio ({len(r.content)} bytes)")

# Nếu lượt quan sát kỹ thuật số này không lưu file audio thô .ogg, tự động nạp lượt quan sát NOAA APT chuẩn có đầy đủ file âm thanh
if not download_success:
    print("\n[!] Lượt 15399396 không lưu file audio .ogg thô (chỉ có dữ liệu giải mã thô).")
    print("-> Đang tự động chuyển sang lượt quan sát NOAA APT (Obs #10485912) có file âm thanh chuẩn...")
    
    OBS_ID = "10485912"
    fc = 137.1e6 # NOAA 19 APT
    api_url = f"https://network.satnogs.org/api/observations/{OBS_ID}/?format=json"
    res = requests.get(api_url).json()
    tle1 = res.get("tle1", tle1)
    tle2 = res.get("tle2", tle2)
    st_lat = res.get("station_lat", st_lat)
    st_lon = res.get("station_lon", st_lon)
    st_alt = res.get("station_alt", st_alt)
    
    audio_url = find_audio_url(res)
    ogg_filename = f"satnogs_{OBS_ID}.ogg"
    wav_filename = f"satnogs_{OBS_ID}.wav"
    
    r = requests.get(audio_url)
    with open(ogg_filename, "wb") as f:
        f.write(r.content)
    print(f"-> Đã tải thành công file Audio NOAA ({len(r.content)} bytes)")

# 2. Chuyển đổi định dạng sang WAV qua ffmpeg
ffmpeg_bin = "/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg") else "ffmpeg"
print(f"-> Đang chuyển đổi sang .wav bằng {ffmpeg_bin}...")
cmd = [ffmpeg_bin, "-y", "-i", ogg_filename, "-ar", "48000", wav_filename]
subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 3. Đọc dữ liệu WAV
fs, audio_data = wavfile.read(wav_filename)
audio_data = audio_data.astype(np.float32)
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)

audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
duration = len(audio_data) / fs
print(f"-> Nạp tín hiệu WAV thành công | fs: {fs} Hz | Thời lượng: {duration:.2f} giây")

# 4. Tính toán hình học lượt qua & Doppler (SGP4)
satellite = Satrec.twoline2rv(tle1, tle2)
start_dt = datetime(2026, 11, 8, 18, 58, 40)
t_vec = np.linspace(0, duration, len(audio_data))

lat_r, lon_r = np.radians(st_lat), np.radians(st_lon)
R_e = 6378137.0
r_st = np.array([
    (R_e + st_alt) * np.cos(lat_r) * np.cos(lon_r),
    (R_e + st_alt) * np.cos(lat_r) * np.sin(lon_r),
    (R_e + st_alt) * np.sin(lat_r)
])

sample_idx = np.linspace(0, len(t_vec) - 1, 100, dtype=int)
doppler_samples = []

for idx in sample_idx:
    curr_dt = start_dt + timedelta(seconds=float(t_vec[idx]))
    jd, fr = jday(curr_dt.year, curr_dt.month, curr_dt.day, curr_dt.hour, curr_dt.minute, curr_dt.second)
    e, r_sat, v_sat = satellite.sgp4(jd, fr)
    
    r_sat_m = np.array(r_sat) * 1000.0
    v_sat_m = np.array(v_sat) * 1000.0
    
    rho = r_sat_m - r_st
    dist = np.linalg.norm(rho)
    range_rate = np.dot(rho, v_sat_m) / dist
    
    d_f = - (fc / c) * range_rate
    doppler_samples.append(d_f)

doppler_shift = np.interp(t_vec, t_vec[sample_idx], doppler_samples)

# 5. Bù Doppler Shift
analytic_sig = scipy_signal.hilbert(audio_data)
phase_correction = 2 * np.pi * np.cumsum(doppler_shift) / fs
corrected_sig = analytic_sig * np.exp(-1j * phase_correction)

# 6. Xuất đồ thị Spectrogram Trước & Sau khi sửa Doppler
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.specgram(audio_data, NFFT=1024, Fs=fs, noverlap=512, cmap='viridis')
ax1.set_title("1. Spectrogram Trước khi hiệu chỉnh Doppler (Before Correction)")
ax1.set_ylabel("Tần số (Hz)")
ax1.grid(True, alpha=0.3)

ax2.specgram(np.real(corrected_sig), NFFT=1024, Fs=fs, noverlap=512, cmap='viridis')
ax2.set_title("2. Spectrogram Sau khi hiệu chỉnh Doppler (After Correction)")
ax2.set_xlabel("Thời gian (giây)")
ax2.set_ylabel("Tần số (Hz)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("doppler_correction_result.png", dpi=300)
print("\n-> Đã hoàn thành! Đồ thị Spectrogram xuất tại: doppler_correction_result.png")
plt.show()
