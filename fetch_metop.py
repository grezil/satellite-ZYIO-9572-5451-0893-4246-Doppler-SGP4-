import requests
import soundfile as sf

OBS_ID = "15399396"

print(f"--- 1. Đang kết nối SatNOGS API cho Observation #{OBS_ID} ---")
url = f"https://network.satnogs.org/api/observations/{OBS_ID}/?format=json"
res = requests.get(url).json()

tle1 = res.get("tle1")
tle2 = res.get("tle2")
station_lat = res.get("station_lat")
station_lon = res.get("station_lon")
station_alt = res.get("station_alt")

print(f"Tọa độ trạm thu: Lat {station_lat}°, Lon {station_lon}°, Alt {station_alt}m")
print(f"TLE Line 1: {tle1}")
print(f"TLE Line 2: {tle2}")

# Hàm trích xuất chuỗi URL từ Dictionary/List của SatNOGS API
def extract_url(data):
    if isinstance(data, str) and data.startswith("http"):
        return data
    elif isinstance(data, dict):
        for val in data.values():
            res_url = extract_url(val)
            if res_url: return res_url
    elif isinstance(data, list) and len(data) > 0:
        return extract_url(data[0])
    return None

raw_payload = res.get("demoddata") or res.get("payload") or res.get("downloads")
download_url = extract_url(raw_payload)

if download_url:
    print(f"\n--- 2. Đang tải tệp dữ liệu từ: {download_url} ---")
    file_res = requests.get(download_url)
    
    ext = ".ogg" if ".ogg" in download_url else (".wav" if ".wav" in download_url else ".dat")
    filename = f"satnogs_{OBS_ID}{ext}"
    
    with open(filename, "wb") as f:
        f.write(file_res.content)
    print(f"-> Đã lưu thành công tệp: {filename}")
    
    try:
        signal, sample_rate = sf.read(filename)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        print(f"-> Tốc độ lấy mẫu (fs): {sample_rate} Hz")
        print(f"-> Tổng số mẫu: {len(signal)} ({len(signal)/sample_rate:.2f} giây)")
    except Exception:
        print(f"-> Đã lưu dạng dữ liệu thô (Demodulated Binary): {filename} ({len(file_res.content)} bytes)")
else:
    print("Không tìm thấy đường dẫn tải tệp!")
