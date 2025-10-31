import requests, cv2, time

API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
API_KEY = "d2bde6d3d60a41ff8c53799dd603e285"

params = {
    "apiKey": API_KEY,
    "type": "its",
    "cctvType": "1",
    "minX": 127.20,
    "maxX": 127.27,
    "minY": 36.49,
    "maxY": 36.52,
    "getType": "json"
}

def get_latest_cctv_url():
    res = requests.get(API_URL, params=params, timeout=5)
    data = res.json()
    for item in data.get("response", {}).get("data", []):
        if "봉안" in item["cctvname"]:
            print("📡 CCTV:", item["cctvname"])
            return item["cctvurl"]
    return None

def play_cctv_stream():
    url = get_latest_cctv_url()
    cap = cv2.VideoCapture(url)
    last_refresh = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 끊김 → 재연결")
            cap.release()
            url = get_latest_cctv_url()
            cap = cv2.VideoCapture(url)
            last_refresh = time.time()
            continue

        # 프레임 크기 조절
        frame = cv2.resize(frame, (640, 360))

        # 프레임 표시
        cv2.imshow("Sejong CCTV (Stable)", frame)

        # 🔹 sleep 추가 → 0.05초(=20fps) 또는 0.1초(=10fps)
        time.sleep(0.05)

        # 🔹 10분마다 URL 새로고침
        if time.time() - last_refresh > 600:
            print("♻️ 10분 경과 → URL 갱신")
            cap.release()
            url = get_latest_cctv_url()
            cap = cv2.VideoCapture(url)
            last_refresh = time.time()

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_cctv_stream()
