import requests, json

API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
params = {
    "apiKey": "d2bde6d3d60a41ff8c53799dd603e285",
    "type": "its",          # 국도 / 고속도로(ex)
    "cctvType": "1",         # HLS
     "minX": 117.20,
    "maxX": 147.35,
    "minY": 26.45,
    "maxY": 56.60,
    "getType": "json"
}

res = requests.get(API_URL, params=params)
print("응답 상태코드:", res.status_code)
print("원문 응답:", res.text[:500], "\n")  # 응답 일부 확인

try:
    data = res.json()
except Exception as e:
    print("❌ JSON 변환 실패:", e)
    exit()

# 응답 구조 확인
if "response" not in data:
    print("❌ response 키 없음 → API 오류일 가능성 높음")
    exit()

response = data["response"]

if "data" not in response:
    print("❌ data 항목 없음")
    print("📩 메시지:", response.get("resultMsg", "데이터 없음"))
    exit()

# 데이터가 정상일 때
for item in response["data"]:
    print(f"📹 {item['cctvname']} ({item['coordy']}, {item['coordx']})")
    print("URL:", item["cctvurl"], "\n")
