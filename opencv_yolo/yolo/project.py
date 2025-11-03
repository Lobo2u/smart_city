from ultralytics import YOLO
import cv2
import numpy as np
import requests
import time

# CCTV API 설정
API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
API_KEY = {{SECRET_KEY}}

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
    """CCTV URL 가져오기"""
    try:
        res = requests.get(API_URL, params=params, timeout=5)
        data = res.json()
        for item in data.get("response", {}).get("data", []):
            if "봉안" in item["cctvname"]:
                print("📡 CCTV:", item["cctvname"])
                return item["cctvurl"]
    except Exception as e:
        print(f"⚠️ URL 가져오기 실패: {e}")
    return None


# YOLO 모델 로드 (트래킹 기능 포함)
model = YOLO("yolov8n.pt")

histories = {}  # track_id별 이동 경로 저장

# ===== CCTV 스트림으로 변경 =====
print("🔄 CCTV URL 가져오는 중...")
cctv_url = get_latest_cctv_url()

if cctv_url is None:
    print("❌ CCTV URL을 가져올 수 없습니다. 종료합니다.")
    exit()

cap = cv2.VideoCapture(cctv_url)

# 스트림 설정 최적화
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps > 60:  # 스트림의 경우 FPS가 정확하지 않을 수 있음
    fps = 25  # 기본값 설정

print(f"✅ CCTV 연결 성공! FPS: {fps}")

# 여러 ROI 영역 설정
ROI_list = []  # ROI 폴리곤 좌표
ROI_directions = []  # 각 ROI의 허용 방향
current_ROI_points = []
current_direction = None  # 현재 선택 중인 ROI의 방향
roi_selected = False
roi_colors = [
    (0, 255, 255),  # 노란색
    (255, 0, 255),  # 자홍색
    (0, 255, 0),  # 녹색
    (255, 128, 0),  # 주황색
    (128, 0, 255),  # 보라색
    (0, 128, 255),  # 하늘색
]

# 방향 정의
DIRECTIONS = {
    '1': '↑ Up',
    '2': '↓ Down',
    '3': '← Left',
    '4': '→ Right'
}


def select_roi_callback(event, x, y, flags, param):
    """마우스 클릭으로 ROI 포인트 선택"""
    global current_ROI_points

    if event == cv2.EVENT_LBUTTONDOWN:
        current_ROI_points.append((x, y))
        print(f"포인트 추가: ({x}, {y}) - 현재 ROI의 포인트: {len(current_ROI_points)}개")

        # 임시 프레임에 점과 선 그리기
        temp_frame = param['frame'].copy()

        # 이미 완성된 ROI들 그리기
        for idx, roi_points in enumerate(ROI_list):
            color = roi_colors[idx % len(roi_colors)]
            cv2.polylines(temp_frame, [np.array(roi_points)], True, color, 2)
            for i, pt in enumerate(roi_points):
                cv2.circle(temp_frame, pt, 5, color, -1)
            # ROI 번호와 방향 표시
            if roi_points:
                centroid = np.mean(roi_points, axis=0).astype(int)
                direction_text = ROI_directions[idx] if idx < len(ROI_directions) else "?"
                cv2.putText(temp_frame, f"ROI {idx + 1}: {direction_text}",
                            (centroid[0] - 50, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 현재 그리고 있는 ROI 그리기
        color = roi_colors[len(ROI_list) % len(roi_colors)]
        for i, pt in enumerate(current_ROI_points):
            cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(temp_frame, str(i + 1), (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if i > 0:
                cv2.line(temp_frame, current_ROI_points[i - 1], pt, (0, 255, 0), 2)

        # 3개 이상의 점이 있으면 첫 점과 마지막 점을 연결하여 폴리곤 표시
        if len(current_ROI_points) >= 3:
            cv2.line(temp_frame, current_ROI_points[-1], current_ROI_points[0], (0, 255, 0), 2)
            cv2.polylines(temp_frame, [np.array(current_ROI_points)], True, color, 2)

            # 현재 선택된 방향 표시
            if current_direction:
                centroid = np.mean(current_ROI_points, axis=0).astype(int)
                cv2.putText(temp_frame, f"Direction: {current_direction}",
                            (centroid[0] - 50, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 범례 표시
        draw_legend(temp_frame)

        param['temp_frame'][:] = temp_frame


def draw_legend(frame):
    """화면에 조작 방법 범례 표시"""
    overlay = frame.copy()
    legend_x, legend_y = 10, 10
    legend_width, legend_height = 460, 280

    cv2.rectangle(overlay, (legend_x, legend_y),
                  (legend_x + legend_width, legend_y + legend_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.rectangle(frame, (legend_x, legend_y),
                  (legend_x + legend_width, legend_y + legend_height),
                  (255, 255, 255), 2)

    title_y = legend_y + 30
    cv2.putText(frame, "=== ROI Selection Guide ===",
                (legend_x + 20, title_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    instructions = [
        ("Left Click", "Add point", (0, 255, 0)),
        ("'r' key", "Remove last point", (0, 200, 255)),
        ("'c' key", "Clear current ROI", (0, 150, 255)),
        ("'1' key", "Set direction: Up", (255, 200, 0)),
        ("'2' key", "Set direction: Down", (255, 200, 0)),
        ("'3' key", "Set direction: Left", (255, 200, 0)),
        ("'4' key", "Set direction: Right", (255, 200, 0)),
        ("'n' key", "Finish & start new ROI", (255, 128, 0)),
        ("Enter", "Complete all ROIs", (0, 255, 255)),
        ("'s' or ESC", "Skip (use full screen)", (150, 150, 150)),
    ]

    start_y = title_y + 30
    line_height = 23

    for i, (action, description, color) in enumerate(instructions):
        y_pos = start_y + i * line_height
        cv2.putText(frame, action,
                    (legend_x + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, f": {description}",
                    (legend_x + 130, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    status_y = start_y + len(instructions) * line_height + 10
    status_text = f"ROIs: {len(ROI_list)} | Points: {len(current_ROI_points)}"
    cv2.putText(frame, status_text,
                (legend_x + 15, status_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 현재 선택된 방향 표시
    if current_direction:
        cv2.putText(frame, f"Selected: {current_direction}",
                    (legend_x + 15, status_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


def is_in_roi(bbox, roi_points, overlap_threshold=0.5):
    """바운딩 박스가 ROI 영역과 일정 비율 이상 겹치는지 확인"""
    if roi_points is None or len(roi_points) < 3:
        return True

    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1

    bbox_corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.int32)

    roi_contour = np.array(roi_points, dtype=np.int32)

    min_x = max(0, x1 - 10)
    min_y = max(0, y1 - 10)
    max_x = x2 + 10
    max_y = y2 + 10

    mask_width = int(max_x - min_x)
    mask_height = int(max_y - min_y)

    if mask_width <= 0 or mask_height <= 0:
        return False

    bbox_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    shifted_bbox = bbox_corners - np.array([min_x, min_y])
    cv2.fillPoly(bbox_mask, [shifted_bbox], 255)

    roi_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    shifted_roi = roi_contour - np.array([min_x, min_y])
    cv2.fillPoly(roi_mask, [shifted_roi], 255)

    intersection = cv2.bitwise_and(bbox_mask, roi_mask)
    overlap_area = np.sum(intersection > 0)
    bbox_area = w * h

    if bbox_area == 0:
        return False

    overlap_ratio = overlap_area / bbox_area

    return overlap_ratio >= overlap_threshold


def get_roi_index_for_bbox(bbox, roi_list):
    """바운딩 박스가 속한 ROI 인덱스 반환"""
    for idx, roi_points in enumerate(roi_list):
        if is_in_roi(bbox, roi_points, overlap_threshold=0.5):
            return idx
    return None


def analyze_speed(history, fps):
    """이동 경로 기반 속도 분석"""
    if len(history) < 5:
        return "insufficient"

    pts = np.array(history[-5:])
    dx, dy = np.diff(pts[:, 0]), np.diff(pts[:, 1])
    v = np.mean(np.sqrt(dx ** 2 + dy ** 2)) * fps / 100

    if v < 0.7:
        return "stopped"
    elif v < 1.8:
        return "slow"
    else:
        return "fast"


def analyze_direction(history, min_distance=30):
    """이동 경로 기반 방향 분석 (4방향) - 개선된 버전"""
    if len(history) < 5:
        return "unknown", 0

    # 최근 포인트 수를 늘려서 더 안정적인 방향 계산
    use_points = min(len(history), 20)  # 최대 20개 포인트 사용
    pts = np.array(history[-use_points:])

    # 시작점과 끝점의 차이 계산
    start_point = pts[0]
    end_point = pts[-1]

    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # 이동 거리가 너무 작으면 정지로 판단
    distance = np.sqrt(dx ** 2 + dy ** 2)
    if distance < min_distance:
        return "stopped", 0

    # 각도 계산 (라디안 -> 도)
    # OpenCV는 y축이 아래로 증가하므로 dy를 반전
    angle = np.degrees(np.arctan2(-dy, dx))

    # 각도를 0-360 범위로 정규화
    if angle < 0:
        angle += 360

    # 4방향으로 분류 (각 방향당 90도 범위)
    if (angle >= 315) or (angle < 45):
        direction = "→ Right"
    elif 45 <= angle < 135:
        direction = "↑ Up"
    elif 135 <= angle < 225:
        direction = "← Left"
    else:  # 225 <= angle < 315
        direction = "↓ Down"

    return direction, angle


def get_movement_vector(history):
    """이동 벡터를 시각적으로 표시하기 위한 정보 반환"""
    if len(history) < 5:
        return None

    use_points = min(len(history), 15)
    pts = np.array(history[-use_points:])

    start_point = pts[0]
    end_point = pts[-1]

    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)

    if distance < 10:
        return None

    # 벡터를 정규화하고 크기를 조절
    scale = min(distance, 50)  # 최대 50픽셀

    return {
        'start': tuple(map(int, start_point)),
        'end': tuple(map(int, end_point)),
        'dx': dx,
        'dy': dy,
        'distance': distance,
        'scale': scale
    }


def get_direction_arrow(direction):
    """방향에 따른 화살표 반환"""
    arrows = {
        "→ Right": "→",
        "↑ Up": "↑",
        "← Left": "←",
        "↓ Down": "↓",
        "stopped": "●",
        "unknown": "●",
        "---": "●"
    }
    return arrows.get(direction, "●")


def get_direction_text(direction):
    """방향 문자열을 짧은 텍스트로 변환"""
    direction_map = {
        "→ Right": "Right",
        "↑ Up": "Up",
        "← Left": "Left",
        "↓ Down": "Down",
        "stopped": "Stop",
        "unknown": "---",
        "---": "---"
    }
    return direction_map.get(direction, direction)


def is_wrong_direction(current_dir, roi_dir):
    """
    현재 방향이 ROI 설정에 맞지 않는지 확인

    규칙:
    - ROI Up: Left 무시, Down/Right 경고
    - ROI Down: Right 무시, Up/Left 경고
    - ROI Left: Right만 경고
    - ROI Right: Left만 경고
    """
    # stopped나 unknown은 경고하지 않음
    if current_dir in ["stopped", "unknown", "---"]:
        return False

    # ROI가 Up인 경우
    if roi_dir == "↑ Up":
        # Left는 허용 (좌회전), Down과 Right는 경고
        if current_dir == "← Left":
            return False
        elif current_dir in ["↓ Down", "→ Right"]:
            return True

    # ROI가 Down인 경우
    elif roi_dir == "↓ Down":
        # Right는 허용 (우회전), Up과 Left는 경고
        if current_dir == "→ Right":
            return False
        elif current_dir in ["↑ Up", "← Left"]:
            return True

    # ROI가 Left인 경우
    elif roi_dir == "← Left":
        # 반대 방향(Right)만 경고
        if current_dir == "→ Right":
            return True

    # ROI가 Right인 경우
    elif roi_dir == "→ Right":
        # 반대 방향(Left)만 경고
        if current_dir == "← Left":
            return True

    return False


def get_center(xyxy):
    """xyxy 형식에서 중심점 계산"""
    x1, y1, x2, y2 = map(int, xyxy)
    return (x1 + x2) // 2, (y1 + y2) // 2


# 첫 프레임 읽기
ret, frame = cap.read()
if not ret:
    print("비디오를 읽을 수 없습니다.")
    exit()

# 🔹 ROI 설정을 위해 화면 크기 조정 (Full HD)
frame = cv2.resize(frame, (1920, 1080))

# 여러 ROI 선택
print("=" * 60)
print("여러 ROI 영역을 마우스 클릭으로 선택하세요.")
print("- 좌클릭: 포인트 추가 (최소 3개 이상)")
print("- 'r' 키: 마지막 포인트 제거")
print("- 'c' 키: 현재 ROI의 모든 포인트 초기화")
print("- '1' 키: 방향 설정 - Up (↑)")
print("- '2' 키: 방향 설정 - Down (↓)")
print("- '3' 키: 방향 설정 - Left (←)")
print("- '4' 키: 방향 설정 - Right (→)")
print("- 'n' 키: 현재 ROI 완성하고 새 ROI 시작 (방향 설정 필수!)")
print("- Enter 키: 모든 ROI 선택 완료")
print("- 's' 키: 건너뛰고 전체 화면 사용")
print("=" * 60)

temp_frame = frame.copy()
draw_legend(temp_frame)

cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("ROI Selection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("ROI Selection", select_roi_callback, {'frame': frame, 'temp_frame': temp_frame})

while not roi_selected:
    cv2.imshow("ROI Selection", temp_frame)
    key = cv2.waitKey(1)

    if key == ord('r'):
        if current_ROI_points:
            removed = current_ROI_points.pop()
            print(f"포인트 제거: {removed} - 남은 포인트: {len(current_ROI_points)}개")

            temp_frame = frame.copy()

            for idx, roi_points in enumerate(ROI_list):
                color = roi_colors[idx % len(roi_colors)]
                cv2.polylines(temp_frame, [np.array(roi_points)], True, color, 2)
                for pt in roi_points:
                    cv2.circle(temp_frame, pt, 5, color, -1)
                if roi_points:
                    centroid = np.mean(roi_points, axis=0).astype(int)
                    direction_text = ROI_directions[idx]
                    cv2.putText(temp_frame, f"ROI {idx + 1}: {direction_text}",
                                (centroid[0] - 50, centroid[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            color = roi_colors[len(ROI_list) % len(roi_colors)]
            for i, pt in enumerate(current_ROI_points):
                cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(temp_frame, str(i + 1), (pt[0] + 10, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if i > 0:
                    cv2.line(temp_frame, current_ROI_points[i - 1], pt, (0, 255, 0), 2)

            if len(current_ROI_points) >= 3:
                cv2.line(temp_frame, current_ROI_points[-1], current_ROI_points[0], (0, 255, 0), 2)
                cv2.polylines(temp_frame, [np.array(current_ROI_points)], True, color, 2)
                if current_direction:
                    centroid = np.mean(current_ROI_points, axis=0).astype(int)
                    cv2.putText(temp_frame, f"Direction: {current_direction}",
                                (centroid[0] - 50, centroid[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            draw_legend(temp_frame)

    elif key == ord('c'):
        current_ROI_points = []
        current_direction = None
        temp_frame = frame.copy()

        for idx, roi_points in enumerate(ROI_list):
            color = roi_colors[idx % len(roi_colors)]
            cv2.polylines(temp_frame, [np.array(roi_points)], True, color, 2)
            for pt in roi_points:
                cv2.circle(temp_frame, pt, 5, color, -1)
            if roi_points:
                centroid = np.mean(roi_points, axis=0).astype(int)
                direction_text = ROI_directions[idx]
                cv2.putText(temp_frame, f"ROI {idx + 1}: {direction_text}",
                            (centroid[0] - 50, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_legend(temp_frame)
        print("현재 ROI 초기화")

    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        current_direction = DIRECTIONS[chr(key)]
        print(f"방향 설정: {current_direction}")

        temp_frame = frame.copy()

        for idx, roi_points in enumerate(ROI_list):
            color = roi_colors[idx % len(roi_colors)]
            cv2.polylines(temp_frame, [np.array(roi_points)], True, color, 2)
            for pt in roi_points:
                cv2.circle(temp_frame, pt, 5, color, -1)
            if roi_points:
                centroid = np.mean(roi_points, axis=0).astype(int)
                direction_text = ROI_directions[idx]
                cv2.putText(temp_frame, f"ROI {idx + 1}: {direction_text}",
                            (centroid[0] - 50, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        color = roi_colors[len(ROI_list) % len(roi_colors)]
        for i, pt in enumerate(current_ROI_points):
            cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(temp_frame, str(i + 1), (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if i > 0:
                cv2.line(temp_frame, current_ROI_points[i - 1], pt, (0, 255, 0), 2)

        if len(current_ROI_points) >= 3:
            cv2.line(temp_frame, current_ROI_points[-1], current_ROI_points[0], (0, 255, 0), 2)
            cv2.polylines(temp_frame, [np.array(current_ROI_points)], True, color, 2)
            centroid = np.mean(current_ROI_points, axis=0).astype(int)
            cv2.putText(temp_frame, f"Direction: {current_direction}",
                        (centroid[0] - 50, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        draw_legend(temp_frame)

    elif key == ord('n'):
        if len(current_ROI_points) >= 3:
            if current_direction is None:
                print("⚠️  방향을 먼저 설정하세요! (1: Up, 2: Down, 3: Left, 4: Right)")
                continue

            ROI_list.append(current_ROI_points.copy())
            ROI_directions.append(current_direction)
            print(f"✓ ROI {len(ROI_list)} 완성! ({len(current_ROI_points)}개 포인트, 방향: {current_direction})")
            current_ROI_points = []
            current_direction = None

            temp_frame = frame.copy()
            for idx, roi_points in enumerate(ROI_list):
                color = roi_colors[idx % len(roi_colors)]
                cv2.polylines(temp_frame, [np.array(roi_points)], True, color, 2)
                for pt in roi_points:
                    cv2.circle(temp_frame, pt, 5, color, -1)
                if roi_points:
                    centroid = np.mean(roi_points, axis=0).astype(int)
                    direction_text = ROI_directions[idx]
                    cv2.putText(temp_frame, f"ROI {idx + 1}: {direction_text}",
                                (centroid[0] - 50, centroid[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            draw_legend(temp_frame)
            print(f"새 ROI {len(ROI_list) + 1} 시작...")
        else:
            print("최소 3개 이상의 포인트가 필요합니다.")

    elif key == 13:
        if len(current_ROI_points) >= 3:
            if current_direction is None:
                print("⚠️  마지막 ROI의 방향을 먼저 설정하세요!")
                continue
            ROI_list.append(current_ROI_points.copy())
            ROI_directions.append(current_direction)
            print(f"✓ ROI {len(ROI_list)} 완성! ({len(current_ROI_points)}개 포인트, 방향: {current_direction})")

        if ROI_list:
            roi_selected = True
            print(f"\n✓ 총 {len(ROI_list)}개의 ROI 선택 완료!")
            for i, direction in enumerate(ROI_directions):
                print(f"  ROI {i + 1}: {direction}")
        else:
            print("최소 1개 이상의 ROI가 필요합니다.")

    elif key == ord('s') or key == 27:
        ROI_list = []
        ROI_directions = []
        roi_selected = True
        print("전체 화면을 사용합니다.")
        break

cv2.destroyWindow("ROI Selection")

# 메인 트래킹 루프
frame_count = 0
track_roi_mapping = {}
last_refresh = time.time()
reconnect_attempts = 0
max_reconnect_attempts = 3

print("\n트래킹 시작... (ESC 키로 종료)")
print("트래커: BoT-SORT (Ultralytics 내장)")

while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠️ 프레임 읽기 실패 → 재연결 시도 중...")
        reconnect_attempts += 1

        if reconnect_attempts > max_reconnect_attempts:
            print(f"❌ {max_reconnect_attempts}회 재연결 실패. 종료합니다.")
            break

        cap.release()
        time.sleep(1)

        cctv_url = get_latest_cctv_url()
        if cctv_url is None:
            print("❌ CCTV URL을 가져올 수 없습니다.")
            continue

        cap = cv2.VideoCapture(cctv_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        last_refresh = time.time()
        continue

    reconnect_attempts = 0

    if time.time() - last_refresh > 600:
        print("♻️ 10분 경과 → URL 갱신 중...")
        cap.release()
        cctv_url = get_latest_cctv_url()
        if cctv_url:
            cap = cv2.VideoCapture(cctv_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        last_refresh = time.time()

    frame_count += 1
    frame = cv2.resize(frame, (1920, 1080))

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[2, 5, 7],
        conf=0.5,
        verbose=False
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for idx, roi_points in enumerate(ROI_list):
            if len(roi_points) >= 3:
                color = roi_colors[idx % len(roi_colors)]
                cv2.polylines(frame, [np.array(roi_points)], True, color, 2)
                for pt in roi_points:
                    cv2.circle(frame, pt, 3, color, -1)

                centroid = np.mean(roi_points, axis=0).astype(int)
                direction_text = ROI_directions[idx]

                arrow = get_direction_arrow(direction_text)
                dir_text = get_direction_text(direction_text)

                cv2.putText(frame, f"ROI {idx + 1}: {arrow} {dir_text}",
                            (centroid[0] - 60, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        active_tracks = 0

        for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            roi_idx = get_roi_index_for_bbox((x1, y1, x2, y2), ROI_list)

            if ROI_list and roi_idx is None:
                continue

            if roi_idx is not None:
                track_roi_mapping[track_id] = roi_idx

            cx, cy = get_center((x1, y1, x2, y2))

            if track_id not in histories:
                histories[track_id] = []
            histories[track_id].append((cx, cy))

            if len(histories[track_id]) > 100:
                histories[track_id] = histories[track_id][-100:]

            behavior = analyze_speed(histories[track_id], fps)
            direction, angle = analyze_direction(histories[track_id], min_distance=30)
            arrow = get_direction_arrow(direction)
            direction_short_text = get_direction_text(direction)

            movement = get_movement_vector(histories[track_id])

            is_wrong_way = False
            roi_direction_text = "N/A"

            if track_id in track_roi_mapping:
                roi_direction = ROI_directions[track_roi_mapping[track_id]]
                roi_direction_text = get_direction_text(roi_direction)

                if direction != "stopped" and direction != "unknown":
                    is_wrong_way = is_wrong_direction(direction, roi_direction)

            if behavior == "stopped":
                color = (255, 255, 0)
            elif behavior == "slow":
                color = (0, 255, 0)
            elif behavior == "fast":
                color = (0, 165, 255)
            else:
                color = (128, 128, 128)

            thickness = 3 if is_wrong_way else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            y_offset = max(y1 - 50, 30)

            if is_wrong_way:
                cv2.putText(frame, f"⚠️ WRONG WAY!",
                            (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 22

            cv2.putText(frame, f"ID:{track_id}",
                        (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f"Speed: {behavior}",
                        (x1, y_offset + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f"Dir: {arrow} {direction_short_text}",
                        (x1, y_offset + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if direction != "stopped" and direction != "unknown":
                cv2.putText(frame, f"{angle:.0f}°",
                            (x1, y_offset + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if roi_direction_text != "N/A":
                cv2.putText(frame, f"ROI: {roi_direction_text}",
                            (x1 + 90, y_offset + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            if len(histories[track_id]) > 1:
                pts = np.array(histories[track_id][-30:], dtype=np.int32)

                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    thickness_line = int(1 + alpha * 2)
                    cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), color, thickness_line)

                if len(pts) >= 5 and direction != "stopped" and direction != "unknown":
                    recent_pts = pts[-5:]
                    start_pt = tuple(recent_pts[0])
                    end_pt = tuple(recent_pts[-1])

                    dist = np.sqrt((end_pt[0] - start_pt[0]) ** 2 + (end_pt[1] - start_pt[1]) ** 2)
                    if dist > 10:
                        arrow_color = (0, 0, 255) if is_wrong_way else (0, 255, 255)
                        cv2.arrowedLine(frame, start_pt, end_pt, arrow_color, 4, tipLength=0.4)

            if direction != "stopped" and direction != "unknown":
                length = 50
                angle_rad = np.radians(angle)

                end_x = int(cx + length * np.cos(angle_rad))
                end_y = int(cy - length * np.sin(angle_rad))

                arrow_color = (0, 0, 255) if is_wrong_way else (0, 255, 255)
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y),
                                arrow_color, 4, tipLength=0.3)

            active_tracks += 1
    else:
        for idx, roi_points in enumerate(ROI_list):
            if len(roi_points) >= 3:
                color = roi_colors[idx % len(roi_colors)]
                cv2.polylines(frame, [np.array(roi_points)], True, color, 2)
                for pt in roi_points:
                    cv2.circle(frame, pt, 3, color, -1)

                centroid = np.mean(roi_points, axis=0).astype(int)
                direction_text = ROI_directions[idx]

                arrow = get_direction_arrow(direction_text)
                dir_text = get_direction_text(direction_text)

                cv2.putText(frame, f"ROI {idx + 1}: {arrow} {dir_text}",
                            (centroid[0] - 60, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        active_tracks = 0

    cv2.putText(frame, f"Tracks: {active_tracks}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"ROIs: {len(ROI_list)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    legend_x = frame.shape[1] - 230
    legend_y = frame.shape[0] - 180

    overlay = frame.copy()
    cv2.rectangle(overlay, (legend_x - 10, legend_y - 30),
                  (frame.shape[1] - 10, frame.shape[0] - 10),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "=== Legend ===", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.putText(frame, "Speed Colors:", (legend_x, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.rectangle(frame, (legend_x, legend_y + 30), (legend_x + 15, legend_y + 42),
                  (255, 255, 0), -1)
    cv2.putText(frame, "Stopped", (legend_x + 20, legend_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.rectangle(frame, (legend_x, legend_y + 48), (legend_x + 15, legend_y + 60),
                  (0, 255, 0), -1)
    cv2.putText(frame, "Slow", (legend_x + 20, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.rectangle(frame, (legend_x, legend_y + 66), (legend_x + 15, legend_y + 78),
                  (0, 165, 255), -1)
    cv2.putText(frame, "Fast", (legend_x + 20, legend_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.putText(frame, "Direction Arrows:", (legend_x, legend_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, "Yellow: Normal", (legend_x, legend_y + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, "Red: Wrong Way", (legend_x, legend_y + 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.putText(frame, "Up: 45-135° | Down: 225-315°", (legend_x, legend_y + 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(frame, "Left: 135-225° | Right: 315-45°", (legend_x, legend_y + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    cv2.imshow("SmartCity AI - Direction Tracking (LIVE CCTV)", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n트래킹 완료! 총 {frame_count}프레임 처리됨")
print(f"총 추적된 객체 수: {len(histories)}")
