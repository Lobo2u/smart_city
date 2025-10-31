from ultralytics import YOLO
import cv2
import numpy as np
import sys

sys.path.append('opencv_yolo/yolo')
from vittrack import VitTrack

model = YOLO("yolov8n.pt")  # 경량모델
trackers = {}  # tracker_id별 VitTrack 인스턴스 저장
tracker_boxes = {}  # 각 트래커의 현재 바운딩 박스 저장
histories = {}
next_id = 0

cap = cv2.VideoCapture("../../data/봉안교차로.mp4")  # 또는 rtsp:// 주소
fps = cap.get(cv2.CAP_PROP_FPS)

# 여러 ROI 영역 설정
ROI_list = []  # 여러 개의 ROI를 저장하는 리스트
current_ROI_points = []  # 현재 그리고 있는 ROI의 포인트들
roi_selected = False
roi_colors = [
    (0, 255, 255),  # 노란색
    (255, 0, 255),  # 자홍색
    (0, 255, 0),    # 녹색
    (255, 128, 0),  # 주황색
    (128, 0, 255),  # 보라색
    (0, 128, 255),  # 하늘색
]

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
            # ROI 번호 표시
            if roi_points:
                centroid = np.mean(roi_points, axis=0).astype(int)
                cv2.putText(temp_frame, f"ROI {idx+1}", tuple(centroid), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 현재 그리고 있는 ROI 그리기
        color = roi_colors[len(ROI_list) % len(roi_colors)]
        for i, pt in enumerate(current_ROI_points):
            cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(temp_frame, str(i+1), (pt[0]+10, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if i > 0:
                cv2.line(temp_frame, current_ROI_points[i-1], pt, (0, 255, 0), 2)
        
        # 3개 이상의 점이 있으면 첫 점과 마지막 점을 연결하여 폴리곤 표시
        if len(current_ROI_points) >= 3:
            cv2.line(temp_frame, current_ROI_points[-1], current_ROI_points[0], (0, 255, 0), 2)
            cv2.polylines(temp_frame, [np.array(current_ROI_points)], True, color, 2)
        
        # 범례 표시
        draw_legend(temp_frame)
        
        param['temp_frame'][:] = temp_frame


def draw_legend(frame):
    """화면에 조작 방법 범례 표시"""
    # 반투명 배경 생성
    overlay = frame.copy()
    legend_x, legend_y = 10, 10
    legend_width, legend_height = 420, 230
    
    cv2.rectangle(overlay, (legend_x, legend_y), 
                  (legend_x + legend_width, legend_y + legend_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 테두리
    cv2.rectangle(frame, (legend_x, legend_y), 
                  (legend_x + legend_width, legend_y + legend_height), 
                  (255, 255, 255), 2)
    
    # 제목
    title_y = legend_y + 30
    cv2.putText(frame, "=== ROI Selection Guide ===", 
                (legend_x + 20, title_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 조작 방법 텍스트
    instructions = [
        ("Left Click", "Add point", (0, 255, 0)),
        ("'r' key", "Remove last point", (0, 200, 255)),
        ("'c' key", "Clear current ROI", (0, 150, 255)),
        ("'n' key", "Finish & start new ROI", (255, 128, 0)),
        ("Enter", "Complete all ROIs", (0, 255, 255)),
        ("'s' or ESC", "Skip (use full screen)", (150, 150, 150)),
    ]
    
    start_y = title_y + 30
    line_height = 28
    
    for i, (action, description, color) in enumerate(instructions):
        y_pos = start_y + i * line_height
        
        # 키/액션 이름
        cv2.putText(frame, action, 
                    (legend_x + 15, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 설명
        cv2.putText(frame, f": {description}", 
                    (legend_x + 140, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 현재 상태 표시
    status_y = start_y + len(instructions) * line_height + 15
    status_text = f"Completed ROIs: {len(ROI_list)} | Current points: {len(current_ROI_points)}"
    cv2.putText(frame, status_text, 
                (legend_x + 15, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def is_in_roi(bbox, roi_points, overlap_threshold=0.5):
    """바운딩 박스가 ROI 영역과 일정 비율 이상 겹치는지 확인"""
    if roi_points is None or len(roi_points) < 3:
        return True
    
    x, y, w, h = bbox
    
    # 바운딩 박스의 4개 꼭지점
    bbox_corners = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.int32)
    
    # ROI 폴리곤
    roi_contour = np.array(roi_points, dtype=np.int32)
    
    # 바운딩 박스 영역의 최소 경계 사각형 찾기
    min_x = max(0, x - 10)
    min_y = max(0, y - 10)
    max_x = x + w + 10
    max_y = y + h + 10
    
    # 작은 영역에 대한 마스크 생성
    mask_width = max_x - min_x
    mask_height = max_y - min_y
    
    # 바운딩 박스 마스크
    bbox_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    shifted_bbox = bbox_corners - np.array([min_x, min_y])
    cv2.fillPoly(bbox_mask, [shifted_bbox], 255)
    
    # ROI 마스크
    roi_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    shifted_roi = roi_contour - np.array([min_x, min_y])
    cv2.fillPoly(roi_mask, [shifted_roi], 255)
    
    # 교집합 계산
    intersection = cv2.bitwise_and(bbox_mask, roi_mask)
    overlap_area = np.sum(intersection > 0)
    bbox_area = w * h
    
    if bbox_area == 0:
        return False
    
    overlap_ratio = overlap_area / bbox_area
    
    # overlap_threshold 이상 겹치면 True (기본값 50%)
    return overlap_ratio >= overlap_threshold


def is_in_any_roi(bbox, roi_list, overlap_threshold=0.5):
    """바운딩 박스가 여러 ROI 중 하나라도 겹치는지 확인"""
    if not roi_list:
        return True
    
    for roi_points in roi_list:
        if is_in_roi(bbox, roi_points, overlap_threshold):
            return True
    
    return False


# ... existing code ...

def analyze_speed(history, fps):
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


def get_center(bbox):
    """bbox (x, y, w, h) 형식에서 중심점 계산"""
    x, y, w, h = bbox
    return int(x + w / 2), int(y + h / 2)


def calculate_iou(box1, box2):
    """두 박스의 IoU 계산 (x, y, w, h 형식)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 교집합 영역 계산
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 합집합 영역 계산
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def is_overlapping_with_existing(new_box, existing_boxes, threshold=0.3):
    """새로운 박스가 기존 박스와 겹치는지 확인"""
    for existing_box in existing_boxes.values():
        if calculate_iou(new_box, existing_box) > threshold:
            return True
    return False


# 첫 프레임 읽기
ret, frame = cap.read()
if not ret:
    print("비디오를 읽을 수 없습니다.")
    exit()

# 여러 ROI 선택
print("=" * 50)
print("여러 ROI 영역을 마우스 클릭으로 선택하세요.")
print("- 좌클릭: 포인트 추가 (최소 3개 이상)")
print("- 'r' 키: 마지막 포인트 제거")
print("- 'c' 키: 현재 ROI의 모든 포인트 초기화")
print("- 'n' 키: 현재 ROI 완성하고 새 ROI 시작")
print("- Enter 키: 모든 ROI 선택 완료")
print("- 's' 키: 건너뛰고 전체 화면 사용")
print("=" * 50)

temp_frame = frame.copy()
draw_legend(temp_frame)  # 초기 범례 표시

cv2.namedWindow("ROI Selection")
cv2.setMouseCallback("ROI Selection", select_roi_callback, {'frame': frame, 'temp_frame': temp_frame})

while not roi_selected:
    cv2.imshow("ROI Selection", temp_frame)
    key = cv2.waitKey(1)
    
    if key == ord('r'):  # 마지막 포인트 제거
        if current_ROI_points:
            removed = current_ROI_points.pop()
            print(f"포인트 제거: {removed} - 남은 포인트: {len(current_ROI_points)}개")
            
            # 프레임 다시 그리기
            temp_frame = frame.copy()
            
            # 완성된 ROI들 그리기
            for idx, roi_points in enumerate(ROI_list):
                color = roi_colors[idx % len(roi_colors)]
                cv2.polylines(temp_frame, [np.array(roi_points)], True, color, 2)
                for pt in roi_points:
                    cv2.circle(temp_frame, pt, 5, color, -1)
                if roi_points:
                    centroid = np.mean(roi_points, axis=0).astype(int)
                    cv2.putText(temp_frame, f"ROI {idx+1}", tuple(centroid), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 현재 ROI 그리기
            color = roi_colors[len(ROI_list) % len(roi_colors)]
            for i, pt in enumerate(current_ROI_points):
                cv2.circle(temp_frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(temp_frame, str(i+1), (pt[0]+10, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if i > 0:
                    cv2.line(temp_frame, current_ROI_points[i-1], pt, (0, 255, 0), 2)
            
            if len(current_ROI_points) >= 3:
                cv2.line(temp_frame, current_ROI_points[-1], current_ROI_points[0], (0, 255, 0), 2)
                cv2.polylines(temp_frame, [np.array(current_ROI_points)], True, color, 2)
            
            # 범례 표시
            draw_legend(temp_frame)
    
    elif key == ord('c'):  # 현재 ROI의 모든 포인트 초기화
        current_ROI_points = []
        temp_frame = frame.copy()
        
        # 완성된 ROI들만 다시 그리기
        for idx, roi_points in enumerate(ROI_list):
            color = roi_colors[idx % len(roi_colors)]
            cv2.polylines(temp_frame, [np.array(roi_points)], True, color, 2)
            for pt in roi_points:
                cv2.circle(temp_frame, pt, 5, color, -1)
            if roi_points:
                centroid = np.mean(roi_points, axis=0).astype(int)
                cv2.putText(temp_frame, f"ROI {idx+1}", tuple(centroid), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 범례 표시
        draw_legend(temp_frame)
        
        print("현재 ROI 초기화")
    
    elif key == ord('n'):  # 현재 ROI 완성하고 새 ROI 시작
        if len(current_ROI_points) >= 3:
            ROI_list.append(current_ROI_points.copy())
            print(f"ROI {len(ROI_list)} 완성! ({len(current_ROI_points)}개 포인트)")
            current_ROI_points = []
            
            # 프레임 다시 그리기
            temp_frame = frame.copy()
            for idx, roi_points in enumerate(ROI_list):
                color = roi_colors[idx % len(roi_colors)]
                cv2.polylines(temp_frame, [np.array(roi_points)], True, color, 2)
                for pt in roi_points:
                    cv2.circle(temp_frame, pt, 5, color, -1)
                if roi_points:
                    centroid = np.mean(roi_points, axis=0).astype(int)
                    cv2.putText(temp_frame, f"ROI {idx+1}", tuple(centroid), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 범례 표시
            draw_legend(temp_frame)
            
            print(f"새 ROI {len(ROI_list) + 1} 시작...")
        else:
            print("최소 3개 이상의 포인트가 필요합니다.")
    
    elif key == 13:  # Enter 키 - 모든 ROI 선택 완료
        if len(current_ROI_points) >= 3:
            ROI_list.append(current_ROI_points.copy())
            print(f"ROI {len(ROI_list)} 완성! ({len(current_ROI_points)}개 포인트)")
        
        if ROI_list:
            roi_selected = True
            print(f"총 {len(ROI_list)}개의 ROI 선택 완료!")
        else:
            print("최소 1개 이상의 ROI가 필요합니다.")
    
    elif key == ord('s') or key == 27:  # 's' 또는 ESC로 건너뛰기
        ROI_list = []
        roi_selected = True
        print("전체 화면을 사용합니다.")
        break

cv2.destroyWindow("ROI Selection")

frame_count = 0
detection_interval = 30  # 30프레임마다 새로운 객체 감지

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # 모든 ROI 영역 표시
    for idx, roi_points in enumerate(ROI_list):
        if len(roi_points) >= 3:
            color = roi_colors[idx % len(roi_colors)]
            cv2.polylines(frame, [np.array(roi_points)], True, color, 2)
            for pt in roi_points:
                cv2.circle(frame, pt, 3, color, -1)
            
            # ROI 번호 표시
            centroid = np.mean(roi_points, axis=0).astype(int)
            cv2.putText(frame, f"ROI {idx+1}", tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 주기적으로 새로운 차량 감지 (기존 트래커와 겹치지 않는 경우만)
    if frame_count % detection_interval == 1:
        detections = model(frame, verbose=False)[0]
        boxes = detections.boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in [2, 5, 7] or conf < 0.5:  # 차량 클래스만, 신뢰도 0.5 이상
                continue

            # xyxy를 xywh로 변환
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            roi = (x1, y1, w, h)

            # 여러 ROI 영역 중 하나라도 겹치는지 체크
            if not is_in_any_roi(roi, ROI_list, overlap_threshold=0.5):
                continue

            # 기존 트래커와 겹치는지 확인
            if is_overlapping_with_existing(roi, tracker_boxes, threshold=0.3):
                continue

            # 새로운 트래커 생성
            tid = next_id
            next_id += 1

            try:
                tracker = VitTrack(
                    model_path='../../data/object_tracking_vittrack_2023sep.onnx',
                    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv2.dnn.DNN_TARGET_CPU
                )
                tracker.init(frame, roi)
                trackers[tid] = tracker
                tracker_boxes[tid] = roi
                histories[tid] = [get_center(roi)]
            except Exception as e:
                print(f"트래커 생성 실패: {e}")
                continue

    # 모든 트래커 업데이트
    active_trackers = {}
    active_boxes = {}

    for tid, tracker in trackers.items():
        try:
            isLocated, bbox, score = tracker.infer(frame)

            if isLocated and score >= 0.3:
                x, y, w, h = bbox
                cx, cy = get_center(bbox)

                # 히스토리 업데이트
                histories.setdefault(tid, []).append((cx, cy))

                # 속도 분석
                beh = analyze_speed(histories[tid], fps)
                color = (0, 255, 0) if beh == "slow" else (0, 0, 255) if beh == "fast" else (255, 255, 0)

                # 바운딩 박스와 정보 표시
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID:{tid} {beh}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"{score:.2f}", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                active_trackers[tid] = tracker
                active_boxes[tid] = bbox
        except Exception as e:
            # 트래킹 실패시 제거
            continue

    # 활성 트래커만 유지
    trackers = active_trackers
    tracker_boxes = active_boxes

    # 정보 표시
    cv2.putText(frame, f"Trackers: {len(trackers)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"ROIs: {len(ROI_list)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("SmartCity AI - VitTrack", frame)
    if cv2.waitKey(1) == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()
