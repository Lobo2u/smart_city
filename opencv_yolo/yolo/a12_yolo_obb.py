import cv2
from ultralytics import YOLO
import torch
import numpy as np

def main():
    print("cuda on" if torch.cuda.is_available() else "cuda off")
    model = YOLO("../yolov8n-obb.pt")

    img = cv2.imread("../../data/car_dron.jpg")

    results = model.predict(img)  # type: ignore
    print(results[0].boxes)
    print(results[0].names)
    print(results[0].obb)

    res = results[0].cpu()

    # OBB가 존재하는지 확인
    if res.obb is not None and len(res.obb.data) > 0:
        # OBB visualization
        for i, obb_data in enumerate(res.obb.data):
            # 클래스 라벨 가져오기
            cls_id = int(res.obb.cls[i])
            label = res.names.get(cls_id, "unknown")

            # OBB 데이터 추출 (x_center, y_center, width, height, rotation)
            obb_info = obb_data.cpu().numpy()

            # 회전된 박스의 4개 꼭짓점 계산
            x_center, y_center, width, height, rotation = obb_info[:5]

            # 회전 행렬 생성
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)

            # 박스의 반폭과 반높이
            half_w = width / 2
            half_h = height / 2

            # 회전되지 않은 상태의 4개 꼭짓점
            corners = np.array([
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h]
            ])

            # 회전 적용
            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            rotated_corners = corners @ rotation_matrix.T

            # 중심점으로 이동
            final_corners = rotated_corners + np.array([x_center, y_center])
            final_corners = final_corners.astype(np.int32)

            print(f"OBB data for {label}: center=({x_center:.1f}, {y_center:.1f}), "
                  f"size=({width:.1f}, {height:.1f}), rotation={rotation:.3f}")

            # 다각형 그리기
            cv2.polylines(img, [final_corners], isClosed=True, color=(0, 255, 0), thickness=2)

            # 라벨 텍스트 추가
            text_pos = (int(x_center) - 20, int(y_center) - 10)
            cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("No OBB detected in the image")

    cv2.imshow("OBB Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()