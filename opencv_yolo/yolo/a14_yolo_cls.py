# 설치
# pip install ultralytics
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 모델 로드
    model = YOLO("yolov8n-cls.pt")  # yolo11n-cls.pt

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    start = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=False, verbose=False)

        res = results[0]
        # print(f"res.boxes: {res.boxes}")
        # print(f"res.names: {res.names}")
        # print(f"res.keypoints: {res.keypoints}")
        # print(f"res.masks: {res.masks}")
        # print(f"res.probes: {res.probs}")

        annotated = results[0].plot()
        frames += 1
        fps = (frames / (time.time() - start))
        # cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        print(res.names)
        print(len(res.names))
        # print top1 class

        if res.probs.top1 is not None:
            print(f"Top1: {res.probs.top1}, {res.names[res.probs.top1]}")
            print(f"Top1 confidence: {res.probs.top1conf.cpu().numpy()}")

        cv2.imshow("YOLOv8 Inference", annotated)
        if cv2.waitKey(2) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()