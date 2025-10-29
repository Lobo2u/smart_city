import time

import cv2
import torch
from ultralytics import YOLO


def main():
    print("cuda on" if torch.cuda.is_available() else "cuda off")

    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    fps = cap.get(cv2.CAP_PROP_FPS) # 30
    print("fps:", fps)

    tm = cv2.TickMeter()
    target_ticks_per_frame = cv2.getTickFrequency() / fps  # 1프레임당 필요한 tick 수

    while True:
        tm.start()
        ret , img = cap.read()
        if not ret:
            break
        # yolo inference
        results = model.predict(img, verbose=False)  # type: ignore
        # plot keypoints
        annotated_frame = results[0].plot()  # type: ignore
        for res in results[0]:
            keypoints = res.keypoints.xy.cpu().numpy()  # type: ignore
            if keypoints.shape[0] < 4:
                continue
            print("keypoints left_eye:", keypoints[1], "right_eye:", keypoints[2])  # type: ignore
        img = annotated_frame
        cv2.putText(img, f"FPS: {tm.getFPS():.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video", img)
        tm.stop()
        print("time:", tm.getTimeSec(), "fps:", tm.getFPS(), "ticks:", tm.getTimeTicks())

        # getTimeTicks를 사용한 FPS 제어
        elapsed_ticks = tm.getTimeTicks()
        if elapsed_ticks < target_ticks_per_frame:
            wait_ticks = target_ticks_per_frame - elapsed_ticks
            wait_ms = int(wait_ticks / cv2.getTickFrequency() * 1000)
            if cv2.waitKey(wait_ms) == 27:
                break
        else:
            if cv2.waitKey(1) == 27:
                break
        tm.reset()
    cap.release()

if __name__ == "__main__":
    main()