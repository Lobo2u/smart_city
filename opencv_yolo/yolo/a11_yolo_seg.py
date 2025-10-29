import enum

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    vtest = 0
    cap = cv2.VideoCapture(vtest)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)
    model = YOLO("yolov8n-seg.pt")

    while True:
        ret , img = cap.read()
        if not ret:
            break

        results = model.predict(img, verbose=True)
        res = results[0]
        # annotated_frame = res.plot()
        class_info = []
        index_info = []
        for i, cls in enumerate(res.boxes.cls):
            label = res.names.get(int(cls), "unknown")
            class_info.append(label)
            index_info.append(i)

        # mask visualization
        for label, idx in zip(class_info, index_info):
            mask = res.masks.data[idx].cpu().numpy()
            kernel = np.ones((15,15), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
            yellow = np.full_like(img, (0,255,255))
            frame = np.where(mask[..., None], yellow, img)
        annotated_frame = frame
        cv2.imshow("video", annotated_frame)
        if cv2.waitKey(3) == 27: # ms fps
            break
    cap.release()

if __name__ == "__main__":
    main()