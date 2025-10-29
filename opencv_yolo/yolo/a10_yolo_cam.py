import cv2
from ultralytics import YOLO


def main():
    vtest = 0
    cap = cv2.VideoCapture(vtest)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)
    model = YOLO("yolov8n.pt")

    while True:
        ret , img = cap.read()
        results = model.predict(img, verbose=True)
        res = results[0]
        annotated_frame = res.plot()
        if not ret:
            break
        cv2.imshow("video", annotated_frame)
        if cv2.waitKey(3) == 27: # ms fps
            break
    cap.release()

if __name__ == "__main__":
    main()