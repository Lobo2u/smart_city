# yunet_ort_optimized.py
import cv2 as cv
import numpy as np
import onnxruntime as ort


class YuNet:
    """
    FaceDetectorYN C++ 로직을 onnxruntime-gpu에 최적화하여 구현
    - Preprocessing: cv2.copyMakeBorder + cv2.dnn.blobFromImage 사용
    - NMS: cv2.dnn.NMSBoxes 사용
    - Postprocessing: 완전 벡터화
    """
    def __init__(self, modelPath,
                 inputSize=(0,0),
                 confThreshold=0.6,
                 nmsThreshold=0.3, topK=5000,
                 providers=("CUDAExecutionProvider","CPUExecutionProvider"),
                 backendId=0, targetId=0):
        self._sess = ort.InferenceSession(modelPath, providers=list(providers))
        self._inW, self._inH = int(inputSize[0]), int(inputSize[1])
        self._score_thr = float(confThreshold)
        self._nms_thr = float(nmsThreshold)
        self._topK = int(topK)
        self._strides = [8, 16, 32]

        in0 = self._sess.get_inputs()[0]
        _, _, mh, mw = [int(x) if x is not None else 0 for x in in0.shape]
        if mh == 0 or mw == 0:
            self._fixed = False
            self._modelH, self._modelW = None, None
        else:
            self._fixed = True
            self._modelH, self._modelW = mh, mw

        self._out_names = [o.name for o in self._sess.get_outputs()]
        self._input_name = in0.name
        self._padW, self._padH = None, None

    def setInputSize(self, size):
        W, H = int(size[0]), int(size[1])
        self._inW, self._inH = W, H
        if self._fixed:
            self._padW, self._padH = self._modelW, self._modelH
        else:
            def ceil32(x): return ((x - 1) // 32 + 1) * 32
            self._padW, self._padH = ceil32(W), ceil32(H)

    def getInputSize(self): return (self._inW, self._inH)
    def setScoreThreshold(self, t): self._score_thr = float(t)
    def setNMSThreshold(self, t): self._nms_thr = float(t)
    def setTopK(self, k): self._topK = int(k)

    def _preprocess(self, image):
        H, W = image.shape[:2]
        if (self._inW, self._inH) != (W, H):
            self.setInputSize((W, H))

        padW, padH = (self._modelW, self._modelH) if self._fixed else (self._padW, self._padH)
        if self._fixed and (W > padW or H > padH):
            raise RuntimeError(f"Image {W}x{H} larger than fixed model {padW}x{padH}.")

        # 최적화: OpenCV C++ 함수를 사용하여 패딩과 blob 생성을 효율적으로 처리
        padded_image = cv.copyMakeBorder(image, 0, padH - H, 0, padW - W, cv.BORDER_CONSTANT, value=(0,0,0))
        blob = cv.dnn.blobFromImage(padded_image, scalefactor=1.0, swapRB=False)
        return blob, (W, H)

    def _flat(self, arr):
        return np.asarray(arr).flatten()

    def infer(self, image):
        blob, (W, H) = self._preprocess(image)
        padH, padW = blob.shape[2], blob.shape[3]

        outs = self._sess.run(self._out_names, {self._input_name: blob})

        faces_xywh, faces_lmk, faces_score = [], [], []

        for i, stride in enumerate(self._strides):
            cols, rows = padW // stride, padH // stride
            N = rows * cols

            cls = self._flat(outs[0 + i])[:N]
            obj = self._flat(outs[3 + i])[:N]
            bbox = self._flat(outs[6 + i])[:N*4]
            kps = self._flat(outs[9 + i])[:N*10]

            score = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
            keep = np.where(score >= self._score_thr)[0]
            if keep.size == 0:
                continue

            r, c = keep // cols, keep % cols

            # BBox decoding
            dx, dy = bbox.reshape(-1, 4)[keep, 0], bbox.reshape(-1, 4)[keep, 1]
            dw, dh = bbox.reshape(-1, 4)[keep, 2], bbox.reshape(-1, 4)[keep, 3]

            cx, cy = (c + dx) * stride, (r + dy) * stride
            w, h = np.exp(dw) * stride, np.exp(dh) * stride

            x1, y1 = cx - w * 0.5, cy - h * 0.5

            # Landmarks decoding (Vectorized)
            kps_kept = kps.reshape(-1, 10)[keep]
            lmk = np.empty_like(kps_kept)
            lmk[:, 0::2] = (kps_kept[:, 0::2] + c[:, np.newaxis]) * stride
            lmk[:, 1::2] = (kps_kept[:, 1::2] + r[:, np.newaxis]) * stride

            # Clipping to original image boundaries
            x1 = np.clip(x1, 0, W - 1)
            y1 = np.clip(y1, 0, H - 1)
            # w, h는 x2, y2를 계산한 후 클리핑하는 것이 더 정확할 수 있습니다.
            x2 = np.clip(x1 + w, 0, W - 1)
            y2 = np.clip(y1 + h, 0, H - 1)
            w, h = x2 - x1, y2 - y1

            faces_xywh.append(np.stack([x1, y1, w, h], axis=1))
            faces_lmk.append(lmk)
            faces_score.append(score[keep])

        if not faces_xywh:
            return np.empty((0, 15), dtype=np.float32)

        boxes = np.concatenate(faces_xywh, axis=0).astype(np.float32)
        lmks = np.concatenate(faces_lmk, axis=0).astype(np.float32)
        scores = np.concatenate(faces_score, axis=0).astype(np.float32)

        # 최적화: NMS 연산 전에 topK로 후보 수를 줄여 연산량 감소
        if scores.shape[0] > self._topK:
            top_indices = scores.argsort()[::-1][:self._topK]
            boxes, lmks, scores = boxes[top_indices], lmks[top_indices], scores[top_indices]

        # 최적화: OpenCV의 고성능 C++ NMS 함수 사용
        keep = cv.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self._score_thr, self._nms_thr)

        if isinstance(keep, tuple): # Older OpenCV versions might return a tuple
            return np.empty((0,15), dtype=np.float32)

        keep = keep.flatten()
        if keep.size == 0:
            return np.empty((0, 15), dtype=np.float32)

        boxes, lmks, scores = boxes[keep], lmks[keep], scores[keep]

        return np.concatenate([boxes, lmks, scores[:, None]], axis=1)

# --- 사용 예 ---
if __name__ == "__main__":
    det = YuNet(
        modelPath="face_detection_yunet_2023mar.onnx",
        inputSize=(0,0),
        confThreshold=0.9,
        nmsThreshold=0.3,
        topK=5000,
        providers=("CUDAExecutionProvider","CPUExecutionProvider")
    )

    img = cv.imread("test.jpg")
    det.setInputSize((img.shape[1], img.shape[0]))
    faces = det.infer(img)
    print(f"Detected {faces.shape[0]} faces.")
    print("First 3 faces:\n", faces[:min(3, len(faces))])