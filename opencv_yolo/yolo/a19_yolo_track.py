# uv run a19_yolo_track.py --source ../../data/vtest.avi

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# (선택) 설치되어 있으면 예쁜 라벨 박스용
# pip install supervision
try:
    import supervision as sv
    HAS_SV = True
except Exception:
    HAS_SV = False


TRACKER_MAP = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}

def load_tracker_config(tracker_path):
    """트래커 설정 파일을 로드하고 설정 정보를 로깅"""
    try:
        with open(tracker_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logging.info(f"=== 트래커 설정 로드됨: {tracker_path} ===")
        logging.info(f"트래커 타입: {config.get('tracker_type', 'Unknown')}")

        if config.get('tracker_type') == 'botsort':
            logging.info(f"ReID 활성화: {config.get('with_reid', False)}")
            if config.get('with_reid', False):
                logging.info(f"ReID 모델: {config.get('model', 'auto')}")
            logging.info(f"트랙 버퍼: {config.get('track_buffer', 30)} 프레임")
            logging.info(f"매칭 임계값: {config.get('match_thresh', 0.8)}")
            logging.info(f"외형 임계값: {config.get('appearance_thresh', 0.25)}")
        logging.info(f"ReID 사용: {config.get('with_reid', False)}")
        logging.info(f"ReID 모델: {config.get('model', 'auto')}")

        return config

    except Exception as e:
        logging.error(f"트래커 설정 파일 로드 실패: {e}")
        return None

def parse_args():
    p = argparse.ArgumentParser(description="YOLO + (ByteTrack/BoT-SORT) MOT 데모")
    p.add_argument("--source", type=str, default="0",
                   help="영상 경로 혹은 카메라 인덱스 (예: 0)")
    p.add_argument("--model", type=str, default="yolov8l.pt",
                   help="Ultralytics YOLO 가중치 (예: yolov8n.pt, yolov8l.pt, yolov8x.pt 등)")
    p.add_argument("--tracker", type=str, default="bytetrack", choices=list(TRACKER_MAP.keys()),
                   help="트래커 선택: bytetrack | botsort")
    p.add_argument("--conf", type=float, default=0.1, help="추론 confidence threshold (낮게 설정)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (높게 설정)")
    # p.add_argument("--device", type=str, default=4,
    #                help="장치 선택 (예: '0' 또는 'cpu'). 기본은 자동")
    p.add_argument("--save", type=str, default=None,
                   help="저장할 출력 영상 경로 (예: output.mp4). 지정 안 하면 저장 안 함")
    p.add_argument("--show", action="store_true", default=True, help="윈도우에 실시간 표시")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="특정 클래스만 추적 (COCO id 리스트, 예: --classes 0 2 7)")
    return p.parse_args()


def open_writer(example_frame, save_path, fps=30):
    h, w = example_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))


def main():
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    args = parse_args()

    # source 처리 (숫자면 웹캠)
    source = args.source
    if source.isdigit():
        source = int(source)

    # 모델 로드
    model = YOLO(args.model)
    logging.info(f"YOLO 모델 로드됨: {args.model}")

    # tracker 설정 파일명 및 로딩
    tracker_cfg = TRACKER_MAP[args.tracker]
    config = load_tracker_config(tracker_cfg)

    if config is None:
        logging.warning("트래커 설정 로드 실패, 기본 설정으로 진행")

    # stream 모드로 한 프레임씩 결과 뽑기
    logging.info(f"트래킹 시작 - Source: {source}, Tracker: {args.tracker}")

    try:

        gen = model.track(
            source=source,
            stream=True,
            tracker=tracker_cfg,
            persist=True,               # ID 유지 필요
            conf=args.conf,
            iou=args.iou,
            # device=args.device,
            classes=args.classes
        )


        logging.info("트래커 초기화 성공")

        # BoT-SORT ReID 모델 로딩 확인
        if args.tracker == 'botsort' and config and config.get('with_reid', False):
            logging.info("ReID 모델 다운로드 및 로딩 상태 확인 중...")
            # 첫 번째 프레임 처리 후 ReID 모델 상태 확인
            frame_count = 0

    except Exception as e:
        logging.error(f"트래커 초기화 실패: {e}")
        return

    writer = None
    last_time = time.time()
    fps = 0.0

    # supervision 설정 (있을 경우)
    if HAS_SV:
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

    for i, result in enumerate(gen):
        frame = result.orig_img
        if frame is None:
            continue

        # ReID 모델 로딩 상태 확인 (첫 5프레임에서만)
        if args.tracker == 'botsort' and config and config.get('with_reid', False) and i < 5:
            if hasattr(result, 'tracker') and hasattr(result.tracker, 'reid_model'):
                if i == 1:  # 두 번째 프레임에서 확인
                    logging.info("✅ ReID 모델 성공적으로 로드됨")
                    logging.info(f"ReID 모델 경로: {config.get('model', 'auto')}")
            elif i == 4:  # 5번째 프레임에서도 확인 안되면 경고
                logging.warning("⚠️ ReID 모델 로딩 상태 확인 불가")

        # FPS 계산
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
        last_time = now

        # 감지 + 트랙 ID 추출
        # result.boxes: bboxes (xyxy), conf, cls, id
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # 트래킹 성능 로깅 (첫 번째 탐지 시)
            if i == 0:
                logging.info(f"첫 번째 객체 탐지됨: {len(boxes)}개 객체")

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
            clss  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
            ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([-1]*len(xyxy), dtype=int)

            # 트래킹 ID 상태 로깅 (10프레임마다)
            if i % 30 == 0 and len(ids) > 0:
                active_ids = [id for id in ids if id != -1]
                logging.info(f"활성 트래킹 ID: {active_ids} (총 {len(active_ids)}개)")

            if HAS_SV:
                detections = sv.Detections(xyxy=xyxy, confidence=confs, class_id=clss, tracker_id=ids)
                labels = [
                    f"ID {tid if tid is not None else -1} | {model.model.names[c]} {conf:.2f}"
                    for (tid, c, conf) in zip(ids, clss, confs)
                ]
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            else:
                # supervision이 없을 경우 기본 OpenCV 라벨링
                for (x1, y1, x2, y2), cid, conf, tid in zip(xyxy, clss, confs, ids):
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"ID {tid} | {model.model.names[cid]} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 좌상단 FPS 표시
        cv2.putText(frame, f"FPS: {fps:.1f} | Model: {args.model} | Tracker: {args.tracker}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 230, 50), 2)

        # writer 준비
        if args.save and writer is None:
            # 가능한 경우, 입력 FPS 추정 (웹캠이면 30 가정)
            guessed_fps = 30
            writer = open_writer(frame, args.save, fps=guessed_fps)

        # 표시/저장
        if args.show:
            cv2.imshow("Better MOT (YOLO + ByteTrack/BoT-SORT)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()