#../../data/

# -*- coding: utf-8 -*-
import json
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# pip install insightface onnxruntime-gpu
# pip uninstall opencv-python opencv-contrib-python
# pip install opencv-python opencv-contrib-python


# uv add insightface


# --- 설정 (Configuration) ---
# 기준이 되는 사람의 이미지 경로
REFERENCE_IMAGE_PATH = '../../data/face.png'
# 비디오 캡처 장치 ID (웹캠: 0, 1, ... 또는 동영상 파일 경로)
CAMERA_ID = 0
# 얼굴 인식을 위한 유사도 임계값
SIMILARITY_THRESHOLD = 0.3
# 로그를 저장할 파일 경로
LOG_FILE_PATH = 'face_analysis_log.jsonl'
# 2D, 3D 랜드마크를 화면에 그릴지 여부
DRAW_LANDMARKS = False

def initialize_face_analysis():
    """InsightFace FaceAnalysis 모델을 초기화합니다."""
    print("InsightFace 모델을 로드합니다...")
    try:
        # GPU 사용을 명시하여 모델 로드 (onnxruntime-gpu 필요)
        app = FaceAnalysis(providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("모델 로드 완료.")
        print(f"사용 모델 정보: {app.models}")
        return app
    except Exception as e:
        print(f"GPU 모델 로드 실패: {e}")
        print("CPU 모델로 다시 시도합니다...")
        try:
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            print("CPU 모델 로드 완료.")
            print(f"사용 모델 정보: {app.models}")
            return app
        except Exception as e_cpu:
            print(f"CPU 모델 로드도 실패했습니다: {e_cpu}")
            exit()

def get_reference_embedding(app, image_path):
    """기준 이미지에서 얼굴 특징 벡터(embedding)를 추출합니다."""
    print(f"기준 이미지({image_path})에서 특징을 추출합니다...")
    try:
        img_ref = cv2.imread(image_path)
        if img_ref is None:
            print(f"오류: 기준 이미지를 찾을 수 없습니다: {image_path}")
            return None

        faces_ref = app.get(img_ref)
        if not faces_ref:
            print(f"오류: 기준 이미지에서 얼굴을 찾지 못했습니다.")
            return None

        # 가장 큰 얼굴을 기준으로 삼음
        main_face = sorted(faces_ref, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
        print("기준 이미지 특징 추출 완료.")
        return main_face.normed_embedding
    except Exception as e:
        print(f"기준 특징 추출 중 오류 발생: {e}")
        return None

def log_data(data):
    """분석된 데이터를 JSONL 형식으로 파일에 기록합니다."""
    try:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"로그 파일 작성 중 오류 발생: {e}")

def process_frame(frame, app, feat_ref, frame_count):
    """
    카메라의 단일 프레임을 처리하고, 얼굴 분석, 로깅, 시각화를 수행합니다.
    """
    faces = app.get(frame)

    if not faces:
        return frame # 얼굴이 없으면 원본 프레임 반환

    all_faces_data = []

    for i, face in enumerate(faces):
        # 1. 특징 벡터(embedding) 추출 및 유사도 계산
        feat_current = face.normed_embedding
        sim = np.dot(feat_ref, feat_current)
        is_same_person = bool(sim > SIMILARITY_THRESHOLD)

        # 2. 로깅할 데이터 생성
        current_face_data = {
            'timestamp': datetime.now().isoformat(),
            'frame': frame_count,
            'face_id': i,
            'is_same_person': is_same_person,
            'similarity_score': float(sim),
            'detection_score': float(face.det_score),
            'bounding_box': face.bbox.astype(int).tolist(),
            'gender': 'Male' if face.gender == 1 else 'Female',
            'age': face.age,
            'landmark_2d_106': face.landmark_2d_106.astype(int).tolist(),
            'landmark_3d_68': face.landmark_3d_68.astype(int).tolist(),
            # 'embedding': feat_current.tolist() # 임베딩 벡터는 용량이 크므로 필요시 주석 해제
        }
        all_faces_data.append(current_face_data)

        # 3. 결과 시각화
        bbox = face.bbox.astype(int)
        color = (0, 255, 0) if is_same_person else (0, 0, 255)

        # Bounding Box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # 정보 텍스트
        label_text = f"{'SAME' if is_same_person else 'DIFF'}: {sim:.2f}"
        info_text = f"Age: {face.age}, Gender: {'M' if face.gender==1 else 'F'}"

        cv2.putText(frame, label_text, (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, info_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 랜드마크 (설정이 True일 경우)
        if DRAW_LANDMARKS:
            # 2D 106 랜드마크
            for p in face.landmark_2d_106:
                cv2.circle(frame, (int(p[0]), int(p[1])), 1, (255, 255, 0), -1)
            # 3D 68 랜드마크
            # for p in face.landmark_3d_68:
            #     cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 255, 255), -1)

    # 생성된 모든 얼굴 데이터를 파일에 로깅
    log_data(all_faces_data)

    return frame

def main():
    """메인 실행 함수"""
    app = initialize_face_analysis()
    feat_ref = get_reference_embedding(app, REFERENCE_IMAGE_PATH)

    if feat_ref is None:
        return

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if not cap.isOpened():
        print(f"오류: 카메라({CAMERA_ID})를 열 수 없습니다.")
        return

    tm = cv2.TickMeter()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 프레임을 읽어오지 못했습니다. 스트림 종료.")
            break

        tm.start()

        # 실제 처리 로직 호출
        processed_frame = process_frame(frame, app, feat_ref, frame_count)

        tm.stop()

        # 성능 정보 표시
        fps = tm.getFPS()
        proc_time = tm.getTimeMilli()

        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Time: {proc_time:.2f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('InsightFace Analysis', processed_frame)

        frame_count += 1
        tm.reset()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()