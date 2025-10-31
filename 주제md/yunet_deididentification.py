import os
import argparse
import cv2 as cv
import numpy as np

from yunet_ort import YuNet

def create_circular_mask(h, w, center, radius):
    """원형 마스크 생성"""
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def create_ellipse_mask(h, w, center, axes, angle=0):
    """타원형 마스크 생성"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    return mask.astype(bool)

def create_polygon_mask(h, w, points):
    """다각형 마스크 생성"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask.astype(bool)

def apply_blur_with_mask(image, bbox, mask_type='circle', blur_strength=99, blur_method='gaussian'):
    """
    마스크 형태로 강력한 블러 적용
    mask_type: 'circle', 'ellipse', 'rectangle', 'face_shape'
    blur_method: 'gaussian', 'median', 'bilateral', 'pixelate', 'black'
    """
    x, y, w, h = bbox
    
    # ROI 영역 추출
    roi = image[y:y+h, x:x+w].copy()
    
    # 마스크 생성
    if mask_type == 'circle':
        center = (w // 2, h // 2)
        radius = min(w, h) // 2
        mask = create_circular_mask(h, w, center, radius)
        
    elif mask_type == 'ellipse':
        center = (w // 2, h // 2)
        axes = (w // 2, h // 2)
        mask = create_ellipse_mask(h, w, center, axes)
        
    elif mask_type == 'face_shape':
        center = (w // 2, h // 2)
        axes = (int(w * 0.45), int(h * 0.5))
        mask = create_ellipse_mask(h, w, center, axes)
        
    elif mask_type == 'polygon':
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 2
        points = []
        for i in range(6):
            angle = i * 60 * np.pi / 180
            px = int(center_x + radius * np.cos(angle))
            py = int(center_y + radius * np.sin(angle))
            points.append([px, py])
        mask = create_polygon_mask(h, w, points)
        
    else:  # rectangle
        mask = np.ones((h, w), dtype=bool)
    
    # 다양한 블러 방법 적용
    if blur_method == 'gaussian':
        # 가우시안 블러 (15번 반복 적용하여 매우 강력하게)
        blurred_roi = roi.copy()
        for _ in range(15):  # 15번 반복 적용
            blurred_roi = cv.GaussianBlur(blurred_roi, (blur_strength, blur_strength), 0)
            
    elif blur_method == 'median':
        # 미디언 블러
        blurred_roi = cv.medianBlur(roi, blur_strength)
        
    elif blur_method == 'bilateral':
        # 양방향 필터 (여러 번 적용)
        blurred_roi = roi.copy()
        for _ in range(10):
            blurred_roi = cv.bilateralFilter(blurred_roi, 9, 75, 75)
            
    elif blur_method == 'pixelate':
        # 픽셀화 (모자이크)
        pixel_size = max(15, min(w, h) // 10)
        temp = cv.resize(roi, (pixel_size, pixel_size), interpolation=cv.INTER_LINEAR)
        blurred_roi = cv.resize(temp, (w, h), interpolation=cv.INTER_NEAREST)
        
    elif blur_method == 'black':
        # 검은색으로 완전 가림
        blurred_roi = np.zeros_like(roi)
        
    else:  # 기본값
        blurred_roi = cv.GaussianBlur(roi, (blur_strength, blur_strength), 0)
    
    # 마스크를 3채널로 확장
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    
    # 마스크 부드럽게 (경계 자연스럽게)
    mask_float = mask.astype(np.float32)
    mask_blur = cv.GaussianBlur(mask_float, (21, 21), 0)
    mask_blur_3ch = np.stack([mask_blur, mask_blur, mask_blur], axis=2)
    
    # 블러 적용된 부분과 원본을 부드럽게 블렌딩
    result_roi = (blurred_roi * mask_blur_3ch + roi * (1 - mask_blur_3ch)).astype(np.uint8)
    
    # 원본 이미지에 적용
    output = image.copy()
    output[y:y+h, x:x+w] = result_roi
    
    return output

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), 
              show_bbox=True, mask_type='face_shape', apply_blur=False, 
              blur_strength=99, blur_method='gaussian'):
    """
    결과 시각화 및 블러 처리
    """
    output = image.copy()

    for det in results:
        bbox = det[0:4].astype(np.int32)
        conf = det[-1]
        
        # 블러 적용
        if apply_blur:
            output = apply_blur_with_mask(output, bbox, mask_type=mask_type, 
                                         blur_strength=blur_strength, 
                                         blur_method=blur_method)
        
        # 바운딩 박스 표시 (옵션)
        if show_bbox:
            cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
            cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), 
                      cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YuNet Face Detection with Custom ROI Masks')
    parser.add_argument('--input', '-i', type=str, default='../data/march.jpg',
                        help='입력 이미지 경로')
    parser.add_argument('--model', '-m', type=str, default='../data/face_detection_yunet_2023mar.onnx',
                        help='모델 파일 경로')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='출력 폴더 경로')
    parser.add_argument('--conf_threshold', type=float, default=0.6,
                        help='신뢰도 임계값 (기본값: 0.6)')
    parser.add_argument('--nms_threshold', type=float, default=0.3,
                        help='NMS 임계값 (기본값: 0.3)')
    parser.add_argument('--top_k', type=int, default=5000,
                        help='Top K (기본값: 5000)')
    parser.add_argument('--max_size', type=int, default=640,
                        help='최대 이미지 크기 (기본값: 640)')
    parser.add_argument('--mask_type', type=str, default='face_shape',
                        choices=['circle', 'ellipse', 'rectangle', 'face_shape', 'polygon'],
                        help='마스크 형태 선택')
    parser.add_argument('--blur', action='store_true',
                        help='얼굴 영역에 블러 적용')
    parser.add_argument('--blur_strength', type=int, default=99,
                        help='블러 강도 (홀수, 기본값: 99, 최대 199 권장)')
    parser.add_argument('--blur_method', type=str, default='gaussian',
                        choices=['gaussian', 'median', 'bilateral', 'pixelate', 'black'],
                        help='블러 방법 선택')
    parser.add_argument('--hide_bbox', action='store_true',
                        help='바운딩 박스 숨기기')
    args = parser.parse_args()

    # 블러 강도를 홀수로 변환
    if args.blur_method in ['gaussian', 'median'] and args.blur_strength % 2 == 0:
        args.blur_strength += 1

    # output 폴더 생성
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"'{args.output}' 폴더가 생성되었습니다.")

    # 이미지 읽기
    image = cv.imread(args.input)
    
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {args.input}")
        exit()
    
    orig_h, orig_w, _ = image.shape
    print(f"원본 이미지 크기: {orig_w}x{orig_h}")
    
    # 이미지 크기가 max_size를 초과하면 리사이즈
    if max(orig_w, orig_h) > args.max_size:
        scale = args.max_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = cv.resize(image, (new_w, new_h))
        h, w, _ = image.shape
        print(f"리사이즈된 이미지 크기: {w}x{h}")
    else:
        h, w = orig_h, orig_w

    # YuNet 모델 초기화
    model = YuNet(modelPath=args.model,
                  inputSize=[w, h],
                  confThreshold=args.conf_threshold,
                  nmsThreshold=args.nms_threshold,
                  topK=args.top_k)

    # 얼굴 검출
    results = model.infer(image)
    
    # 결과 출력
    print(f'{results.shape[0]} 개의 얼굴이 검출되었습니다.')
    print(f'마스크 타입: {args.mask_type}')
    print(f'블러 방법: {args.blur_method}')
    print(f'블러 강도: {args.blur_strength}')
    
    # 각 얼굴의 신뢰도 출력
    for idx, det in enumerate(results):
        conf = det[-1]
        print(f'  얼굴 {idx+1}: 신뢰도 {conf:.4f}')

    # 결과 시각화
    image = visualize(image, results, 
                     show_bbox=not args.hide_bbox,
                     mask_type=args.mask_type,
                     apply_blur=args.blur,
                     blur_strength=args.blur_strength,
                     blur_method=args.blur_method)
    
    # 결과 저장
    input_filename = os.path.basename(args.input)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_{args.mask_type}_{args.blur_method}{args.blur_strength if args.blur else ''}{ext}"
    output_path = os.path.join(args.output, output_filename)
    cv.imwrite(output_path, image)
    print(f"결과가 저장되었습니다: {output_path}")
    
    cv.imshow('YuNet Face Detection', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
