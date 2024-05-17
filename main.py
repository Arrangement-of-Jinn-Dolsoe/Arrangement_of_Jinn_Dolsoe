import cv2
from ultralytics import YOLO

# 모델 불러오기
model = YOLO("yolov8n.pt")  # COCO 사전 훈련된 모델 사용

# 이미지 불러오기
image = cv2.imread('20240502_142118.jpg')

# 객체 탐지 및 정렬
results = model(image)

# 탐지된 객체 정보 출력
for result in results[0].xyxy:
    # 신뢰도가 0.5 이상인 객체만 출력
    if result.conf > 0.5:
        label = result.name  # 객체 클래스 이름
        x1, y1, x2, y2 = result.xyxy  # 경계 상자 좌표
        confidence = result.conf  # 신뢰도

        # 경계 상자 그리기
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # 객체 클래스 및 신뢰도 텍스트 출력
        cv2.putText(image, f"{label}: {confidence:.2f}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 정렬된 객체가 포함된 이미지 출력
cv2.imshow('Object Detection', image)
cv2.waitKey(0)