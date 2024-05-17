import cv2

# 이미지 읽기
img = cv2.imread('20240502_142118.jpg', 0)

# 이미지 이진화
_, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

# 윤곽선 찾기
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 윤곽선 그리기
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 사진에 있는 선반과 물체들을 감지하는 기능
# ㄴ 선반을 사직으로 찍게 할건지 선반의 종류를 사용자가 선택하게 할건지
# 감지된 물체중에 사격형이 아닌 물체를 사각화 시켜주는 기능

# 사각화 된 물체를 종류에 맞게 분리해주는 기능

# 분리된 물체를 선반에 맞춰 정리해주는기능

# 정리된 모습의 물체화 선반을 단말기에 출력하는 기능
