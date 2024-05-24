from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from yolov8_seg import process_image
import cv2

app = Flask(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 업로드 폴더가 없으면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # YOLO 모델로 이미지 처리
            boxes, result_image = process_image(file_path)
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
            cv2.imwrite(result_image_path, result_image)
            return render_template('result.html', boxes=boxes, result_image=f'result_{filename}')
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
