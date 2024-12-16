from flask import Flask, request, render_template
import os
import requests

app = Flask(__name__)

# 폴더 설정
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# ONNX 서버 URL
ONNX_SERVER_URL = "http://onnx:6000/inference"

# 디렉토리 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file uploaded.", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected.", 400

    # 업로드된 파일 저장
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # ONNX 서버로 요청
    with open(file_path, 'rb') as f:
        response = requests.post(ONNX_SERVER_URL, files={'image': f})
    
    if response.status_code != 200:
        return "ONNX inference failed.", 500

    result_image = response.content  # 응답에서 이미지를 byte로 받음

    # 결과 이미지를 파일로 저장
    result_image_path = os.path.join(app.config['RESULT_FOLDER'], f'result_{file.filename}')
    with open(result_image_path, 'wb') as result_file:
        result_file.write(result_image)

    # 결과 처리 후 반환 (결과 이미지 경로 전달)
    return render_template('index.html', uploaded_image=file.filename, result_image=os.path.basename(result_image_path))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

