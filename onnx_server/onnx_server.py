from flask import Flask, request, send_file
import onnxruntime as ort
import numpy as np
from PIL import Image
from inference import onnx_model
from io import BytesIO

app = Flask(__name__)

# ONNX 모델 경로
onnx_model_path = 'onnx/super_resolution.onnx'

@app.route('/inference', methods=['POST'])
def inference():
    if 'image' not in request.files:
        return {"error": "No file uploaded"}, 400

    # 이미지 가져오기
    file = request.files['image']
    image = Image.open(BytesIO(file.read()))
    
    # ONNX 추론
    result_img = onnx_model(onnx_model_path, image)

    img_byte_arr = BytesIO()

    result_img.save(img_byte_arr, format='JPEG', quality=50)  # quality 85는 압축률을 조절합니다.
    img_byte_arr.seek(0)  # BytesIO 포인터를 처음으로 이동

    # 결과 이미지를 byte로 반환
    return send_file(img_byte_arr, mimetype='image/jpeg', as_attachment=True, download_name='result_image.jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)

