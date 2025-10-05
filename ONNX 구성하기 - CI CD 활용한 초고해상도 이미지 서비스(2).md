# ONNX 구성하기 - CI/CD 활용한 초고해상도 이미지 서비스(2)

---

# 1. ONNX란?

Open Neural Network Exchang, 딥러닝 및 머신러닝 모델을 프레임워크 간에 교환하고 배포할 수 있도록 설계된 표준 모델 형식. 최적화 및 효율적인 배포를 지원.

### 양자화된 네트워크 표현 방식

- ONNX은 모델의 양자화를 지원하여 모델의 크기를 줄여 계산 속도를 높인다 → FP32를 INT8로 변환
- ONNX 모델은 리소스가 제한적인 디바이스에서도 실행 가능하게 한다.

### Single Protobuf

- 모델 구조와 가중치가 Protobuf 형식으로 직렬화되어 `.onnx`에 저장
- Protobuf는 구조화된 데이터를 직렬화하는 형식, 크기가 작고 속도 빠름
    - 데이터를 이진으로 직렬화 하므로 크기가 작고 처리 속도 빠름
    - 모델 사용 시 Protobuf에서 데이터 역질렬화 하여 ONNX Runtime이 그래프와 가중치를 로드 → 추론 수행

### MLIR

- 최적화와 하드웨어별 코드 생성
- 하드웨어 및 소프트웨어 환경에 맞는 중간표현(IR)을 제공하여 모델 변환과 실행 최적화를 간소화
- CPU, GPU등 다양한 하드웨어에서 ONNX 모델 효율적 실행 가능

### ONNX Runtime

모델 실행 최적화 하기 위해 사용 

1. 그래프 최적화
    - 모델의 계산 그래프를 단순화하거나 병렬 처리를 변경하여 실행 속도 향상
    - 불필요한 노드 제거, 연산 합병 통해 성능 개선
2. Model Partitioning
    - 모델을 여러 파티션으로 나눠 하드웨어별 병렬 처리 지원
    - CPU, GPU를 혼합하여 사용, 분산 처리 통해 성능 극대화

### ONNX 구성요소

구조화된 데이터들이며, 이를 직렬화를 진행한다. 

1. Tensor
    - 가중치 등
2. Operator
    - Conv, ReLU 등 모델 구조

### ONNX 한계

1. 정확도 저하
    - 프레임워크 간 모델 변환 과정에 연산 방식 차이로 결과가 다를 수 있다(Pytorch에서 Tensorflow) → tensorRT사용
    - 양자화 및 최적화로 인해 정밀도가 손실된다.
2. 학습 성능 저하
    - ONNX는 추론에 최적화 되어 있어, 학습 과정에는 적합하지 않다.
3. 사전 훈련된 가중치 사용 불가
    - ONNX 표준에서 지원하지 않는 연산자는 변환 불가
4. 크기 및 속도 저하
    - 복잡한 모델일수록 변환 후 파일 크기가 커지고, 일부 최적화가 누락되면 속도 저하
    - ONNX Runtime 없이 실행 시 성능 떨어짐

참고 : https://medium.com/@enerzai/onnx-%EB%84%88-%EB%88%84%EA%B5%AC%EC%95%BC-who-are-you-5c1435b997e2

# 2. 구현

구조 

```python
├── Dockerfile
├── README.md
├── inference.py
├── models
│   ├── __init__.py
│   └── super_resolution.py
├── onnx_server.py
├── requirements.txt
├── super_resolution.onnx
└── utils
    ├── __init__.py
    ├── image_utils.py
    ├── model_loader.py
    └── onnx_utils.py
```

## 1. Model

```python
import torch.nn as nn
import torch.nn.init as init

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()
				
				# 입력 텐서 자체를 수정하여 메모리 사용 줄임 
        self.relu = nn.ReLU(inplace=inplace)
        # 입력 채널1(Y성분만), 출력 채널64, 커널 5, 스트라이드 1, 패딩 2 
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x
		
		# 가중치 초기화 직교 초기화 방법 사용 
    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

```

## 2. Utill

### 2.1 image_utils

```python
from PIL import Image
import numpy as np

# Y(밝기 성분)에 사람이 예민하여 이 성분 만을 활용하여 고해상도의 디테일을 복원
# 색상은 그대로 둔다. 
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size, Image.BICUBIC)
   
    return img

def postprocess_output(img_y, img_cb, img_cr):
    img_out_y = Image.fromarray(np.uint8((img_y * 255.0).clip(0, 255)[0]), mode='L')
    return Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC)
        ]
    ).convert("RGB")

```

### 2.2 model_loader

학습된 모델 파라미터 로드하여 적용하여 사용. 

```python
# utils/model_loader.py
import torch
import torch.utils.model_zoo as model_zoo
from models.super_resolution import SuperResolutionNet  # 모델 정의를 가져옵니다.
from torch.quantization import quantize_dynamic

def load_pretrained_model(upscale_factor):
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    model = SuperResolutionNet(upscale_factor)  # 모델 초기화
    map_location = lambda storage, loc: storage
    # GPU,CPU 사용 여부에 따라 로드 방법 설정 
    if torch.cuda.is_available():
        map_location = None
    
    # 미리 학습된 가중치로 모델을 초기화
    model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return quantized_model

```

### 2.3 onnx_utils

onnxruntime을 통해 예측, 추론을 수행한다. 

```python
import numpy as np
import onnx
from onnxruntime import GraphOptimizationLevel
import onnxruntime

def export_to_onnx(model, input_tensor, file_path, opset_version=10):
    import torch.onnx
    torch.onnx.export(
        model, 
        input_tensor, 
        file_path, 
        export_params=True, 
        opset_version=opset_version, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load_onnx_model(file_path):
    return onnx.load(file_path)

def check_onnx_model(onnx_model):
    onnx.checker.check_model(onnx_model)

def infer_onnx_model(onnx_model_path, input_tensor):
    ort_session_options = onnxruntime.SessionOptions()
    # 최적화 수준 설정 (더 빠른 실행을 위해)
    ort_session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model_path, sess_options=ort_session_options, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

```

`torch.onnx.exprot()` : Pytorch모델을 ONNX 형식으로 내보낼 때 사용

`input_tensor` : 모델에 대한 샘플 입력 텐서, ONNX모델 생성하기 위해 사용

`file_path` : 모델 저장 경로

`export_params=True`: 모델의 학습된 파라미터(가중치)도 함께 저장

`opset_version=opset_version`: 사용하는 ONNX의 opset 버전을 설정

`do_constant_folding=True`: 상수 연산을 미리 계산하여 그래프를 최적화

`input_names=['input']`: 입력 텐서의 이름

`output_names=['output']`: 출력 텐서의 이름

`dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}`: 배치 크기가 동적으로 변경될 수 있음을 명시하여, 입력 및 출력 텐서의 첫 번째 차원(batch_size)을 동적으로 처리할 수 있도록 .

## 3. Inference

```python
import torch
# from moonnx_modeldels.super_resolution import SuperResolutionNet
from utils.onnx_utils import infer_onnx_model
# from utils.onnx_utils import  export_to_onnx
from utils.image_utils import preprocess_image, postprocess_output
from utils.model_loader import load_pretrained_model
import torchvision.transforms as transforms
from PIL import Image
# if __name__ == "__main__":

def onnx_model(model, image):
    # 모델 초기화
    upscale_factor = 3
    torch_model = load_pretrained_model(upscale_factor)
    torch_model.eval()

    # ONNX 변환 및 추론
    input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
    export_to_onnx(torch_model, input_tensor, model)
    print("ONNX 모델이 성공적으로 변환되었습니다.")

    # 이미지 처리
    img = preprocess_image(image)
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    # ONNX 추론
    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y).unsqueeze(0)
    ort_outs = infer_onnx_model(model, img_y)

    # 후처리
    final_img = postprocess_output(ort_outs[0], img_cb, img_cr)
    # final_img.save("cat_result.jpg")
    print("결과 이미지가 저장되었습니다.")

    return final_img

```

## 4. onnx_server

Flask를 활용하여 Restful API 방식으로 이미지를 송수신 

Restful API

- HTTP 프로토콜을 사용하여 클라이언트와 서버 간 데이터 주고 받는 방법
- 자원을 URI로 표현 [`https://api.example.com/users`](https://api.example.com/users) 하여 GET, POST 가능

```python
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
		
		# RestAPI로 전달하기 위해 이미지를 JPEG방식으로 압축하고 바이너리로 바꿔 전달 
    result_img.save(img_byte_arr, format='JPEG', quality=50)  # quality 85는 압축률을 조절합니다.
    img_byte_arr.seek(0)  # BytesIO 포인터를 처음으로 이동

    # 결과 이미지를 byte로 반환
    return send_file(img_byte_arr, mimetype='image/jpeg', as_attachment=True, download_name='result_image.jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)

```

# 5. 메모리, 속도 측정

```python
import onnxruntime
import numpy as np
import time
import psutil
from onnxruntime import GraphOptimizationLevel
import os
import torch
import matplotlib.pyplot as plt

# 텐서를 numpy 배열로 변환하는 함수
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 메모리 사용량을 측정하는 함수
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB 단위로 반환

# 추론 시간 측정 함수
def measure_inference_time_origin(onnx_model_path, input_tensor):
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}

    start_time = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    end_time = time.time()

    inference_time = end_time - start_time
    return inference_time

# 메모리 사용량 측정 함수
def measure_inference_memory_origin(onnx_model_path, input_tensor):
    before_memory = get_memory_usage()

    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)

    after_memory = get_memory_usage()
    memory_used = after_memory - before_memory
    return memory_used

def measure_inference_time(onnx_model_path, input_tensor):
    ort_session_options = onnxruntime.SessionOptions()

    # 최적화 수준 설정 (더 빠른 실행을 위해)
    ort_session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = onnxruntime.InferenceSession(onnx_model_path, sess_options=ort_session_options, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}

    start_time = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    end_time = time.time()

    inference_time = end_time - start_time
    return inference_time

# 메모리 사용량 측정 함수
def measure_inference_memory(onnx_model_path, input_tensor):
    before_memory = get_memory_usage()

    ort_session_options = onnxruntime.SessionOptions()

    # 최적화 수준 설정 (더 빠른 실행을 위해)
    ort_session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = onnxruntime.InferenceSession(onnx_model_path, sess_options=ort_session_options, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)

    after_memory = get_memory_usage()
    memory_used = after_memory - before_memory
    return memory_used

# 입력 텐서 예시
input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)

# 모델 경로
onnx_model_path1 = "super_resolution_origin.onnx"
onnx_model_path2 = "super_resolution.onnx"

# 모델 1의 추론 시간과 메모리 사용량
inference_time1 = measure_inference_time_origin(onnx_model_path1, input_tensor)
memory_usage1 = measure_inference_memory_origin(onnx_model_path1, input_tensor)

# 모델 2의 추론 시간과 메모리 사용량
inference_time2 = measure_inference_time(onnx_model_path2, input_tensor)
memory_usage2 = measure_inference_memory(onnx_model_path2, input_tensor)

# 결과 출력
# 그래프를 위한 데이터 준비
models = ['Model_origin', 'Model_optim']
inference_times = [inference_time1, inference_time2]
memory_usages = [memory_usage1, memory_usage2]

# 추론 시간 그래프
plt.figure(figsize=(12, 6))

# Subplot 1: Inference Time
plt.subplot(1, 2, 1)
plt.bar(models, inference_times, color=['blue', 'orange'])
plt.title('Inference Time Comparison')
plt.ylabel('Time (second)')
plt.xlabel('Model')

# Subplot 2: Memory Usage
plt.subplot(1, 2, 2)
plt.bar(models, memory_usages, color=['blue', 'orange'])
plt.title('Memory Usage Comparison')
plt.ylabel('Memory Usage (MB)')
plt.xlabel('Model')

# 그래프 표시
plt.tight_layout()
plt.show()

# 결과 출력
print(f"Model 1 - Inference Time: {inference_time1:.4f} seconds, Memory Usage: {memory_usage1:.2f} MB")
print(f"Model 2 - Inference Time: {inference_time2:.4f} seconds, Memory Usage: {memory_usage2:.2f} MB")
```

참고 : [https://tutorials.pytorch.kr/advanced/super_resolution_with_onnxruntime.html](https://tutorials.pytorch.kr/advanced/super_resolution_with_onnxruntime.html)
