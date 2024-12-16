FROM python:3.10-slim

LABEL maintainer "suhan_SuperResolution"

RUN pip install --upgrade pip
RUN mkdir -p /onnx

ENV APP_PATH /onnx

COPY requirements.txt $APP_PATH/
RUN pip install --no-cache-dir -r $APP_PATH/requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY onnx_server.py $APP_PATH/
COPY inference.py $APP_PATH/
COPY models/ $APP_PATH/models/
COPY utils/ $APP_PATH/utils/
COPY super_resolution.onnx $APP_PATH/

EXPOSE 6000

CMD ["python", "/onnx/onnx_server.py"]

