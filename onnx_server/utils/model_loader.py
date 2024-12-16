# utils/model_loader.py
import torch
import torch.utils.model_zoo as model_zoo
from models.super_resolution import SuperResolutionNet  # 모델 정의를 가져옵니다.

def load_pretrained_model(upscale_factor):
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    model = SuperResolutionNet(upscale_factor)  # 모델 초기화
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    
    # 미리 학습된 가중치로 모델을 초기화
    model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))
    
    return model
