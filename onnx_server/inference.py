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
    #upscale_factor = 3
    #torch_model = load_pretrained_model(upscale_factor)
    #torch_model.eval()

    # ONNX 변환 및 추론
    # input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
    # export_to_onnx(torch_model, input_tensor, model)
    # print("ONNX 모델이 성공적으로 변환되었습니다.")

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
