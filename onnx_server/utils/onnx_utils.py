import numpy as np
import onnx
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
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]
