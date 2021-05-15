import torch

from .arcface_arch import Backbone


def init_recognition_model(model_name, half=False):
    if model_name == 'arcface':
        model_path = 'facexlib/recognition/weights/arcface_ir_se50.pth'
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').to('cuda').eval()
        model.load_state_dict(torch.load(model_path))
    return model
