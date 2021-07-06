import torch

from facexlib.utils import load_file_from_url
from .hyperiqa_net import HyperNet


def init_assessment_model(model_name, half=False, device='cuda'):
    if model_name == 'hypernet':
        model = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/hyperIQA.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # load our pre-trained model on the koniq-10k dataset
    model_path = load_file_from_url(url=model_url, model_dir='facexlib/weights', progress=True, file_name=None)
    model.load_state_dict((torch.load(model_path, map_location=lambda storage, loc: storage)))
    model = model.eval()
    model = model.to(device)
    return model
