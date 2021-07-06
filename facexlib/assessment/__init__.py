import torch

from facexlib.utils import load_file_from_url
from .hypernet import HyperNet


def init_assessment_model(model_name, half=False, device='cuda'):
    if model_name == 'hypernet':
        model = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # load our pre-trained model on the koniq-10k dataset
    model_path = load_file_from_url(
        url='https://cloud.tsinghua.edu.cn/f/95797c79af2d40eba851/?dl=1',
        model_dir='facexlib/weights',
        progress=True,
        file_name='koniq_pretrained.pkl')
    model.load_state_dict((torch.load(model_path, map_location=lambda storage, loc: storage)))
    model.train(False)
    model = model.to(device)
    return model
