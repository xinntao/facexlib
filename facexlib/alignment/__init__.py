import torch

from .awing_arch import FAN
from .convert_98_to_68_landmarks import landmark_98_to_68

__all__ = ['FAN', 'landmark_98_to_68']


def init_alignment_model(model_name, half=False, device='cuda'):
    if model_name == 'awing_fan':
        model_path = 'facexlib/alignment/weights/WFLW_4HG.pth'
        model = FAN(num_modules=4, num_landmarks=98)
        model.load_state_dict(torch.load(model_path)['state_dict'], strict=True)
        model.eval()
        model = model.to(device)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')
    return model
