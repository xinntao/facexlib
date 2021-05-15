from .retinaface import RetinaFace


def init_detection_model(model_name, half=False):
    if model_name == 'retinaface_resnet50':
        det_model = RetinaFace(network_name='resnet50', half=half)
    elif model_name == 'retinaface_mobile0.25':
        det_model = RetinaFace(network_name='mobile0.25', half=half)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')
    return det_model
