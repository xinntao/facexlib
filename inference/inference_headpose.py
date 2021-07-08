import argparse
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

from facexlib.detection import init_detection_model
from facexlib.headpose import init_headpose_model
from facexlib.utils.misc import img2tensor
from facexlib.visualization import visualize_headpose


def main(args):
    # initialize model
    det_net = init_detection_model(args.detection_model_name, half=args.half)
    headpose_net = init_headpose_model(args.headpose_model_name, half=args.half)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        bboxes = det_net.detect_faces(img, 0.97)
        # x0, y0, x1, y1, confidence_score, five points (x, y)
        bbox = list(map(int, bboxes[0]))
        # crop face region
        thld = 10
        h, w, _ = img.shape
        top = max(bbox[1] - thld, 0)
        bottom = min(bbox[3] + thld, h)
        left = max(bbox[0] - thld, 0)
        right = min(bbox[2] + thld, w)

        det_face = img[top:bottom, left:right, :].astype(np.float32) / 255.

        # resize
        det_face = cv2.resize(det_face, (224, 224), interpolation=cv2.INTER_LINEAR)
        det_face = img2tensor(np.copy(det_face), bgr2rgb=False)

        # normalize
        normalize(det_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
        det_face = det_face.unsqueeze(0).cuda()

        yaw, pitch, roll = headpose_net(det_face)
        visualize_headpose(img, yaw, pitch, roll, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--img_path', type=str, default='assets/test.jpg')
    parser.add_argument('--save_path', type=str, default='assets/test_headpose.png')
    parser.add_argument('--detection_model_name', type=str, default='retinaface_resnet50')
    parser.add_argument('--headpose_model_name', type=str, default='hopenet')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)
