import argparse
import cv2
import torch

from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_detection


def main(args):
    # initialize model
    det_net = init_detection_model(args.model_name, half=args.half)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        bboxes = det_net.detect_faces(img, 0.97)
        # x0, y0, x1, y1, confidence_score, five points (x, y)
        print(bboxes)
        visualize_detection(img, bboxes, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test.jpg')
    parser.add_argument('--save_path', type=str, default='test_detection.png')
    parser.add_argument(
        '--model_name', type=str, default='retinaface_resnet50', help='retinaface_resnet50 | retinaface_mobile0.25')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)
