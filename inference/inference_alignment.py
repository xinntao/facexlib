import argparse
import cv2
import torch

from facexlib.alignment import init_alignment_model, landmark_98_to_68
from facexlib.visualization import visualize_alignment


def main(args):
    # initialize model
    align_net = init_alignment_model(args.model_name)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        landmarks = align_net.get_landmarks(img)
        if args.to68:
            landmarks = landmark_98_to_68(landmarks)
        visualize_alignment(img, [landmarks], args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument('--save_path', type=str, default='test_alignment.png')
    parser.add_argument('--model_name', type=str, default='awing_fan')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--to68', action='store_true')
    args = parser.parse_args()

    main(args)
