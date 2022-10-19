import argparse
from glob import glob
from time import time
from os import path, makedirs
import cv2
import torch
import numpy as np

from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_detection


def main(args):
    # initialize model
    det_net = init_detection_model(args.model_name, half=args.half)

    tic = time()
    imgs = []
    for img_path in glob(f'{path.join(args.img_dir, "")}/*.png'):
        imgs.append(torch.tensor(cv2.imread(img_path)))
    with torch.no_grad():
        bboxes_list = det_net.batched_detect_faces(torch.stack(imgs), 0.97)
        # x0, y0, x1, y1, confidence_score, five points (x, y)
        bboxes_list = [np.concatenate(b, axis=-1) for b in zip(*bboxes_list)]
        print(bboxes_list)
        print(f'Took {time() - tic:.2f}s to detect {len(imgs)} images.')
        makedirs(args.save_dir, exist_ok=True)
        for i, (img, bboxes) in enumerate(zip(imgs, bboxes_list)):
            visualize_detection(img, bboxes, path.join(args.save_dir, f'{i}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='assets/batch_test')
    parser.add_argument('--save_dir', type=str, default='test_detection')
    parser.add_argument(
        '--model_name', type=str, default='retinaface_mobile0.25', help='retinaface_resnet50 | retinaface_mobile0.25')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)
