import argparse
import cv2
import numpy as np
import os
import torch
import torchvision
from PIL import Image

from facexlib.assessment import init_assessment_model
from facexlib.detection import init_detection_model


def main(args):
    """Scripts about evaluating face quality.
        Two steps:
        1) detect the face region and crop the face
        2) evaluate the face quality by hyperIQA
    """
    # initialize model
    det_net = init_detection_model(args.detection_model_name, half=False)
    assess_net = init_assessment_model(args.assess_model_name, half=False)

    # specified face transformation in original hyperIQA
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 384)),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img = cv2.imread(args.img_path)
    img_name = os.path.basename(args.img_path)
    basename, _ = os.path.splitext(img_name)
    with torch.no_grad():
        bboxes = det_net.detect_faces(img, 0.97)
        box = list(map(int, bboxes[0]))
        pred_scores = []
        # BRG -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(10):
            detect_face = img[box[1]:box[3], box[0]:box[2], :]
            detect_face = Image.fromarray(detect_face)

            detect_face = transforms(detect_face)
            detect_face = torch.tensor(detect_face.cuda()).unsqueeze(0)

            pred = assess_net(detect_face)
            pred_scores.append(float(pred.item()))
        score = np.mean(pred_scores)
        # quality score ranges from 0-100, a higher score indicates a better quality
        print(f'{basename} {score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument('--detection_model_name', type=str, default='retinaface_resnet50')
    parser.add_argument('--assess_model_name', type=str, default='hypernet')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)
