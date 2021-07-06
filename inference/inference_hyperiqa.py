import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torchvision
from PIL import Image

from facexlib.assessment import init_assessment_model
from facexlib.assessment import targetnet as targetnet
from facexlib.detection import init_detection_model


def main(args):

    # initialize model
    det_net = init_detection_model(args.dect_model_name, half=False)
    hyper_net = init_assessment_model(args.assess_model_name, half=False)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 384)),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img_list = sorted(glob.glob(os.path.join(args.img_folder, '*.png')))
    scores = []
    for img_path in img_list:
        img = cv2.imread(img_path)[:, :, [2, 1, 0]]
        img_name = os.path.basename(img_path)
        basename, _ = os.path.splitext(img_name)
        with torch.no_grad():
            bboxes = det_net.detect_faces(img, 0.97)
            box = list(map(int, bboxes[0]))
            pred_scores = []
            for i in range(10):
                detect_face = img[box[1]:box[3], box[0]:box[2], :]
                detect_face = Image.fromarray(detect_face)

                detect_face = transforms(detect_face)
                detect_face = torch.tensor(detect_face.cuda()).unsqueeze(0)
                # params contain the network weights conveyed to target network
                net_params = hyper_net(detect_face)

                # Build target network
                target_net = targetnet.TargetNet(net_params).cuda()
                for param in target_net.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = target_net(net_params['target_in_vec'])
                pred_scores.append(float(pred.item()))
            score = np.mean(pred_scores)
            # quality score ranges from 0-100, a higher score indicates a better quality
            print(f'{basename}' ': %.2f' % score)
            scores.append(score)
    # std = np.std(scores, ddof=1)
    # print('The standard derivation is: %.2f' % std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_folder',
        type=str,
        default=  # noqa E251
        '/cfs/cfs-3b315583b/liangbinxie/datasets/VFHQ4K/GT_Train512/Clip+2g9ZQ7rTHM4+P0+C5+F5007-5998'  # noqa E501
    )
    parser.add_argument('--dect_model_name', type=str, default='retinaface_resnet50')
    parser.add_argument('--assess_model_name', type=str, default='hypernet')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)
