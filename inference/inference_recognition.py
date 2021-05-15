import argparse
import glob
import math
import numpy as np
import os
import torch

from facexlib.recognition import ResNetArcFace, cosin_metric, load_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1', type=str)
    parser.add_argument('--folder2', type=str)
    parser.add_argument('--model_path', type=str, default='facexlib/recognition/weights/arcface_resnet18.pth')

    args = parser.parse_args()

    img_list1 = sorted(glob.glob(os.path.join(args.folder1, '*')))
    img_list2 = sorted(glob.glob(os.path.join(args.folder2, '*')))
    print(img_list1, img_list2)
    model = ResNetArcFace(block='IRBlock', layers=(2, 2, 2, 2), use_se=False)
    model.load_state_dict(torch.load(args.model_path))
    model.to(torch.device('cuda'))
    model.eval()

    dist_list = []
    identical_count = 0
    for idx, (img_path1, img_path2) in enumerate(zip(img_list1, img_list2)):
        basename = os.path.splitext(os.path.basename(img_path1))[0]
        img1 = load_image(img_path1)
        img2 = load_image(img_path2)

        data = torch.stack([img1, img2], dim=0)
        data = data.to(torch.device('cuda'))
        output = model(data)
        print(output.size())
        output = output.data.cpu().numpy()
        dist = cosin_metric(output[0], output[1])
        dist = np.arccos(dist) / math.pi * 180
        print(f'{idx} - {dist} o : {basename}')
        if dist < 1:
            print(f'{basename} is almost identical to original.')
            identical_count += 1
        else:
            dist_list.append(dist)

    print(f'Result dist: {sum(dist_list) / len(dist_list):.6f}')
    print(f'identical count: {identical_count}')
