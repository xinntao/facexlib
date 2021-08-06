import argparse
import cv2
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize

from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor


def vis_parsing_maps(img, parsing_anno, stride, save_anno_path=None, save_vis_path=None):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                   [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                   [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    # 0: 'background'
    # attributions = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
    #                 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose',
    #                 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l',
    #                 16 'cloth', 17 'hair', 18 'hat']
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    if save_anno_path is not None:
        cv2.imwrite(save_anno_path, vis_parsing_anno)

    if save_vis_path is not None:
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(img, 0.4, vis_parsing_anno_color, 0.6, 0)

        cv2.imwrite(save_vis_path, vis_im)


def main(img_path, output):
    net = init_parsing_model(model_name='bisenet')

    img_name = os.path.basename(img_path)
    img_basename = os.path.splitext(img_name)[0]

    img_input = cv2.imread(img_path)
    img_input = cv2.resize(img_input, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = img2tensor(img_input.astype('float32') / 255., bgr2rgb=True, float32=True)
    normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
    img = torch.unsqueeze(img, 0).cuda()

    with torch.no_grad():
        out = net(img)[0]
    out = out.squeeze(0).cpu().numpy().argmax(0)

    vis_parsing_maps(
        img_input,
        out,
        stride=1,
        save_anno_path=os.path.join(output, f'{img_basename}.png'),
        save_vis_path=os.path.join(output, f'{img_basename}_vis.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='datasets/ffhq/ffhq_512/00000000.png')
    parser.add_argument('--output', type=str, default='results', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    main(args.input, args.output)
