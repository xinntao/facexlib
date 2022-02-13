import argparse
import cv2
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize

from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor


def vis_parsing_maps(img, parsing_anno, stride, save_anno_path=None, save_vis_path=None):
    # Colors for all parts
    part_colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
                   [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
                   [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    #     0: 'background' 1: 'skin'   2: 'nose'
    #     3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
    #     6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
    #     9: 'r_ear'  10: 'mouth' 11: 'u_lip'
    #     12: 'l_lip' 13: 'hair'  14: 'hat'
    #     15: 'ear_r' 16: 'neck_l'    17: 'neck'
    #     18: 'cloth'
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
    net = init_parsing_model(model_name='parsenet')

    img_name = os.path.basename(img_path)
    img_basename = os.path.splitext(img_name)[0]

    img_input = cv2.imread(img_path)
    # resize to 512 x 512 for better performance
    img_input = cv2.resize(img_input, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = img2tensor(img_input.astype('float32') / 255., bgr2rgb=True, float32=True)
    normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
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
