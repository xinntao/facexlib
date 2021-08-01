import argparse
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from facexlib.matting import init_matting_model
from facexlib.utils import img2tensor


def main(args):
    modnet = init_matting_model()

    # read image
    img = cv2.imread(args.img_path) / 255.
    # unify image channels to 3
    if len(img.shape) == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, 0:3]

    img_t = img2tensor(img, bgr2rgb=True, float32=True)
    normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    img_t = img_t.unsqueeze(0).cuda()

    # resize image for input
    _, _, im_h, im_w = img_t.shape
    ref_size = 512
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    img_t = F.interpolate(img_t, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(img_t, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    cv2.imwrite(args.save_path, (matte * 255).astype('uint8'))

    # get foreground
    matte = matte[:, :, None]
    foreground = img * matte + np.full(img.shape, 1) * (1 - matte)
    cv2.imwrite(args.save_path.replace('.png', '_fg.png'), foreground * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test.jpg')
    parser.add_argument('--save_path', type=str, default='test_matting.png')
    args = parser.parse_args()

    main(args)
