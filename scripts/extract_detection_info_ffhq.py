import cv2
import glob
import numpy as np
import os
import torch
from PIL import Image
from tqdm import tqdm

from facexlib.detection import init_detection_model


def draw_and_save(image, bboxes_and_landmarks, save_path, order_type=1):
    """Visualize results
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32)
    for b in bboxes_and_landmarks:
        # confidence
        cv2.putText(image, '{:.4f}'.format(b[4]), (int(b[0]), int(b[1] + 12)), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (255, 255, 255))
        # bounding boxes
        b = list(map(int, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        # landmarks
        if order_type == 0:  # mtcnn
            cv2.circle(image, (b[5], b[10]), 1, (0, 0, 255), 4)
            cv2.circle(image, (b[6], b[11]), 1, (0, 255, 255), 4)
            cv2.circle(image, (b[7], b[12]), 1, (255, 0, 255), 4)
            cv2.circle(image, (b[8], b[13]), 1, (0, 255, 0), 4)
            cv2.circle(image, (b[9], b[14]), 1, (255, 0, 0), 4)
        else:  # retinaface, centerface
            cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)
    # save image
    cv2.imwrite(save_path, image)


det_net = init_detection_model('retinaface_resnet50')
half = False

det_net.cuda().eval()
if half:
    det_net = det_net.half()

img_list = sorted(glob.glob('../../BasicSR-private/datasets/ffhq/ffhq_512/*'))


def get_center_landmark(landmarks, center):
    center = np.array(center)
    center_dist = []
    for landmark in landmarks:
        landmark_center = np.array([(landmark[0] + landmark[2]) / 2, (landmark[1] + landmark[3]) / 2])
        dist = np.linalg.norm(landmark_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return landmarks[center_idx]


pbar = tqdm(total=len(img_list), unit='image')
save_np = []
for idx, path in enumerate(img_list):
    img_name = os.path.basename(path)
    pbar.update(1)
    pbar.set_description(path)
    img = Image.open(path)
    with torch.no_grad():
        bboxes, warped_face_list = det_net.align_multi(img, 0.97, half=half)
        if len(bboxes) > 1:
            bboxes = [get_center_landmark(bboxes, (256, 256))]
        save_np.append(bboxes)
        # draw_and_save(img, bboxes, os.path.join('tmp', img_name), 1)
np.save('ffhq_det_info.npy', save_np)
