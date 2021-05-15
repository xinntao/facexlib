import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

from facexlib.detection import init_detection_model
from facexlib.tracking.sort import SORT


def main(args):
    detect_interval = args.detect_interval
    margin = args.margin
    face_score_threshold = args.face_score_threshold

    save_frame = True
    if save_frame:
        colors = np.random.rand(32, 3)

    # init detection model and tracker
    det_net = init_detection_model('retinaface_resnet50', half=False)
    tracker = SORT(max_age=1, min_hits=2, iou_threshold=0.2)
    print('Start track...')

    # track over all frames
    frame_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.jpg')))
    pbar = tqdm(total=len(frame_paths), unit='frames', desc='Extract')
    for idx, path in enumerate(frame_paths):
        img_basename = os.path.basename(path)
        frame = cv2.imread(path)
        img_size = frame.shape[0:2]

        # detection face bboxes
        with torch.no_grad():
            bboxes = det_net.detect_faces(frame, 0.97)

        additional_attr = []
        face_list = []

        for idx_bb, bbox in enumerate(bboxes):
            score = bbox[4]
            if score > face_score_threshold:
                bbox = bbox[0:5]
                det = bbox[0:4]

                # face rectangle
                det[0] = np.maximum(det[0] - margin, 0)
                det[1] = np.maximum(det[1] - margin, 0)
                det[2] = np.minimum(det[2] + margin, img_size[1])
                det[3] = np.minimum(det[3] + margin, img_size[0])
                face_list.append(bbox)
            additional_attr.append([score])
        trackers = tracker.update(np.array(face_list), img_size, additional_attr, detect_interval)

        pbar.update(1)
        pbar.set_description(f'{idx}: detect {len(bboxes)} faces in {img_basename}')

        # save frame
        if save_frame:
            for d in trackers:
                d = d.astype(np.int32)
                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colors[d[4] % 32, :] * 255, 3)
                if len(face_list) != 0:
                    cv2.putText(frame, 'ID : %d  DETECT' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, colors[d[4] % 32, :] * 255, 2)
                    cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (1, 1, 1), 2)
                else:
                    cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                colors[d[4] % 32, :] * 255, 2)
            save_path = os.path.join(args.save_folder, img_basename)
            cv2.imwrite(save_path, frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Path to the input folder', type=str)
    parser.add_argument('--save_folder', help='Path to save visualized frames', type=str, default=None)

    parser.add_argument(
        '--detect_interval',
        help=('how many frames to make a detection, trade-off '
              'between performance and fluency'),
        type=int,
        default=1)
    # if the face is big in your video ,you can set it bigger for easy tracking
    parser.add_argument('--margin', help='add margin for face', type=int, default=20)
    parser.add_argument(
        '--face_score_threshold', help='The threshold of the extracted faces,range 0 < x <=1', type=float, default=0.85)

    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    main(args)

    # add verification
    # remove last few frames
