import cv2
import numpy as np
from PIL import Image

bboxes = np.load('ffhq_det_info.npy', allow_pickle=True)

bboxes = np.array(bboxes).squeeze(1)

bboxes = np.mean(bboxes, axis=0)

print(bboxes)


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


img = Image.open('inputs/00000000.png')
# bboxes = np.array([
#     118.177826 * 2, 92.759514 * 2, 394.95926 * 2, 472.53278 * 2, 0.9995705 * 2,  # noqa: E501
#     686.77227723, 488.62376238, 586.77227723, 493.59405941, 337.91089109,
#     488.38613861, 437.95049505, 493.51485149, 513.58415842, 678.5049505
# ])
# bboxes = bboxes / 2
draw_and_save(img, [bboxes], 'template_detall.png', 1)
