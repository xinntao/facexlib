import cv2
import numpy as np
import os
import torch

from facexlib.detection import init_detection_model
from facexlib.utils.misc import imwrite


def get_largest_face(det_faces, h, w):

    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array([(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx


class FaceRestoreHelper(object):
    """Helper for the face restoration pipeline (base class)."""

    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 template_3points=True):
        self.template_3points = template_3points  # improve robustness
        self.upscale_factor = upscale_factor
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1), 'crop ration only supports >=1'
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))

        if self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])

        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []

        # init face detection model
        self.face_det = init_detection_model(det_model, half=False)

    def set_upscale_factor(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def read_image(self, img):
        """img can be image path or cv2 loaded image."""
        # self.input_img is Numpy array, (h, w, c), BGR, uint8, [0, 255]
        if isinstance(img, str):
            self.input_img = cv2.imread(img)
        else:
            self.input_img = img

    def get_face_landmarks_5(self, only_keep_largest=False, only_center_face=False, pad_blur=False):
        with torch.no_grad():
            bboxes = self.face_det.detect_faces(self.input_img, 0.97)
        for bbox in bboxes:
            if self.template_3points:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 11, 2)])
            else:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            self.all_landmarks_5.append(landmark)
            self.det_faces.append(bbox[0:5])
        if len(self.det_faces) == 0:
            return 0
        if only_keep_largest:
            h, w, _ = self.input_img.shape
            self.det_faces, largest_idx = get_largest_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[largest_idx]]
        elif only_center_face:
            h, w, _ = self.input_img.shape
            self.det_faces, center_idx = get_center_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[center_idx]]
        return len(self.all_landmarks_5)

    def align_warp_face(self, save_cropped_path=None, border_mode='constant'):
        """Align and warp faces with face template.
        """
        for idx, landmark in enumerate(self.all_landmarks_5):
            # use 5 landmarks to get affine matrix
            affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template)[0]
            self.affine_matrices.append(affine_matrix)
            # warp and crop faces
            if border_mode == 'constant':
                border_mode = cv2.BORDER_CONSTANT
            elif border_mode == 'reflect101':
                border_mode = cv2.BORDER_REFLECT101
            elif border_mode == 'reflect':
                border_mode = cv2.BORDER_REFLECT
            cropped_face = cv2.warpAffine(
                self.input_img, affine_matrix, self.face_size, borderMode=border_mode,
                borderValue=(135, 133, 132))  # gray
            self.cropped_faces.append(cropped_face)
            # save the cropped face
            if save_cropped_path is not None:
                path = os.path.splitext(save_cropped_path)[0]
                save_path = f'{path}_{idx:02d}.{self.save_ext}'
                imwrite(cropped_face, save_path)

    def get_inverse_affine(self, save_inverse_affine_path=None):
        """Get inverse affine matrix."""
        for idx, affine_matrix in enumerate(self.affine_matrices):
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine *= self.upscale_factor
            self.inverse_affine_matrices.append(inverse_affine)
            # save inverse affine matrices
            if save_inverse_affine_path is not None:
                path, _ = os.path.splitext(save_inverse_affine_path)
                save_path = f'{path}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)

    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def paste_faces_to_input_image(self, save_path=None):
        h, w, _ = self.input_img.shape
        h_up, w_up = h * self.upscale_factor, w * self.upscale_factor
        # simply resize the background
        upsample_img = cv2.resize(self.input_img, (w_up, h_up))
        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))
            mask = np.ones((*self.face_size, 3), dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            # remove the black borders
            inv_mask_erosion = cv2.erode(inv_mask, np.ones((2 * self.upscale_factor, 2 * self.upscale_factor),
                                                           np.uint8))
            inv_restored_remove_border = inv_mask_erosion * inv_restored
            total_face_area = np.sum(inv_mask_erosion) // 3
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            upsample_img = inv_soft_mask * inv_restored_remove_border + (1 - inv_soft_mask) * upsample_img

        upsample_img = upsample_img.astype(np.uint8)
        if save_path is not None:
            path = os.path.splitext(save_path)[0]
            save_path = f'{path}.{self.save_ext}'
            imwrite(upsample_img, save_path)
        return upsample_img

    def clean_all(self):
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
