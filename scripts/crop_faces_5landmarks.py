import glob
import os

import facexlib.utils.face_restoration_helper as face_restoration_helper


def crop_one_img(img, save_cropped_path=None):
    FaceRestoreHelper.clean_all()
    FaceRestoreHelper.read_image(img)
    # get face landmarks
    FaceRestoreHelper.get_face_landmarks_5()
    FaceRestoreHelper.align_warp_face(save_cropped_path)


if __name__ == '__main__':
    # initialize face helper
    FaceRestoreHelper = face_restoration_helper.FaceRestoreHelper(upscale_factor=1)

    img_paths = glob.glob('/home/wxt/Projects/test/*')
    save_path = 'test'
    for idx, path in enumerate(img_paths):
        print(idx, path)
        file_name = os.path.basename(path)
        save_cropped_path = os.path.join(save_path, file_name)
        crop_one_img(path, save_cropped_path=save_cropped_path)
