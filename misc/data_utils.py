from __future__ import print_function

import os
import cv2
import yaml
import numpy as np

from EfficientPose.inference import get_linemod_camera_matrix, get_linemod_3d_bboxes, preprocess

def load_linemod(path_to_images, path_to_gt, image_size, n_img=5):   
    '''Return
    - X_train: (N_tr, 512, 512) array of training images
    - y_train: (N_tr,) array of training labels'''

    step = 25

    image_extension = ".png"
    if not os.path.exists(path_to_images):
        print("Error: the given path to the images {} does not exist!".format(path_to_images))
        return

    image_list = [filename for filename in os.listdir(path_to_images) if image_extension in filename]
    image_list.sort()

    camera_matrix = get_linemod_camera_matrix()
    translation_scale_norm = 1000.0

    X_img = []
    X_cam = []
    y = np.zeros((n_img, 3))

    with open(path_to_gt, 'r') as gt:
        gt_data = yaml.safe_load(gt)

    for i in range(0, n_img*step, step):

        image_filename = image_list[i]
        image_path = os.path.join(path_to_images, image_filename)
        image = cv2.imread(image_path)

        R = np.array(gt_data[i][0]['cam_R_m2c'])
        R = np.reshape(R, (3,3))
        angles, _ = cv2.Rodrigues(R)
        y[i // step] = np.squeeze(angles, axis=1)
        input_list, scale = preprocess(image, image_size, camera_matrix, translation_scale_norm)

        X_img.append(input_list[0])
        X_cam.append(input_list[1])

    X_img = np.array(X_img).squeeze(1)
    X_cam = np.array(X_cam).squeeze(1)

    X = [X_img, X_cam]

    return X, y
