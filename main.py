# As usual, a bit of setup
from __future__ import print_function
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'EfficientPose'))

import cv2
import click
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# from efficientpose import EP_rotation, EP_classification
from misc.data_utils import load_linemod
from misc.saliency import compute_gradcam_maps, compute_backprop_maps
from misc.image_utils import deprocess_image

from EfficientPose.inference import build_model_and_load_weights

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def show_saliency_maps(sess, X, model, attr='gradcam', task='rotation', dest=None):

    n_img = len(X[0])
    print('shape of one image:', X[0][0].shape)

    sm_fn = compute_backprop_maps if attr == 'backprop' else compute_gradcam_maps
    sm = sm_fn(sess, X, model, task=task)

    for i in range(n_img):
        img_orig = deprocess_image(X[0][i, :, :, :])
        # min-max rescaling
        print(sm[i, :, :].max(), sm[i, :, :].min(), sm[i, :, :].mean())
        heatmap = cv2.resize(sm[i, :, :], img_orig.shape[:2])
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + 1e-8
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.8, img_orig, 0.5, 0)
        print(super_imposed_img.shape)

        if dest is not None:
            cv2.imwrite(f"{dest}/{i}.png", super_imposed_img)
        else:
            while cv2.waitKey(1) != ord('q'):
                cv2.imshow("result", super_imposed_img)

@click.command()
@click.option('--dataset', '-d', type=click.Path(exists=True), default=None, required=False,
              help="Root directory of the preprocessed LineMod Dataset, this should point to the dowloaded `Linemod_and_Occlusion` diirectory")
@click.option('--dest_folder', '-f', type=click.Path(exists=True), default=None, required=False,
              help="Destination path for the produced saliency images. If omitted the images are displayed and not saved.")
@click.option('--object', '-o', type=int, default=1, required=False,
              help="ID of the desired LineMOD object")
@click.option('--weights', '-w', type=click.Path(exists=True), required=True,
              help="Directory of the `.h5` weights file for EfficientPose. Be careful to select the weight file for the right object."
                   "(i.e, `../weights/object_0/phi_0_linemod_best_ADD-S.h5` for object 0)")
@click.option('--method', '-m', type=click.Choice(['gradcam', 'backprop'], case_sensitive=False), default="gradcam",
              help="Attribution method to use, either `gradcam` or `backprop`")
@click.option('--task', '-t', type=click.Choice(['rotation', 'translation'], case_sensitive=False), default="rotation",
              help="EfficientPose Subtask to analize, either `rotation` or `translation`")
@click.option('--noaruco', type=click.BOOL, default=False, is_flag=True,
              help="If present, the saliency method is evaluated on the test images with ArUCO codes blocked."
                   "The subdirectory `rgb_noaruco` must be available in the dataset directory for the target object")
@click.option('--cuda', '-c', type=bool, default=False, required=False, help="Use cuda for inference, default=False")
def main(dataset, dest_folder, object, weights, method, task, noaruco, cuda):

    if not cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    tf.reset_default_graph()
    sess = tf.compat.v1.keras.backend.get_session()

    # Load EP Model
    phi = 0
    class_to_name = {object: "object"}
    score_threshold = 0.1
    num_classes = len(class_to_name)

    # Load Images
    dataset_root = Path(dataset)
    rgb_dir = "rgb_noaruco" if noaruco else "rgb"
    path_to_images = dataset_root / f"{str(object).zfill(2)}" / rgb_dir
    path_to_gt = dataset_root / f"{str(object).zfill(2)}" / "gt.yml"

    n_img = 25
    X, y_gt = load_linemod(path_to_images, path_to_gt, 512, n_img)
    if not path_to_images.exists():
        print("Error: the given path to the images {} does not exist!".format(path_to_images))
        return

    image_list = [filename for filename in os.listdir(path_to_images) if '*.png' in str(filename)]
    print("\nInfo: found {} image files".format(len(image_list)))

    # build model and load weights
    #path_to_weights = f"../weights/Linemod/object_{object}/{weights}"
    model, image_size = build_model_and_load_weights(phi, num_classes, score_threshold, weights)

    """# Sanity check
    boxes, scores, labels, rotations, translations = model.predict_on_batch(X)
    best_pred = tf.argmax(scores, axis=1)
    indices = tf.gather(rotations, best_pred, batch_dims=1)"""

    rot_saliency = show_saliency_maps(sess, X, model, method, task, dest_folder)
    print(rot_saliency)

if __name__ == '__main__':
    main()
