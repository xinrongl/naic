import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from check_mask import plot_mask
import os
import threading
import cv2 as cv
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from argparse import ArgumentParser
from PIL import Image



def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical




def label_resize_vis(label, img=None, alpha=0.5):
    '''
    :param label:原始标签
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    label = cv.resize(label.copy(), None, fx=0.1, fy=0.1)
    r = np.where(label == 1, 255, 0)
    g = np.where(label == 2, 255, 0)
    b = np.where(label == 3, 255, 0)
    yellow = np.where(label == 4, 255, 0)
    anno_vis = np.dstack((b, g, r)).astype(np.uint8)
    # 黄色分量(红255, 绿255, 蓝0)
    anno_vis[:, :, 0] = anno_vis[:, :, 0] + yellow
    anno_vis[:, :, 1] = anno_vis[:, :, 1] + yellow
    anno_vis[:, :, 2] = anno_vis[:, :, 2] + yellow
    if img is None:
        return anno_vis
    else:
        overlapping = cv.addWeighted(img, alpha, anno_vis, 1 - alpha, 0)
        return overlapping


def remove_small_objects_and_holes(class_type, label, min_size, area_threshold, in_place=True):
    print("------------- class_n : {} start ------------".format(class_type))
    if class_type == 4:
        # kernel = cv.getStructuringElement(cv.MORPH_RECT,(500,500))
        # label = cv.dilate(label,kernel)
        # kernel = cv.getStructuringElement(cv.MORPH_RECT,(10,10))
        # label = cv.erode(label,kernel)
        label = remove_small_objects(label == 1, min_size=min_size, connectivity=1, in_place=in_place)
        label = remove_small_holes(label == 1, area_threshold=area_threshold, connectivity=1, in_place=in_place)
    else:
        label = remove_small_objects(label == 1, min_size=min_size, connectivity=1, in_place=in_place)
        label = remove_small_holes(label == 1, area_threshold=area_threshold, connectivity=1, in_place=in_place)
    print("------------- class_n : {} finished ------------".format(class_type))
    return label


result_dir = Path("data/vote1/results")
sorted(result_dir.glob("*"))

img_id ="10087"


image = cv2.cvtColor(cv2.imread(f"data/vote1/{img_id}.jpg"), cv2.COLOR_BGR2RGB)

mask = Image.open(r.esult_dir/f"{img_id}.png")
mask = cv2.imread(f"{result_dir}/{img_id}.png", -1)
masks = [mask == cls for cls in sorted(np.unique(mask))]

plt.imshow(mask)
plot_mask(mask)

from skimage.morphology import  remove_small_holes, remove_small_objects
import matplotlib.pyplot as plt
plot_mask2(remove_small_objects(mask, min_size=10, connectivity=4))
plot_mask2(mask)
plt.imshow(masks[0])
(mask == 100)

traffic_mask = np.where(mask==200, mask, np.uint16(800))
traffic_mask = np.where(mask==200, 1, 0)
plot_mask2(traffic_mask)
plot_mask2(remove_small_objects(traffic_mask, 900, connectivity=1))

np.unique(traffic_mask)
np.unique(remove_small_objects(traffic_mask, 800, connectivity=4))

remove_small_holes()

np.sum(remove_small_objects(traffic_mask, 900, connectivity=1)==traffic_mask)