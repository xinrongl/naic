from pathlib import Path
import cv2
import numpy as np
from check_mask import plot_mask
from skimage.morphology import remove_small_holes, remove_small_objects

filenames = sorted(Path("data/results").glob("*"))


def correct_small_area(filename):
    in_mask = cv2.imread(str(filename), -1)
    out_mask = np.zeros((256, 256))
    for cls in np.unique(in_mask):
        cls_mask = cls == in_mask
        remove_small_objects(cls_mask, min_size=100, connectivity=4, in_place=True)
        remove_small_holes(cls_mask, area_threshold=100, connectivity=4, in_place=True)
        out_mask[cls_mask] = cls
    print(np.unique(out_mask))
    for cls in np.unique(in_mask):
        cls_mask = cls == in_mask
        if cls == 200:
            out_mask[cls_mask] = cls
        else:
            out_mask[cls_mask & (out_mask == 0)] = cls
    # cv2.imwrite(f"{save_dir}/{filename.parts[-1]}", out_mask.astype(np.uint16))
    return out_mask.astype(np.uint16)


plot_mask(correct_small_area("data/results/10087.png"))
