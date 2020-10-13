"""
python augmentation.py --input_dir data/train --output_dir data/train/train_aug -n 10

data/train
├── image
├── label
├── dst_img_id.txt
├── src_img_id.txt
└── train_aug
    ├── image
    └── label
"""


import argparse
import random
from concurrent import futures
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def mixup(src_id, dst_id, image_dir, mask_dir, output_dir):
    dst_img = cv2.imread(f"{image_dir}/{dst_id}.tif")
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
    dst_mask = cv2.imread(f"{mask_dir}/{dst_id}.png", -1)

    src_img = cv2.imread(f"{image_dir}/{src_id}.tif")
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_mask = cv2.imread(f"{mask_dir}/{src_id}.png", -1)

    src_cls = np.unique(src_mask)
    dst_cls = np.unique(dst_mask)[0]
    src_cls = src_cls[src_cls != dst_cls]

    if src_cls.size > 0:
        random.shuffle(src_cls)
        n_cls_to_mix = int(len(src_cls) * 0.5)
        src_cls = src_cls[:n_cls_to_mix]

        m_masks = [src_mask == cls for cls in src_cls]
        for m_mask in m_masks:
            i_mask = np.dstack([m_mask] * 3)

            dst_img[i_mask] = src_img[i_mask]
            dst_mask[m_mask] = src_mask[m_mask]

        cv2.imwrite(f"{output_dir}/image/{src_id}_{dst_id}.tif", dst_img)
        cv2.imwrite(f"{output_dir}/label/{src_id}_{dst_id}.png", dst_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create mixed images for image with single class.")
    parser.add_argument(
        "--input_dir",
        required=True,
        type=lambda x: Path(x),
        help="Directory contains image and label path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=lambda x: Path(x),
        help="Directory to output mixed image and mask.",
    )
    parser.add_argument(
        "-n", "--num", type=int, default=100, help="Number of paired image to mix."
    )
    args = parser.parse_args()
    image_dir = args.input_dir.joinpath("image")
    mask_dir = args.input_dir.joinpath("label")
    output_dir = args.output_dir
    output_dir.joinpath("image").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("label").mkdir(parents=True, exist_ok=True)

    with open(args.input_dir.joinpath("dst_img_id.txt"), "r") as f1:
        dst_id_list = [l1.rstrip("\n") for l1 in f1]
    with open(args.input_dir.joinpath("src_img_id.txt"), "r") as f2:
        src_id_list = [l2.rstrip("\n") for l2 in f2]
    maxnum = min(len(dst_id_list), len(src_id_list))
    assert args.num <= maxnum, f"maximum mixup is {maxnum}"
    random.shuffle(dst_id_list)
    random.shuffle(src_id_list)
    dst_id_list = dst_id_list[: args.num]
    src_id_list = src_id_list[: args.num]

    with futures.ThreadPoolExecutor() as executor:
        result = [
            executor.submit(mixup, src_id, dst_id, image_dir, mask_dir, output_dir)
            for src_id, dst_id in zip(src_id_list, dst_id_list)
        ]
        for f in tqdm(futures.as_completed(result), total=len(result)):
            f.result()
