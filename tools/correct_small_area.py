# python tools/correct_small_area.py --input_dir data/results --output_dir data/vote3/results -t 100

import argparse
import os
from concurrent import futures
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from tqdm import tqdm


def correct_small_area(in_mask, threshold):
    out_mask = np.zeros((256, 256))
    for cls in np.unique(in_mask):
        cls_mask = cls == in_mask
        remove_small_objects(
            cls_mask, min_size=threshold, connectivity=4, in_place=True
        )
        remove_small_holes(
            cls_mask, area_threshold=threshold, connectivity=4, in_place=True
        )
        out_mask[cls_mask] = cls
    for cls in np.unique(in_mask):
        cls_mask = cls == in_mask
        if cls == 200:
            out_mask[cls_mask] = cls
        else:
            out_mask[cls_mask & (out_mask == 0)] = cls
    return out_mask.astype(np.uint16)


def _submit(save_dir, filename):
    in_mask = cv2.imread(filename, -1)
    corrected_mask = correct_small_area(in_mask, threshold=args.threshold)
    cv2.imwrite(f"{save_dir}/{filename.parts[-1]}", corrected_mask.astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Postprocess small object from the predicted mask."
    )
    parser.add_argument(
        "--input_dir",
        type=lambda x: Path(x),
        required=True,
        help="Directory to the predicted mask.",
    )
    parser.add_argument(
        "--output_dir",
        type=lambda x: Path(x),
        required=True,
        help="Directory to output the corrected mask.",
    )
    parser.add_argument("-t", "--threshold", type=int, default=100)

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    filenames = list(args.input_dir.glob("*"))
    with futures.ThreadPoolExecutor() as executor:
        results = [
            executor.submit(_submit, args.output_dir, filename)
            for filename in filenames
        ]
        for f in tqdm(futures.as_completed(results), total=len(results)):
            f.result()
    if args.output_dir.parts[-1] == "results":
        print(f"Creating zip file to: {args.output_dir.parent}")
        os.system(f"cd {args.output_dir.parent} && zip -r -q results.zip results/")
