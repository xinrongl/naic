import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import remove_small_holes
from tqdm import tqdm
from concurrent import futures

parser = argparse.ArgumentParser("Remove small object for the predicted mask.")
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
parser.add_argument("--threshold", type=int, default=5000)

args = parser.parse_args()
args.output_dir.mkdir(exist_ok=True, parents=True)


def correct_small_area(mask_filename, save_dir):
    filename = mask_filename.parts[-1]
    in_mask = cv2.imread(str(mask_filename), -1)
    out_mask = np.zeros((256, 256))
    for cls in np.unique(in_mask):
        cls_mask = cls == in_mask
        # remove_small_objects(cls_mask, min_size=100, connectivity=4, in_place=True)
        remove_small_holes(
            cls_mask, area_threshold=args.threshold, connectivity=4, in_place=True
        )
        out_mask[cls_mask] = cls
    cv2.imwrite(f"{save_dir}/{filename}", out_mask.astype(np.uint16))


filenames = list(args.input_dir.glob("*"))
with futures.ThreadPoolExecutor() as executor:
    results = [
        executor.submit(correct_small_area, filename, args.output_dir)
        for filename in filenames
    ]
    for f in tqdm(futures.as_completed(results), total=len(results)):
        f.result()
