# python tools/correct_small_area.py --input_dir data/results --output_dir data/vote3/results -t 100

import argparse
import os
from concurrent import futures
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from tqdm import tqdm

parser = argparse.ArgumentParser("Postprocess small object from the predicted mask.")
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


def correct_small_area(filename, save_dir):
    in_mask = cv2.imread(str(filename), -1)
    out_mask = np.zeros((256, 256))
    for cls in np.unique(in_mask):
        if cls == 200:  # ignore traffic
            label = in_mask == cls
        else:
            label = remove_small_holes(
                in_mask == cls, area_threshold=args.threshold, connectivity=4,
            )
        out_mask[label] = cls
    cv2.imwrite(f"{save_dir}/{filename.parts[-1]}", out_mask.astype(np.uint16))


if __name__ == "__main__":
    filenames = list(args.input_dir.glob("*"))
    with futures.ThreadPoolExecutor() as executor:
        results = [
            executor.submit(correct_small_area, filename, args.output_dir)
            for filename in filenames
        ]
        for f in tqdm(futures.as_completed(results), total=len(results)):
            f.result()
    if args.output_dir.parts[-1] == "results":
        print(f"Creating zip file to: {args.output_dir.parent}")
        os.system(f"cd {args.output_dir.parent} && zip -r -q results.zip results/")
