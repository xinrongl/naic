import argparse
import os
import shutil
import sys
from concurrent import futures
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cv2_imread

parser = argparse.ArgumentParser("Implement local histogram equalize to images")
parser.add_argument(
    "-dst", type=lambda x: Path(x), help="Output directory to save processed image"
)
parser.add_argument("-src", type=lambda x: Path(x), help="Source image directory")
args = parser.parse_args()

LABEL_DF = pd.read_csv("data/train/label_90.csv")
TRAIN_FN = LABEL_DF.query("label=='train'")["image"].values
VAL_FN = LABEL_DF.query("label=='val'")["image"].values


def local_hist_equalizer(image):
    # local hist equalize
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    channels = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[2])

    image = cv2.merge(channels, hsv)
    return image


def global_hist_equalizer(image):
    r, g, b = cv2.split(image)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)
    return cv2.merge((r, g, b))


def submit_(filename):
    if filename.name in TRAIN_FN:
        image = cv2_imread(str(filename), "image")
        image_post = global_hist_equalizer(image)
        cv2.imwrite(f"{args.dst}/{filename.name}", image_post)
    else:
        shutil.copy(args.src.joinpath(filename.name), args.dst.joinpath(filename.name))


if __name__ == "__main__":
    args.dst.mkdir(exist_ok=True, parents=True)
    filenames = sorted(args.src.glob("*"))
    with futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(submit_, filename) for filename in filenames]
    for r in tqdm(futures.as_completed(results), total=100000):
        r.result()
