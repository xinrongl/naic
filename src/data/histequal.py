import os
import sys
from pathlib import Path
import argparse
from concurrent import futures
from tqdm import tqdm
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import cv2_imread

parser = argparse.ArgumentParser("Implement local histogram equalize to images")
parser.add_argument(
    "-dst", type=lambda x: Path(x), help="Output directory to save processed image"
)
parser.add_argument("-src", type=lambda x: Path(x), help="Source image directory")
args = parser.parse_args()


def hist_equalizer(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    channels = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[2], channels[1])
    cv2.merge(channels, hsv)
    return image


def submit_(filename):
    image = cv2_imread(str(filename), "image")
    image_post = hist_equalizer(image)
    cv2.imwrite(f"{args.dst}/{filename.name}", image_post)


if __name__ == "__main__":
    args.dst.mkdir(exist_ok=True, parents=True)
    filenames = sorted(args.src.glob("*"))
    with futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(submit_, filename) for filename in filenames]
    for r in tqdm(futures.as_completed(results), total=len(results)):
        r.result()
