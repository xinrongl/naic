"""
python ../scripts/voting.py --results results1/ results2/ results3/ --out voted_results
"""
import argparse
from concurrent import futures
from concurrent.futures import Executor
from pathlib import Path

import cv2
import mmcv
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm

parser = argparse.ArgumentParser(description="voting")
parser.add_argument("--out", type=str, default="")
parser.add_argument("--results", type=str, nargs="+")
args = parser.parse_args()
imgs = [str(x.parts[-1]) for x in Path(args.results[0]).rglob("*.png")]


def vote(image, results):
    images = np.stack([cv2.imread(f"{x}/{image}", -1) for x in results])
    image_voted = np.squeeze(stats.mode(images, axis=0)[0])
    mmcv.imwrite(image_voted, f"{args.out}/{image}")


# Parallel(n_jobs=100)(delayed(vote)(image, args.results) for image in tqdm(imgs))

with futures.ThreadPoolExecutor() as executor:
    result = [executor.submit(vote, image, args.results) for image in imgs]
    for f in tqdm(futures.as_completed(result)):
        f.result()
