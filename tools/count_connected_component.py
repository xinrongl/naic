from pathlib import Path
import cv2
import numpy as np
import sys
from tqdm import tqdm

filenames = sorted(Path(sys.argv[1]).glob("*"))
summary = open("connected_component.txt", "w")
summary.write("image,cls,size\n")
for filename in tqdm(filenames):
    image = cv2.imread(str(filename), -1)
    area = []
    for cls in sorted(np.unique(image)):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            (image == cls).astype(np.uint8), 4, cv2.CV_32S
        )
        size = stats[1:, -1]
        for s in size:
            summary.write(f"{filename.parts[-1]}, {cls}, {s}\n")
summary.close()
