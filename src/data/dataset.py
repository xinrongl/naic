from pathlib import Path, PurePath
import cv2
import numpy as np
from torch.utils.data import Dataset


class NAICDataset(Dataset):
    CLASSES = [100, 200, 300, 400, 500, 600, 700, 800]

    def __init__(
        self,
        images_dir: PurePath,
        masks_dir: PurePath,
        label_df,
        mode,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        assert mode in ["train", "val", "testcode"]
        if mode != "testcode":
            self.label_df = label_df[label_df["label"] == mode]
            self.label_df.reset_index(drop=True, inplace=True)
        else:
            self.label_df = label_df.iloc[:100]
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        # convert str names to class values on masks
        self.class_values = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        sample = self.label_df.iloc[i]
        image_fname = sample["image"]
        mask_fname = sample["mask"]

        # read data
        image = cv2.imread(f"{self.images_dir / image_fname}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f"{self.masks_dir / mask_fname}", -1)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.label_df)
