# python inference.py --checkpoint checkpoints/deeplabv3plus_efficientnet-b3/202010041423/epoch_9_0.6591.pth --input_dir data/image_A --output_dir ./results
import argparse
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.correct_small_area import correct_small_area

from src.data.aug import get_preprocessing
from src.data.dataset import NAICTestDataset

parser = argparse.ArgumentParser("Inference")
parser.add_argument("--checkpoint", help="Path to chechkpoint file")
parser.add_argument(
    "--input_dir", help="Path to test image directory", default="./data/image_A"
)
parser.add_argument("-dst", "--output_dir", help="Path to store output mask")
parser.add_argument(
    "--postprocessing",
    type="store_true",
    help="Remove isolated small area from predicted mask",
)
args = parser.parse_args()


checkpoint = torch.load(args.checkpoint)
arch_dict = {
    "unet": smp.Unet(
        encoder_name=checkpoint["encoder"],
        encoder_weights=checkpoint["encoder_weight"],
        classes=8,
        activation=checkpoint["activation"],
        decoder_attention_type="scse",
        decoder_use_batchnorm=True,
    ),
    "linknet": smp.Linknet(
        encoder_name=checkpoint["encoder"],
        encoder_weights=checkpoint["encoder_weight"],
        classes=8,
        activation=checkpoint["activation"],
    ),
    "fpn": smp.FPN(
        encoder_name=checkpoint["encoder"],
        encoder_weights=checkpoint["encoder_weight"],
        classes=8,
        activation=checkpoint["activation"],
    ),
    "pspnet": smp.PSPNet(
        encoder_name=checkpoint["encoder"],
        encoder_weights=checkpoint["encoder_weight"],
        classes=8,
        activation=checkpoint["activation"],
    ),
    "deeplabv3": smp.DeepLabV3(
        encoder_name=checkpoint["encoder"],
        encoder_weights=checkpoint["encoder_weight"],
        classes=8,
        activation=checkpoint["activation"],
    ),
    "deeplabv3plus": smp.DeepLabV3Plus(
        encoder_name=checkpoint["encoder"],
        encoder_weights=checkpoint["encoder_weight"],
        classes=8,
        activation=checkpoint["activation"],
    ),
    "pan": smp.PAN(
        encoder_name=checkpoint["encoder"],
        encoder_weights=checkpoint["encoder_weight"],
        classes=8,
        activation=checkpoint["activation"],
    ),
}


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = arch_dict[checkpoint["arch"]]
state_dict = OrderedDict()
for key, value in checkpoint["state_dict"].items():
    tmp = key[7:]
    state_dict[tmp] = value
model.load_state_dict(state_dict)
model.cuda()
model.eval()
preprocessing_fn = smp.encoders.get_preprocessing_fn(
    checkpoint["encoder"], checkpoint["encoder_weight"]
)
test_dataset = NAICTestDataset(
    test_dir=args.input_dir, preprocessing=get_preprocessing(preprocessing_fn)
)
print(f"Number of testing image: {len(test_dataset)}")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    for i, (image, filename) in tqdm(enumerate(test_loader), total=len(test_dataset)):
        pred = model.predict(image.to(device))
        pred = pred.squeeze().cpu().numpy().round()
        out_mask = np.zeros((256, 256))
        for mask, cls in zip(pred, range(100, 801, 100)):
            cls_mask = mask == 1.0
            out_mask[cls_mask] = cls
        if args.postprocessing:
            out_mask = correct_small_area(in_mask=out_mask, threshold=150)
        out_mask = out_mask.astype(np.uint16)
        cv2.imwrite(
            str(out_dir.joinpath(filename[0].replace(".tif", ".png"))),
            out_mask,
        )
