import argparse
import configparser
from datetime import datetime
from pathlib import Path

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from src.data import aug
from src.data.dataset import NAICDataset
from src.utiles.logger import MyLogger

TIMESTAMP = datetime.now()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

config = configparser.ConfigParser()
config.read("cfg.ini")
encoder_lists = config["model"]["encoder"].split()
path_dict = dict(config["path"])

parser = argparse.ArgumentParser("Train UNet with SMP api.")
parser.add_argument(
    "--encoder",
    default="efficientnet-b5",
    help="model encoder: " + " | ".join(encoder_lists),
)
parser.add_argument(
    "-w", "--weight", default="imagenet", help="Encoder pretrained weight"
)
parser.add_argument("--activation", default="sigmoid")
parser.add_argument(
    "--arch",
    choices=["unet", "linkednet", "fpn", "pspnet", "deeplabv3", "deeplabv3plus", "pan"],
)
parser.add_argument("-b", "--batch_size", type=int, default=16)
parser.add_argument("-lr", "--learning_rate", type=float, default=5e-4)
parser.add_argument("-wd", "--weight_decay", type=float, default=3e-4)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--num_epoch", default=10, type=int)
parser.add_argument("--loglevel", default="INFO")
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
args, _ = parser.parse_known_args()
arch_dict = {
    "unet": smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=8,
        activation=args.activation,
    ),
    "linknet": smp.Linknet(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=8,
        activation=args.activation,
    ),
    "fpn": smp.FPN(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=8,
        activation=args.activation,
    ),
    "pspnet": smp.PSPNet(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=8,
        activation=args.activation,
    ),
    "deeplabv3": smp.DeepLabV3(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=8,
        activation=args.activation,
    ),
    "deeplabv3plus": smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=8,
        activation=args.activation,
    ),
    "pan": smp.PAN(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=8,
        activation=args.activation,
    ),
}


def main():
    data_dir = Path(path_dict["train_data"])
    images_dir = data_dir.joinpath("image")
    masks_dir = data_dir.joinpath("label")
    label_df = pd.read_csv(data_dir.joinpath("label.csv"))
    if args.resume:
        logger.info(f"Loading checkpoints and resume training from {args.resume}")
        model = torch.load(args.resume)
    else:
        model = arch_dict[args.arch]

    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.weight)

    train_dataset = NAICDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        label_df=label_df,
        mode="train",
        classes=range(100, 801, 100),
        augmentation=aug.get_training_augmentation(),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    valid_dataset = NAICDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        label_df=label_df,
        mode="val",
        classes=range(100, 801, 100),
        augmentation=aug.get_validation_augmentation(),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.num_workers / 3),
    )
    loss = smp.utils.losses.JaccardLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.2, patience=5, verbose=False
    # )
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=DEVICE, verbose=True,
    )

    checkpoint_path = Path(f"{path_dict['checkpoints']}/{TIMESTAMP:%Y%m%d%H%M}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    max_score = 0.0
    for epoch in range(args.num_epoch):
        print(f"Epoch: {epoch+1}\tlr: {scheduler.get_last_lr()}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        scheduler.step()
        logger.debug(train_logs, valid_logs)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            torch.save(model, checkpoint_path.joinpath(f"best_model_{epoch+1}.pth"))
            logger.info("Model saved!")


if __name__ == "__main__":
    logger = MyLogger(args.loglevel)
    logger.set_stream_handler()
    logger.set_file_handler(
        f"{path_dict['logs']}/{TIMESTAMP:%Y%m%d%H%M}_{args.model}_{args.encoder}.log"
    )
    for arg, val in sorted(vars(args).items()):
        logger.info(f"{arg}: {val}")
    main()
