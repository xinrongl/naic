# (1) train segmentation model from the scratch using deeplabv3plus and efficientnet-b3 as encoder.
# cd naic
# python train.py --encoder efficientnet-b3 --weight imagenet --arch deeplabv3plus --batch_size 48 --num_workers 12 --parallel --num_epoch 10
import argparse
import configparser
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn
from torch import optim
from torch.utils.data import DataLoader

from src.data import aug
from src.data.dataset import NAICTrainDataset
from src.metrics import FWIoU
from src.optimizer import RAdam
from src.logger import MyLogger
from src.scheduler import PolynomialLRDecay

TIMESTAMP = datetime.now()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

config = configparser.ConfigParser()
config.read("cfg.ini")
encoder_lists = config["model"]["encoder"].split()
path_dict = dict(config["path"])
cls_frequency = list(map(lambda x: float(x), config["data"]["frequency"].split()))
parser = argparse.ArgumentParser("Train segmentation model with SMP api.")
parser.add_argument(
    "--encoder",
    default="efficientnet-b5",
    help="model encoder: " + " | ".join(encoder_lists),
)
parser.add_argument(
    "-w", "--weight", default=None, help="Encoder pretrained weight", required=True
)
parser.add_argument("--activation", default="sigmoid")
parser.add_argument(
    "--arch",
    required=True,
    help="model arch: "
    + " | ".join(
        ["unet", "linkednet", "fpn", "pspnet", "deeplabv3", "deeplabv3plus", "pan"]
    ),
    type=lambda arch: arch.lower(),
)
parser.add_argument("-depth", "--encoder_depth", type=int, default=5)
parser.add_argument("-b", "--batch_size", type=int, default=16)
parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5)
parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument(
    "--save_threshold",
    type=float,
    default=0.0,
    help="Save model if model score greater than threshold",
)
parser.add_argument("--patience", default=2, type=int)
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
parser.add_argument(
    "--parallel", action="store_true", help="Use multi-gpus for training"
)
parser.add_argument("--test", action="store_true", help="Test code use small dataset")
args, _ = parser.parse_known_args()
arch_dict = {
    "unet": smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.weight,
        classes=8,
        activation=args.activation,
        decoder_attention_type="scse",
        decoder_use_batchnorm=True,
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


def save_checkpoint(state, filename):
    torch.save(state, filename)


def main():
    data_dir = Path(path_dict["train_data"])
    # images_dir = data_dir.joinpath("image")
    images_dir = Path(config["train"]["image_dir"])
    logger.info(f"Loading images from {images_dir}")
    masks_dir = data_dir.joinpath("label")
    label_df = pd.read_csv(data_dir.joinpath("label_90.csv"))
    if args.test:
        train_df = label_df[label_df["label"] == "train"][:100]
        val_df = label_df[label_df["label"] == "val"][:100]
        label_df = pd.concat([train_df, val_df])

    logger.info(f"Use {len(label_df)} for training")
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.weight)
    train_dataset = NAICTrainDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        label_df=label_df,
        mode="train",
        classes=range(100, 801, 100),
        augmentation=aug.get_training_augmentation(),
        preprocessing=aug.get_preprocessing(preprocessing_fn),
    )

    valid_dataset = NAICTrainDataset(
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
        num_workers=args.num_workers,
    )

    if args.resume:
        checkpoints = torch.load(args.resume)
        state_dict = OrderedDict()
        for key, value in checkpoints["state_dict"].items():
            tmp = key[7:]
            state_dict[tmp] = value

        model = arch_dict[checkpoints["arch"]]
        model.load_state_dict(state_dict)
        logger.info(
            f"=> loaded checkpoint '{args.resume}' (epoch {args.resume.split('_')[-2]})"
        )
        optimizer = RAdam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(checkpoints["optimizer"])
    else:
        model = arch_dict[args.arch]
        optimizer = RAdam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

    if args.parallel:
        model = torch.nn.DataParallel(model).cuda()

    metrics = [
        smp.utils.metrics.IoU(threshold=args.threshold),
        FWIoU(threshold=args.threshold, frequency=cls_frequency),
    ]
    loss = smp.utils.losses.JaccardLoss()

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=args.patience, verbose=True
    # )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    scheduler = PolynomialLRDecay(
        optimizer, max_decay_steps=1000, end_learning_rate=5e-8, power=0.7
    )
    # optimizer.param_groups[0].update(
    #     {"lr": args.learning_rate, "wd": args.weight_decay}
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
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    checkpoint_path = Path(
        f"{path_dict['checkpoints']}/{args.arch}_{args.encoder}/{TIMESTAMP:%Y%m%d%H%M}"
    )
    if not args.test:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    max_score = checkpoints["best_fwiou"] if args.resume else 0.0
    start_epoch = checkpoints["epoch"] + 1 if args.resume else 0
    end_epoch = start_epoch + args.num_epoch
    logger.info(f"Current best score: {max_score:.4f}")
    # optimizer.param_groups[0]["lr"] = args.learning_rate
    logger.info(f"Start learning rate: {optimizer.param_groups[0]['lr']:f}")
    for epoch in range(start_epoch, end_epoch):
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_loss, train_iou, train_fwiou = train_logs.values()
        val_loss, val_iou, val_fwiou = valid_logs.values()
        logger.info(
            f"epoch [{epoch:03d} | {end_epoch:03d}] | lr: {optimizer.param_groups[0]['lr']:f} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | train iou: {train_iou:.4f} | val iou: {val_iou:.4f} | train fwiou: {train_fwiou:.4f} | val fwiou: {val_fwiou:.4f}"
        )
        scheduler.step(val_loss)
        if all([max_score < val_fwiou, val_fwiou > args.save_threshold, not args.test]):
            max_score = val_fwiou
            save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "encoder": args.encoder,
                    "encoder_weight": args.weight,
                    "state_dict": model.state_dict(),
                    "best_iou": val_iou,
                    "best_fwiou": val_fwiou,
                    "optimizer": optimizer.state_dict(),
                    "activation": args.activation,
                },
                checkpoint_path.joinpath(f"epoch_{epoch}_{val_fwiou:.4f}.pth"),
            )
            logger.info(f"Save checkpoint at {epoch}.")


if __name__ == "__main__":
    logger_dir = Path(f"{path_dict['logs']}/{args.arch}_{args.encoder}")
    logger_dir.mkdir(parents=True, exist_ok=True)
    logger = MyLogger(args.loglevel)
    logger.set_stream_handler()
    if not args.test:
        logger.set_file_handler(f"{logger_dir}/{TIMESTAMP:%Y%m%d%H%M}.log")
    for arg, val in sorted(vars(args).items()):
        logger.info(f"{arg}: {val}")
    logger.info("\n")
    main()
