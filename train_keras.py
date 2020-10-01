from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

in_path = Path("/home/isabella/tensorflow-gpu/2020_NAIC/")
TRAIN_PATH = Path("/home/isabella/tensorflow-gpu/2020_NAIC/train/image/")
MASK_PATH = Path("/home/isabella/tensorflow-gpu/2020_NAIC/train/label/")
TEST_PATH = Path("/home/isabella/tensorflow-gpu/2020_NAIC/holdout/image_A/")

# TRAIN_PATH = Path("data/train/image")
# MASK_PATH = Path("data/train/label")
# TEST_PATH = Path("data/image_A")

train_ids = sorted(TRAIN_PATH.glob("*.tif"))
mask_ids = sorted(MASK_PATH.glob("*.png"))
test_ids = sorted(TEST_PATH.glob("*"))

X_train = np.array([cv2.imread(str(filename)) for filename in train_ids])
Y_train = np.array([cv2.imread(str(filename), -1) for filename in mask_ids])


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2.0 * intersection + smooth) / (
        K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth
    )


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9024)],
        )
        tf.config.experimental.set_virtual_device_configuration(
            gpus[1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9024)],
        )
        # tf.config.experimental.set_virtual_device_configuration(gpus[1], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6024)])
    except RuntimeError as e:
        print(e)


# When the devices=none is labelled, it will use all available CPU and GPUs, otherwise devices=["/gpu:0", "/gpu:1"] will force to use teh first 2 GPUs found.
strategy = tf.distribute.MirroredStrategy(
    devices=["/gpu:0", "/gpu:1"],
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(),
)

with strategy.scope():
    # Build U-net model

    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # Contraction path - encoding
    c1 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = tf.keras.layers.Dropout(0.5)(c4)
    c4 = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(
        1024, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = tf.keras.layers.Dropout(0.5)(c5)
    c5 = tf.keras.layers.Conv2D(
        1024, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)

    # Expansive path -encoding
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(
        c5
    )
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = tf.keras.layers.Dropout(0.5)(c6)
    c6 = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(
        c6
    )
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(
        c7
    )
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])

    c9 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)

    c9 = tf.keras.layers.Dropout(0.1)(c9)

    c9 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    c10 = tf.keras.layers.Conv2D(
        2, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c10)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[dice_coef])

    # model.summary()
    earlystopper = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto"
    )
    checkpointer = ModelCheckpoint(
        "naic_dice_coef_bs20.h5", verbose=1, save_best_only=True, mode="min"
    )
    results = model.fit(
        X_train,
        Y_train,
        validation_split=0.1,
        batch_size=20,
        epochs=10,
        callbacks=[earlystopper, checkpointer],
    )
