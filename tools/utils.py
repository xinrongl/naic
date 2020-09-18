import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def zh_ch(string):
    return string.encode("gbk").decode(errors="ignore")


def cv2_imshow(image):
    cv2.imshow("", image)


def get_mask(image, cls):
    return (image == cls).astype(float) * 255


# def plot_mask(mask, image=None):
#     id2cls = {100: 'water',
#               200: 'traffic',
#               300: 'building',
#               400: 'cropland',
#               500: 'grassland',
#               600: 'forest',
#               700: 'bare soil',
#               800: 'other'}
#     cls = sorted(set(mask.flatten()))
#     ncol = len(cls) + 1 if image is not None else len(cls)
#     fig, axs = plt.subplots(1, ncol, figsize=(4 * ncol, 5))
#     if image is not None:
#         axs[0].imshow(image)
#         axs[0].title.set_text("mask")
#         for i in range(ncol - 1):
#             axs[i + 1].imshow(get_mask(mask, cls[i]))
#             axs[i + 1].title.set_text(id2cls[cls[i]])
#     else:
#         for i in range(ncol):
#             axs[i].imshow(get_mask(mask, cls[i]))
#             axs[i].title.set_text(id2cls[cls[i]])
#     plt.tight_layout()
#     plt.show()


def plot_mask(mask, image=None):
    palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0]]
    classes = [
        'water', 'traffic', 'building', 'cropland', 'grassland', 'forest', 'bare soil', 'other'
    ]

    palette_dict = dict(zip(range(100, 801, 100), palette))
    label_dict = dict(zip(range(100, 801, 100), classes))
    mask_p = np.zeros((256, 256, 3))
    patches = []
    for label, color in palette_dict.items():
        mask_p[mask == label, :] = color
        patches.append(mpatches.Patch(facecolor="#{0:02x}{1:02x}{2:02x}".format(color[0], color[1], color[2]),
                                      label=label_dict[label],
                                      linewidth=2,
                                      edgecolor='black'))
    mask_p = mask_p.astype(np.uint8)

    if image is not None:
        ncol = 2
        fig, axs = plt.subplots(1, ncol, figsize=(4 * ncol, 5))
        axs[0].imshow(image)
        axs[0].title.set_text(f"image")
        axs[1].imshow(mask_p)
        axs[1].title.set_text(f"mask")
    else:
        ncol = 1
        fig, axs = plt.subplots(1, ncol, figsize=(4 * ncol, 5))
        axs.imshow(mask_p)
        axs.title.set_text("mask")
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
