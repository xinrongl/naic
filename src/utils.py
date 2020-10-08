import cv2


def cv2_imread(filename, mode):
    # cv2.imread wrapper
    assert mode in ["image", "mask"]
    filename = str(filename)
    if mode == "image":
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(filename, -1)
    return image


def hist_equalizer(img):
    # local hist equalize
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    channels = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[2], channels[1])

    cv2.merge(channels, hsv)
    return img