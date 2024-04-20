from skimage import img_as_ubyte, color


__all__ = ['image_preprocess']


def image_preprocess(img):
    img = img_as_ubyte(color.rgb2gray(img))

    return img
