from torchvision import transforms


__all__ = ['Tilt']


class Tilt(object):
    """ Tilts an image at an angle """

    def __init__(self, angle=45):
        self.angle = angle

    def __call__(self, img):
        # Convert to PIL Image for rotation
        img_pil = transforms.ToPILImage()(img)
        # Rotate using PIL
        img_rotated = transforms.functional.rotate(img_pil, self.angle)
        # Convert back to PyTorch tensor
        return transforms.ToTensor()(img_rotated)
        # return transforms.ToTensor()(img_pil)