"""
Augmentation Functions for DataLoader.
"""

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random

from torchvision.transforms import transforms


class ChannelShuffle(object):
    """
    Shuffles rgb channels randomly.
    """
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: image with randomly shuffled channels
        """
        rgb = img.split()

        rgb = list(rgb)
        random.shuffle(rgb)
        return Image.merge(img.mode, rgb)


class GaussianBlur(object):
    """
    Gaussian blur.
    """
    def __init__(self, radius=1):
        """
        Args:
            radius (int or list): if int -> fixed raduis for gaussian blur, if list -> random selection of list elements for gaussian blur radius
        """
        self.radius = radius 
        
    def __call__(self, img):
        if isinstance(self.radius,list):
            _rad = random.randint(*self.radius)
        else:
            _rad = self.radius

        return img.filter(ImageFilter.GaussianBlur(radius=_rad))
    
    def __repr__(self):
        return self.__class__.__name__ + '(rad={0})'.format(self.radius)


class RandomNoise(object):
    """
    Adds random pixel noise to all channels.
    """
    def __init__(self, prob=0., val=1.):
        """
        Args:
            prob (float): pixel-wise probability for noise addition
            val (float): noise value
        """
        self._prob = prob
        self._val = val

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: noisy image
        """
        img = np.array(img)

        rnd = np.random.rand(img.shape[0],img.shape[1])

        img[rnd<self._prob] = self._val

        return Image.fromarray(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'prob={0}'.format(self._prob)
        format_string += ', val={0})'.format(self._val)
        return format_string


class SaltAndPepperNoise(object):
    """
    Salt and pepper noise.
    """
    def __init__(self, prob=0., upper_val=255, lower_val=0):
        self._prob = prob
        self._upper_val = upper_val
        self._lower_val = lower_val

    @staticmethod
    def get_params(prob, upper_val, lower_val):
        """
        Get a randomized salt and pepper noise
        Args:
            prob (float): probability
            upper_val (int): salt value for image (usually high; default=255)
            lower_val (int): pepper value for image (usually low; default=0)

        Returns:
            Transform which randomly adds salt and pepper noise
        """
        salt_noise = RandomNoise(prob/2, upper_val)
        pepper_noise = RandomNoise(prob/2, lower_val)

        augm = []
        
        augm.append(transforms.Lambda(lambda img: salt_noise(img)))
        augm.append(transforms.Lambda(lambda img: pepper_noise(img)))

        random.shuffle(augm)
        augm = transforms.Compose(augm)

        return augm

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: noisy image
        """
        transform = self.get_params(self._prob, self._upper_val,
                                    self._lower_val)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'prob={0}'.format(self._prob)
        format_string += ', upper_val={0}'.format(self._upper_val)
        format_string += ', lower_val={0})'.format(self._lower_val)
        return format_string


class AugmentationPIL:
    """
    Abstract class for PIL Augmentations.
    """
    def __init__(self, pillow_fct, factor_interval):
        """
        Args:
            pillow_fct: PIL ImageEnhance classes
            factor_interval: tuple of value interval
        """
        self.pillow_fct = pillow_fct
        self.factor_interval = factor_interval

    def __call__(self, img):
        return self.pillow_fct(img).enhance(factor=random.uniform(*self.factor_interval))


class SharpnessAugmentation(AugmentationPIL):
    def __init__(self, factor_interval=(0., 50.)):
        super().__init__(pillow_fct=ImageEnhance.Sharpness, factor_interval=factor_interval)


class ContrastAugmentation(AugmentationPIL):
    def __init__(self, factor_interval=(0.2, 50.)):
        super().__init__(pillow_fct=ImageEnhance.Contrast, factor_interval=factor_interval)


class BrightnessAugmentation(AugmentationPIL):
    def __init__(self, factor_interval=(0.1, 6.0)):
        super().__init__(pillow_fct=ImageEnhance.Brightness, factor_interval=factor_interval)


class ColorAugmentation(AugmentationPIL):
    """
    This augmentation adjusts the color balance of the image.
    An enhancement factor of 0.0 gives a black and white image. A factor of 1.0 gives the original image.
    """
    def __init__(self, factor_interval=(0.0, 1.0)):
        super().__init__(pillow_fct=ImageEnhance.Color, factor_interval=factor_interval)
