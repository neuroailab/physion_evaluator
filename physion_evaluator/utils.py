IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_tuple):
        tensor, label = tensor_tuple
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return (tensor, label)


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple

        if img_group[0].mode == 'L':
            return (np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return (np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label)
            else:
                return (np.concatenate(img_group, axis=2), label)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic_tuple):
        pic, label = pic_tuple

        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return (img.float().div(255.) if self.div else img.float(), label)


class IdentityTransform(object):

    def __call__(self, data):
        return data


class DataAugmentationForVideoMAE(object):
    def __init__(self,
                 imagenet_normalize=True,
                 rescale_size=None):

        transform_list = []

        if rescale_size is not None:
            self.rescale = GroupScale(rescale_size)
            transform_list.append(self.rescale)

        transform_list.extend([Stack(roll=False),
                               ToTorchFormatTensor(div=True)])

        if imagenet_normalize:
            self.input_mean = IMAGENET_DEFAULT_MEAN
            self.input_std = IMAGENET_DEFAULT_STD
            normalize = GroupNormalize(self.input_mean, self.input_std)
            transform_list.append(normalize)

        self.transform = transforms.Compose(transform_list)

    def __call__(self, images):

        images = (images, None)

        process_data, _ = self.transform(images)

        return process_data
