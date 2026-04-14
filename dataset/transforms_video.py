import PIL
import torch
import torchvision.transforms.functional as F
from numpy import random as rand
from PIL import Image



class Check(object):
    def __init__(self,):
        pass
    def __call__(self,  img, flow, target):
        # check if edge still exist after transforms
        if "edge_maps" in target:
            if torch.count_nonzero(target['edge_maps'][:, 0, :, :].flatten()) == torch.tensor(0):
                keep = False
            else:
                keep = True

        target['valid'] = keep

        return img, flow, target


def crop_ovis(clip, flow, target, size):
    w, h = clip[0].size
    pad_w = 0
    pad_h = 0
    if w < size or h < size:
        pad_w = size-w if size-w > 0 else 0
        pad_h = size-h if size-h > 0 else 0
        clip, target = pad(clip, target, (pad_w, pad_h))

    w, h = clip[0].size     # refresh w and h if padded
    crop_i = rand.randint(h-size) if pad_h == 0 else 0
    crop_j = rand.randint(w-size) if pad_w == 0 else 0
    crop_w = size
    crop_h = size

    cropped_image = []
    for image in clip:
        cropped_image.append(F.crop(image, crop_i, crop_j, crop_h, crop_w))     # i->top j->left

    cropped_flow = []
    for f in flow:
        cropped_flow.append(F.crop(f, crop_i, crop_j, crop_h, crop_w))  # i->top j->left

    target = target.copy()

    # should we do something wrt the original size?
    target["size"] = torch.tensor([crop_h, crop_w])

    fields = ["labels"]

    if "edge_maps" in target:
        target['edge_maps'] = target['edge_maps'][:, :, crop_i:crop_i + crop_h, crop_j:crop_j + crop_w]
        fields.append("edge_maps")

    return cropped_image, cropped_flow, target

def pad_ovis(img, size):
    w, h = img.size
    if w < size or h < size:
        pad_w = size - w if size - w > 0 else 0
        pad_h = size - h if size - h > 0 else 0
        img = pad(img, None, (pad_w, pad_h))

    return img


def pad(clip, target, padding):
    # assumes that we only pad on the bottom right corners
    if type(clip) == PIL.Image.Image:
        padded_img = F.pad(clip, (0, 0, padding[0], padding[1]))
        return padded_img

    else:
        padded_image = []
        for image in clip:
            padded_image.append(F.pad(image, (0, 0, padding[0], padding[1])))
        if target is None:
            return padded_image, None
        target = target.copy()
        target["size"] = torch.tensor(padded_image[0].size[::-1])
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
        if "edge_maps" in target:
            target['edge_maps'] = torch.nn.functional.pad(target['edge_maps'], (0, padding[0], 0, padding[1]))
        return padded_image, target



class ToTensor(object):
    def __call__(self, clip, flow, target):
        img = []
        for im in clip:
            img.append(F.to_tensor(im))
        flow_list = []
        for f in flow:
            flow_list.append(F.to_tensor(f))
        return img, flow_list, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip, flow, target=None):
        image = []
        for im in clip:
            image.append(F.normalize(im, mean=self.mean, std=self.std))
        flow_list = []
        for f in flow:
            flow_list.append(F.normalize(f, mean=self.mean, std=self.std))
        if target is None:
            return image, None
        target = target.copy()
        h, w = image[0].shape[-2:]
        return image, flow_list, target


class CropOVIS(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, flow, target):
        return crop_ovis(img, flow, target, self.size)

class PadOVIS(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return pad_ovis(img, self.size)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, flow, target):
        for t in self.transforms:
            image, flow, target = t(image, flow, target)
        return image, flow, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
