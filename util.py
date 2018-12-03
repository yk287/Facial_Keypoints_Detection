

import torch
import os
import numpy as np
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import RandomHorizontalFlip
import torchvision.transforms as transforms

import cv2


import random

class FaceLandmarksDataset(Dataset):
    '''
    Fac Landmarks Dataset
    Inherits Dataset class and overrides 2 methods

    __len__ and __getitem__
    '''


    def __init__(self, csv_file, root_dir, transform=None):
        '''

        :param csv_file: Path to the csv file with annotations
        :param root_dir: Directory with all the images
        :param transform: Optional transform to be applied on a sample
        '''

        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            '''if transform is given transform the data'''
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


class HorizontalFlip(object):
    """Flips the image and the landmarks left to right with a given probability"""
    def __init__(self, p=0.5):
        assert isinstance(p, (int, float))
        self.p = p

    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]

        if random.random() < self.p:

            image = cv2.flip(image, 1)
            landmarks[:,0] = w - landmarks[:,0]

        return {'image': image,
                'landmarks': landmarks}

class VerticalFlip(object):
    """Flips the image and the landmarks over with a given probability"""
    def __init__(self, p=0.5):
        assert isinstance(p, (int, float))
        self.p = p

    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]

        if random.random() < self.p:

            image = cv2.flip(image, 0)
            landmarks[:, 1] = h - landmarks[:, 1]

        return {'image': image,
                'landmarks': landmarks}


class RotateScale(object):
    '''Rotate and scale up the image Using OpenCV's warpAffine() '''

    def __init__(self, angle=45, scale =1):
        '''

        :param angle: rotation angle
        :param scale: how much to scale up the image
        '''

        self.angle = angle
        self.scale = scale

    def __call__(self, sample):

        image, landmarks = sample['image'], sample['landmarks']

        cols, rows = image.shape[:2]

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.angle, self.scale)
        image = cv2.warpAffine(image, M, (cols, rows))

        landmarks = np.matmul((np.array(np.concatenate((landmarks, np.ones(len(landmarks))[:, None]), axis=1))), M.transpose())

        return {'image': image,
                'landmarks': landmarks}






