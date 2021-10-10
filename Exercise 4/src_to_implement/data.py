"""
@description
When you start working on a new machine learning problem, you have to deal with the format of the given data collection
and implement a pipeline to load, preprocess and augment the data.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    """
    Solar modules dataset.
    """

    def __init__(self, data, mode):
        """
        :param data (pandas.dataframe): Pandas dataframe of the .csv-file storing the data.
        :param mode (string): Specify the mode, either training (train) or validation (val).
        """

        # Store the data found in the "data.csv"-file
        self.data = data

        # Store the current mode
        assert mode == "train" or mode == "val"
        self.mode = mode

        # Create composition of several transforms since among other aspects, this is interesting for data augmentation.
        #   - toPILImage() is used, because torchvision transformations only work on PIL.Images not on numpy arrays.
        # ToDO: Two different transformations depending on train or validation mode
        if self.mode == "train":
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(train_mean, train_std)])
        else:
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(train_mean, train_std)])

    def __len__(self):
        """
        Override parent method __len__() to return the size of the dataset.

        :return: Size of dataset.
        """

        return len(self.data)

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """

        # Extract current image and convert it to a color image
        image_path = Path(__file__).parent.absolute().joinpath(self.data.iloc[index, 0])
        image = imread(image_path)
        image = gray2rgb(image)

        # Extract the image labels
        labels = self.data.iloc[index, 1:]
        labels = torch.FloatTensor(labels)

        # Create the respective sample and execute the specified transformations on the image
        sample = (self._transform(image), labels)

        return sample



