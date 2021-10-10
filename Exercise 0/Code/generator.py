"""
@description

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import os.path
import json
import scipy.misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    """
    This class creates an image generator.
    Generator objects in python are defined as having a next function.
    This next function returns the next generated object.
    In our case it returns the input of a neural network each time it gets called. 
    This input consists of a batch of images and its corresponding labels.
    """

    def __init__(self, file_path: str, label_path: str, batch_size: int, image_size: [int, int, int], rotation: bool=False, mirroring: bool=False, shuffle: bool=False):
        """
        Constructor.
        
        :param file_path: Path to the directory containing all images.
        :param label_path: Path to the JSON file containing the labels.
        :param batch_size: Number of images in a batch.
        :param image_size: list of [height, width, channel] defining the desired image size.
        :param rotation: If the rotation flag is True, randomly rotate the images by 90, 180 or 270° in the method "next()".
        :param mirroring: If the mirroring flag is True, randomly mirror the images in the method "next()".
        :param shuffle: If the shuffle flag is True, the order in which the images appear is random.
        """

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.index = 0


    def next(self):
        """
        This function creates a batch of images and corresponding labels and returns them.
        In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        Note that your amount of total data might not be divisible without remainder with the batch_size.
        Think about how to handle such cases
        """

        list_of_all_images = os.listdir(self.file_path)

        with open(self.label_path) as json_file:
            dictionary_content = json.load(json_file)

        batch = []
        labels = []

        if self.shuffle:
            np.random.shuffle(list_of_all_images)

        for number in range(self.batch_size):
            image_number = list_of_all_images[self.index].split('.')[0]
            image_label = dictionary_content[image_number]
            image = np.load(os.path.join(self.file_path, list_of_all_images[self.index]))

            if self.index == (len(list_of_all_images) - 1):
                self.index = 0
            else:
                self.index = self.index + 1

            #image = scipy.misc.imresize(image, self.image_size, interp='bilinear', mode=None)
            image = np.array(Image.fromarray(image).resize((self.image_size[0], self.image_size[1])))

            self.augment(image)

            image = np.expand_dims(image, axis=0)

            if batch == []:
                batch = image
            else:
                batch = np.concatenate((batch, image), axis=0)
                # image_stack = np.squeeze(np.stack((image_stack, image), axis=1))

            labels = np.append(labels, image_label)

        return batch, labels


    def augment(self, img):
        """
        This function takes a single image as an input and performs a random transformation
        (mirroring and/or rotation) on it and outputs the transformed image.
        
        :param img: Respective image.
        """

        if self.mirroring:
            mirror = np.random.randint(2, size=1)
            if mirror == 0:
                img = np.fliplr(img)

        if self.rotation:
            rotate = np.random.randint(2, size=1)
            if rotate == 0:
                num_rot = 1 + np.random.randint(3, size=1)
                for i in range(num_rot[0]):
                    img = np.rot90(img)

        return img


    def class_name(self, x):
        """
        This function returns the class name for a specific input label "x"
        
        :param x: Specific label.
        :return: Class name corresponding to input label.
        """

        if x == 0:
            class_name = 'airplane'
        elif x == 1:
            class_name = 'automobile'
        elif x == 2:
            class_name = 'bird'
        elif x == 3:
            class_name = 'cat'
        elif x == 4:
            class_name = 'deer'
        elif x == 5:
            class_name = 'dog'
        elif x == 6:
            class_name = 'frog'
        elif x == 7:
            class_name = 'horse'
        elif x == 8:
            class_name = 'ship'
        elif x == 9:
            class_name = 'truck'
        else:
            class_name = None

        return class_name


    def show(self):
        """
        In order to verify that the generator creates batches as required, this functions calls next to get a batch of
        images and labels and visualizes it.
        """

        image_batch = self.next()
        row = math.ceil(self.batch_size / 3)
        column = 3

        images = image_batch[0]
        labels = image_batch[1]
        fig = plt.figure()

        for i in range(self.batch_size):
            img = images[i]
            lbl_num = labels[i]
            lbl_name = self.class_name(lbl_num)
            fig.add_subplot(row, column, i + 1, title=lbl_name)
            plt.imshow(img)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

