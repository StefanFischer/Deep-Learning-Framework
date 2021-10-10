"""
@description
Classes to create different types of patterns as numpy.ndarrays.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution: int, tile_size: int):
        """
        Create a checkerboard object.
        
        :param resolution: Number of pixels in each dimension.
        :param tile_size: Number of pixel an individual tile has in each dimension.
        """

        self.resolution = resolution
        self.tile_size = tile_size

        self.output = None  # type = np.ndarray


    def draw(self):
        """
        Create a checkerboard pattern, where the tile in the top left corner is black, as a numpy array.
        """

        # In order to avoid truncated checkerboard patterns, the code only allows values for resolution that are evenly
        # dividable by 2 * tile_size
        if (self.resolution % (2 * self.tile_size)) == 0:
            # Create tile_size consecutive array entries with the same color [1., ..., 1.] and [0., ..., 0.]
            black_pixels = np.zeros(self.tile_size)
            white_pixels = np.ones(self.tile_size)

            # Create black-white and white-black consecutive array entries, e.g. [1., 1., 0., 0.] and [0., 0., 1., 1.]
            consec_pixels_black_white = np.concatenate((black_pixels, white_pixels))
            consec_pixels_white_black = np.concatenate((white_pixels, black_pixels))

            # Create a black-white and white-black board row
            # [[1., 1., 0., 0., ..., 1., 1., 0., 0.]                    [[0., 0., 1., 1., ..., 0., 0., 1., 1.]
            #                   ...                         and                           ...
            #  [1., 1., 0., 0., ..., 1., 1., 0., 0.]]                    [0., 0., 1., 1., ..., 0., 0., 1., 1.]]
            number_array_entries = int(self.resolution / (2 * self.tile_size))
            board_row_black_start = np.tile(consec_pixels_black_white, (number_array_entries, number_array_entries))
            board_row_white_start = np.tile(consec_pixels_white_black, (number_array_entries, number_array_entries))

            # Create the whole checkerboard pattern
            board_rows_black_white = np.concatenate((board_row_black_start, board_row_white_start), axis=0)
            checkerboard = np.tile(board_rows_black_white, (number_array_entries, 1))

            self.output = checkerboard

        else:
            raise SystemExit('ERROR: Creation with used resolution and tile_size not feasible')


    def show(self):
        """
        Show the checkerboard pattern.
        """

        plt.imshow(self.output, cmap='gray')
        plt.show()



class Circle:
    def __init__(self, resolution: int, radius: int, position: tuple):
        """
        Create a circle pattern object.
        
        :param resolution: Number of pixels in each dimension.
        :param radius: Radius of the circle.
        :param position: Position of the circle center in the image.
        """

        self.resolution = resolution
        self.radius = radius
        self.position = position

        self.output = None  # type = np.ndarray


    def draw(self):
        """
        Implement a binary circle with a given radius at a specified position in the image.
        """

        pattern = np.zeros((self.resolution, self.resolution))

        x_center, y_center = self.position

        for iy in range(self.resolution):
            for ix in range(self.resolution):
                if np.sqrt((ix - x_center) ** 2 + (iy - y_center) ** 2) <= self.radius:
                    pattern[iy, ix] = 1

        self.output = pattern


    def show(self):
        """
        Show the circle pattern.
        """

        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:

    def __init__(self, resolution):
        """
        Create a Spectrum pattern object.

        :param resolution: Number of pixels in each dimension.
        """

        self.resolution = resolution
        self.output = None

    def draw(self):
        """
        Implement an RGB color spectrum.
        """

        red = self.resolution * [(255, 0, 0)]
        yellow = (255, 255, 0)
        blue = (0, 0, 255)
        cyan = (0, 255, 255)

        diff_list = np.linspace(0, 255, num=self.resolution)
        diff_list = np.expand_dims(diff_list, axis=1)

        redGradient1D = list(np.array(red) - np.array(diff_list))
        redGradient1D = np.array(redGradient1D).clip(min=0)
        redGradient2D = np.expand_dims(redGradient1D, axis=0)
        redGradient2D = np.repeat(redGradient2D, self.resolution, axis=0)
        redGradient2D = redGradient2D + np.rot90(redGradient2D)
        redGradient2D = np.rot90(redGradient2D, 2)

        yellowGradient1D = list(np.array(yellow) - np.array(diff_list))
        yellowGradient1D = np.array(yellowGradient1D).clip(min=0)
        yellowGradient2D = np.expand_dims(yellowGradient1D, axis=0)
        yellowGradient2D = np.repeat(yellowGradient2D, self.resolution, axis=0)
        yellowGradient2D = yellowGradient2D + np.rot90(yellowGradient2D)
        yellowGradient2D = np.rot90(yellowGradient2D, 1)

        blueGradient1D = list(np.array(blue) - np.array(diff_list))
        blueGradient1D = np.array(blueGradient1D).clip(min=0)
        blueGradient2D = np.expand_dims(blueGradient1D, axis=0)
        blueGradient2D = np.repeat(blueGradient2D, self.resolution, axis=0)
        blueGradient2D = blueGradient2D + np.rot90(blueGradient2D)
        blueGradient2D = np.rot90(blueGradient2D, 3)

        cyanGradient1D = list(np.array(cyan) - np.array(diff_list))
        cyanGradient1D = np.array(cyanGradient1D).clip(min=0)
        cyanGradient2D = np.expand_dims(cyanGradient1D, axis=0)
        cyanGradient2D = np.repeat(cyanGradient2D, self.resolution, axis=0)
        cyanGradient2D = cyanGradient2D + np.rot90(cyanGradient2D)

        result = redGradient2D + yellowGradient2D + cyanGradient2D + blueGradient2D
        self.output = (result - result.min()) / (result.max() - result.min())

        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()
