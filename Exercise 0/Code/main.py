"""
@description

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import os

from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator

if __name__ == "__main__":
    """
    checker = Checker(8, 2)
    checker.draw()
    checker.show()

    circle = Circle(255, 50, (100, 100))
    circle.draw()
    circle.show()

    spectrum = Spectrum(2550)
    spectrum.draw()
    spectrum.show()
    """

    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), r'Data/exercise_data/')
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), r'Data/Labels.json')

    gen = ImageGenerator(file_path, json_path, 12, [30, 30, 3], rotation=True, mirroring=True, shuffle=True)
    gen.next()
    gen.show()

