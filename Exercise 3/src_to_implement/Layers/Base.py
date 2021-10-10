"""
@description
Since the behaviour of some layers changes depending on whether the network is currently training or testing,
we need to refactor our layers.
Moreover, we choose to introduce a "base-optimizer", which provides some basic functionality,
in order to enable the use of regularizers.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import numpy as np


class BaseLayer:
    def __init__(self, testing_phase=False):
        """
        Constructor for the class "Base Layer", which acts like the parent class for all layers of the network.

        :param testing_phase: Define if the testing or the test phase is currently executed.
        """

        # Testing or training phase
        self.testing_phase = testing_phase

        # Default weights for each layer, to ease the loss calculation.
        self.weights = np.zeros(1)


