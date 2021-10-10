import torch as t
from trainer import Trainer
import model
import sys
import torchvision as tv
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from pathlib import Path

epoch = 20
#TODO: Enter your model here
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 7
EPOCHS = 20

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO

# create an instance of our ResNet model
# TODO
# ======================================================================================================================
# Step 4: Create an instance of the ResNet model.
# ======================================================================================================================
PATH_DATA_CSV = Path(__file__).parent.absolute() / 'data.csv'  # Path to the csv-file containing the data.
complete_dataset = pd.read_csv(PATH_DATA_CSV, sep=";")

# ======================================================================================================================
# Step 2: Split the data into a train- and test-set.
# ======================================================================================================================
raw_train_set, raw_test_set = train_test_split(complete_dataset, test_size=0.1)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
# ======================================================================================================================
# Step 3: Load, Preprocess and augment the train as well as test set using the class "ChallengeDataset".
# ======================================================================================================================
train_set = ChallengeDataset(raw_train_set, "train")
test_set = ChallengeDataset(raw_test_set, "val")

resnet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
# ======================================================================================================================
# Step 5: Set up a suitable loss criterion.
# ======================================================================================================================
loss_criterion = F.binary_cross_entropy_with_logits
loss_criterion = F.binary_cross_entropy

# ======================================================================================================================
# Step 6: Set up the optimizer.
# ======================================================================================================================
optimizer = t.optim.Adam(resnet.parameters(), lr=LEARNING_RATE)

# ======================================================================================================================
# Step : Create an instance of the Trainer.
# ======================================================================================================================
trainer = Trainer(resnet, loss_criterion, optim=optimizer, train_dl=train_set, val_test_dl=test_set, cuda=True,
                  early_stopping_patience=EARLY_STOPPING_PATIENCE)


trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
