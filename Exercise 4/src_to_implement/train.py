import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
import torch.nn.functional as F

# ======================================================================================================================
# Step 0: Settings.
# ======================================================================================================================
PATH_DATA_CSV = Path(__file__).parent.absolute() / 'data.csv'  # Path to the csv-file containing the data.
# BATCH_SIZE = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 7
EPOCHS = 100

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
# ======================================================================================================================
# Step 1: Read the csv-file containing the data
# ======================================================================================================================
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

print('Dataset split into {:d} training and {:d} validation samples'.format(len(train_set), len(test_set)))

# create an instance of our ResNet model
# TODO
# ======================================================================================================================
# Step 4: Create an instance of the ResNet model.
# ======================================================================================================================
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

# go, go, go... call fit on trainer
# TODO
res = trainer.fit(EPOCHS)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')