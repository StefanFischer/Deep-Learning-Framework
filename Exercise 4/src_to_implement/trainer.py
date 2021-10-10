"""
@description
The training process consists of alternating between training for one epoch on the training dataset (training step) and
then assessing the performance on the validation dataset (validation step).
After that, a decision is made if the training process should be continued.
A common stopping criterion is called EarlyStopping with the following behaviour:
If the validation loss does not decrease after a specified number of epochs, then the training process will be stopped.
This criterion will be used in our implementation.

@version
python 3

@author
Stefan Fischer
Sebastian Doerrich
"""

import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

import operator
import numpy as np


class Trainer:

    def __init__(self, model, crit, optim=None, train_dl=None, val_test_dl=None, cuda=True, early_stopping_patience=-1):
        """
        :param model: Model to be trained.
        :param crit: Loss function.
        :param optim: Optimizer.
        :param train_dl: Training data set.
        :param val_test_dl: Validation (or test) data set.
        :param cuda: Whether to use the GPU.
        :param early_stopping_patience: The patience for early stopping.
        """

        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            #self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})


    def train_step(self, x, y):
        """
        Execute one step of the training.

        :param x: Batch of Input Images.
        :param y: Labels.
        :return: Loss.
        """

        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        # TODO

        # ==============================================================================================================
        # Step 1: Reset the gradients.
        #           - Needed, because gradients are beeing accumulated with every backpropagation step instead of beeing
        #             replaced.
        # ==============================================================================================================
        if self._optim is not None:
            self._optim.zero_grad()

        # ==============================================================================================================
        # Step 2: Propagate through the network.
        # ==============================================================================================================
        predictions = self._model(x)

        # ==============================================================================================================
        # Step 3: Calculate the loss.
        # ==============================================================================================================
        loss = self._crit(predictions, y)

        # ==============================================================================================================
        # Step 4: Compute gradient by backward propagation
        # ==============================================================================================================
        loss.backward()

        # ==============================================================================================================
        # Step 5: Update weights.
        # ==============================================================================================================
        if self._optim is not None:
            self._optim.step()
            # self._optim(self._model.parameters())

        # ==============================================================================================================
        # Step 6: Return the loss.
        # ==============================================================================================================
        return loss

    def val_test_step(self, x, y):
        """
        Execute one step of the validation.

        :param x: Validation sample.
        :param y: Respective Label.
        :return: Loss and Predictions.
        """

        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        # TODO

        # ==============================================================================================================
        # Step 1: Propagate through the network and calculate predictions.
        # ==============================================================================================================
        predictions = self._model(x)

        # ==============================================================================================================
        # Step 2: Calculate the loss.
        # ==============================================================================================================
        loss = self._crit(predictions, y)

        # ==============================================================================================================
        # Step 3: Return the loss and the predictions.
        # ==============================================================================================================
        return loss, predictions

    def train_epoch(self):
        """
        Execute the training for one epoch.

        :return: Average loss for epoch.
        """

        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        # TODO

        # ==============================================================================================================
        # Step 1: Set to training mode.
        # ==============================================================================================================
        self._train_dl.mode = "train"

        # ==============================================================================================================
        # Step 2: Create a list containing the batch separation indices of the training set
        #           - Batch size is chosen to be 100 here.
        # ==============================================================================================================
        # trainloader = t.utils.data.BatchSampler(t.utils.data.RandomSampler(range(len(self._train_dl))), 100, False)
        trainloader = t.utils.data.DataLoader(self._train_dl, batch_size=8, shuffle=True)

        # ==============================================================================================================
        # Step 3: Iterate batchwise through the training set calculate the loss for each batch.
        # ==============================================================================================================
        total_loss = 0

        # for index in tqdm(trainloader):
        for batch_id, (data, label) in enumerate(trainloader):
        #for index, batch in tqdm(enumerate(trainloader, 0)):
            # ==========================================================================================================
            # Step 3.1: Extract the current batch.
            # ==========================================================================================================
            # batch = operator.itemgetter(*index)(self._train_dl)

            # ==========================================================================================================
            # Step 3.2: Transfer the batch to "cuda()" -> the gpu if a gpu is given.
            # ==========================================================================================================
            if self._cuda:
                data = data.cuda()
                label = label.cuda()


            # ==========================================================================================================
            # Step 3.3: Collect all images and their respective labels of the current batch.
            # ==========================================================================================================

            # ==========================================================================================================
            # Step 3.4: Calculate the loss of the current batch and add it to the total_loss of the epoch.
            # ==========================================================================================================
            data = t.autograd.Variable(data)
            label = t.autograd.Variable(label)
            loss = self.train_step(data, label)

            if self._cuda:
                loss = loss.cpu()

            total_loss = total_loss + loss

        # ==============================================================================================================
        # Step 4: Calculate the average loss of the epoch.
        # ==============================================================================================================
        average_loss_epoch = total_loss / len(trainloader)
        print("average loss of epoch: " +str(average_loss_epoch))

        # ==============================================================================================================
        # Step 5: Return the average loss of the epoch.
        # ==============================================================================================================
        return average_loss_epoch

    def val_test(self):
        """
        Execute the validation.

        :return: Average loss of the validation.
        """

        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        # TODO

        # ==============================================================================================================
        # Step 1: Set to validation mode.
        # ==============================================================================================================
        self._train_dl.mode = "val"

        # ==============================================================================================================
        # Step 2: Disable gradient computation.
        # ==============================================================================================================
        with t.no_grad():
            # ==========================================================================================================
            # Step 3: Create a list containing the batch separation indices of the validation set
            #           - Batch size is chosen to be 100 here.
            # ==========================================================================================================
            # valloader = t.utils.data.BatchSampler(t.utils.data.RandomSampler(range(len(self._val_test_dl))), 100, False)
            valloader = t.utils.data.DataLoader(self._val_test_dl, batch_size=8, shuffle=True)

            # ==========================================================================================================
            # Step 4: Iterate batchwise through the validation set
            # ==========================================================================================================
            total_loss = 0
            labels = []
            predictions = []

            total_labels = 0
            correct_labels = 0
            f1 = 0
            i = 0

            # for index in tqdm(batch_sampler):
            for batch_id, (data, label) in enumerate(valloader):
            #for index, batch in tqdm(enumerate(valloader, 0)):
                # ======================================================================================================
                # Step 4.1: Extract the current batch.
                # ======================================================================================================
                # batch = operator.itemgetter(*index)(self._train_dl)

                # ======================================================================================================
                # Step 4.2: Transfer the batch to "cuda()" -> the gpu if a gpu is given.
                # ======================================================================================================
                if self._cuda:
                    data = data.cuda()

                    label = label.cuda()

                # ======================================================================================================
                # Step 4.3: Collect all images and their respective labels of the current batch.
                # ======================================================================================================


                # ======================================================================================================
                # Step 4.4: Perform a validation step and calculate the loss as well as the predictions.
                # ======================================================================================================
                data = t.autograd.Variable(data)
                label = t.autograd.Variable(label)
                loss, preds = self.val_test_step(data, label)
                if self._cuda:
                    loss = loss.cpu()
                    preds = preds.cpu()
                    label = label.cpu()
                # ======================================================================================================
                # Step 4.5: Save the predictions and the labels for the current batch.
                # ======================================================================================================

                # ======================================================================================================
                # Step 4.6: Calculate the loss of the current batch and add it to the total loss.
                # ======================================================================================================
                total_loss += loss

                # ======================================================================================================
                # Step 4.7: Store the number of labels and the number of correctly labeled samples.
                # ======================================================================================================
                total_labels += label.numel()
                predsFloat = t.FloatTensor(preds)
                predsToOne = t.where(predsFloat > 0.5, 1, 0)#
                #correct_label = t.where(predsToOne == label, 1, 0)
                #correct_labels += correct_label.sum()

                correct_label = t.sum((predsToOne == label))
                correct_labels += correct_label
                # can not work as preds-el are never

                f1 += f1_score(y_true=label, y_pred=predsToOne, average="weighted")
                i+=1
        # ==============================================================================================================
        # Step 5: Calculate the average loss.
        # ==============================================================================================================
        average_loss = total_loss / len(valloader)

        # ==============================================================================================================
        # Step : Return the average loss and print the chosen metric
        # ==============================================================================================================
        f1 = f1 / i
        print('Accuracy of the network on the test images: %d %%' % (100 * correct_labels / total_labels))
        #
        print('F1-score of the network on the test images: ' + str(f1))
        return average_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        # TODO

        training_losses = []
        validation_losses = []
        epoch_counter = 0

        while True:

            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            # TODO
            # ==========================================================================================================
            # Step : Stop if number of epochs is reached.
            # ==========================================================================================================
            if epoch_counter >= epochs:
                self.save_checkpoint(epoch_counter)
                break

            # ==========================================================================================================
            # Step : Train for an epoch and store the train loss.
            # ==========================================================================================================
            training_loss = self.train_epoch()
            training_losses.append(training_loss)

            # ==========================================================================================================
            # Step : Calculate the loss and metrics on the validation set and store the loss.
            # ==========================================================================================================
            validation_loss = self.val_test()
            validation_losses.append(validation_loss)

            # ==========================================================================================================
            # Step : Use the save_checkpoint function to save the model.
            # ==========================================================================================================
            self.save_checkpoint(epoch_counter)

            # ==========================================================================================================
            # Step : Check whether early stopping should be performed.
            # ==========================================================================================================
            if epoch_counter == self._early_stopping_patience:
                if validation_losses[-1] >= validation_losses[0]:
                    break

            epoch_counter += 1
        # ==============================================================================================================
        # Step : Return the losses for both training and validation.
        # ==============================================================================================================
        return training_losses, validation_losses
