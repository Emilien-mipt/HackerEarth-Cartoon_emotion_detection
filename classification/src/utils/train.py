import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm


class Trainer:
    """Class for training the neural net."""

    def __init__(
        self, model, train_data, val_data, loss, optimizer, device, num_epochs
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.statistic_train_loss = []
        self.metric_train_loss = []
        self.statistic_val_loss = []
        self.metric_val_loss = []

    def train(self):
        """Training the neural net."""
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}:".format(epoch, self.num_epochs - 1), flush=True)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    dataloader = self.train_data
                    self.model.train()  # Set model to training mode
                else:
                    dataloader = self.val_data
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                predictions = []
                ground_truth = []

                # Iterate over data
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    ground_truth.append(labels)

                    # Set the gradients to zero since they tend to accumulate
                    self.optimizer.zero_grad()

                    # forward and backward
                    with torch.set_grad_enabled(phase == "train"):
                        preds = self.model(inputs)
                        loss_value = self.loss(preds, labels)
                        preds_class = preds.argmax(dim=1)
                        predictions.append(preds_class)
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss_value.backward()
                            self.optimizer.step()
                    # statistics
                    running_loss += loss_value.item()

                # Normalize by the size of the batch
                epoch_loss = running_loss / len(dataloader)

                # Predictions and labels for the whole epoch
                predictions_numpy = torch.cat(predictions).cpu().numpy()
                labels_numpy = torch.cat(ground_truth).cpu().numpy()

                epoch_metric_loss = metrics.f1_score(
                    predictions_numpy, labels_numpy, average="weighted"
                )

                # Save loss for drawing
                if phase == "train":
                    self.statistic_train_loss.append(epoch_loss)
                    self.metric_train_loss.append(epoch_metric_loss)
                else:
                    self.statistic_val_loss.append(epoch_loss)
                    self.metric_val_loss.append(epoch_metric_loss)
                print(
                    "{} Loss: {:.4f} Metric loss: {:.4f}".format(
                        phase, epoch_loss, epoch_metric_loss
                    ),
                    flush=True,
                )
        return self.model

    def save_weights(self, model_name: str, optimizer_name: str):
        """Save the weights of the model."""
        path_to_weights = "../checkpoints/weights/"
        if not os.path.exists(path_to_weights):
            print("Creating directory to save weights...")
            print("See at ../checkpoints/weights/")
            os.makedirs(path_to_weights)

        weight_name = "{}_{}_{}_epochs.pth".format(
            model_name, optimizer_name, str(self.num_epochs)
        )

        torch.save(self.model.state_dict(), os.path.join(path_to_weights, weight_name))
        print("The trained weights of the model have been saved!")

    def save_model(self, model_name: str, optimizer_name: str):
        """Save the weights as well as the architecture of the model."""
        path_to_models = "../checkpoints/models/"
        if not os.path.exists(path_to_models):
            print("Creating directory to save models...")
            print("See at ../checkpoints/models/")
            os.makedirs(path_to_models)

        model = "{}_{}_{}_epochs.pt".format(
            model_name, optimizer_name, str(self.num_epochs)
        )

        torch.save(self.model.state_dict(), os.path.join(path_to_models, model))
        print("The trained model has been saved!")

    def draw_losses(
        self, model_name: str, optimizer_name: str, save_plots: bool = True
    ):
        """Draw and save the plots for the losses gained during training."""
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(
            np.arange(self.num_epochs),
            self.statistic_train_loss,
            color="red",
            lw=2.0,
            label="Train loss",
        )
        axs[0].plot(
            np.arange(self.num_epochs),
            self.statistic_val_loss,
            color="blue",
            lw=2.0,
            label="Validation loss",
        )
        axs[0].set_title("CrossEntropyLoss loss", loc="left")
        axs[0].set_xlabel("n epochs")

        axs[1].plot(
            np.arange(self.num_epochs),
            self.metric_train_loss,
            color="red",
            lw=2.0,
            label="Train score",
        )
        axs[1].plot(
            np.arange(self.num_epochs),
            self.metric_val_loss,
            color="blue",
            lw=2.0,
            label="Validation score",
        )
        axs[1].set_title("f1 score", loc="left")
        axs[1].set_xlabel("n epochs")

        axs[0].legend()
        axs[1].legend()

        fig.tight_layout()
        plt.show()

        # Save plots
        if save_plots:
            path_to_plots = "../logs/plots/"
            if not os.path.exists(path_to_plots):
                print("Creating directory to save plots...")
                print("See at ../logs/plots/")
            plot_name = "{}_{}_{}_epochs.png".format(
                model_name, optimizer_name, str(self.num_epochs)
            )
            fig.savefig(os.path.join(path_to_plots, plot_name))
            print("The plots of losses have been saved!")
