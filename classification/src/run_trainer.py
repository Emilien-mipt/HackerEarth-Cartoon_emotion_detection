import argparse

import torch
from torchvision import models
from utils.data_processing import train_val_dataloaders
from utils.show_batch import show_batch
from utils.train import Trainer

DATA_PATH = "../data/"
TRAIN_DIR = DATA_PATH + "train/"
DEV_DIR = DATA_PATH + "val/"

class_names = ["Angry", "Happy", "Sad", "Surprised", "Unknown"]


def main():
    batch_size = args.batch_size
    augment = args.augment
    dataloaders = train_val_dataloaders(TRAIN_DIR, DEV_DIR, augment, batch_size)

    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]

    show = args.show_batch

    if show:
        # Let's have a look at the first batch
        print("Show batch from train dataloader: ")
        show_batch(train_dataloader, class_names)

        print("Show batch from val dataloader: ")
        show_batch(val_dataloader, class_names)

    # Model - pretrained ResNet18, trained on ImageNet
    model = models.resnet18(pretrained=True)

    # Disable grad for all conv layers
    for param in model.parameters():
        param.requires_grad = False
    print(
        "Output of ResNet18 before FC layer, that we add later: ", model.fc.in_features
    )
    # Add FC layer with 2 outputs: cleaned or dirty
    model.fc = torch.nn.Linear(model.fc.in_features, 5)

    # Put model on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function - binary Cross-Entropy
    loss = torch.nn.CrossEntropyLoss()

    learning_rate = args.learning_rate
    # Optimization method - Adam
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=learning_rate)

    number_epochs = args.num_epochs
    # Training
    model_trainer = Trainer(
        model,
        train_dataloader,
        val_dataloader,
        loss,
        optimizer,
        device,
        num_epochs=number_epochs,
    )

    print("Begin training: ")
    model_trainer.train()

    # Draw losses and save the plot
    model_trainer.draw_losses("ResNet18", "Adam")

    # Save weights of the model
    model_trainer.save_weights("ResNet18", "Adam")

    # Save trained model
    model_trainer.save_model("ResNet18", "Adam")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the arguments for the instance of the Trainer class"
    )
    parser.add_argument(
        "--augment",
        type=bool,
        default=False,
        help="Apply augmentation to train set or not",
    )
    parser.add_argument(
        "--show_batch",
        type=bool,
        default=False,
        help="Show sample batches before training or not",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3.0e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="N epochs for training"
    )
    args = parser.parse_args()
    print(args.__dict__)
    main()
