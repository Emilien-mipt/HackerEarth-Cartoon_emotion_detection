import matplotlib.pyplot as plt
import numpy as np


def show_input(input_tensor, title=""):
    """Show a single image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = input_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    plt.imshow(image.clip(0, 1))
    plt.title(title)
    plt.show()
    plt.pause(0.001)


def show_batch(dataloader, class_names: list):
    """Show images from a batch."""
    X_batch, y_batch = next(iter(dataloader))
    for x_item, y_item in zip(X_batch, y_batch):
        show_input(x_item, title=class_names[y_item])
