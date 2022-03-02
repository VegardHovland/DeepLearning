import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import max_pool2d, nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            #Layer 1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),           
            nn.MaxPool2d((2,2), stride= 2),
            nn.Dropout2d(p=0.1),

            #Layer 2
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters *4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=num_filters *4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride= 2),
            nn.Dropout2d(p=0.1),

            #Layer 3
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=num_filters*8,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=num_filters*8,
                out_channels=num_filters*8,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride= 2),
            nn.Dropout2d(p=0.1),
        )
        

    
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 128*4*4*2
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        # Pass through convolutional layers
        conv_layers = self.feature_extractor(x)
        # Flatten
        conv_layers = conv_layers.view((x.shape[0], -1))
        # Pass through fully connetcted
        out = self.classifier(conv_layers)


        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"

        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    utils.plot_loss(trainer.train_history["accuracy"], label="Train Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    adap_momentum = 0.9
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)

    # Model 1
    trainer1 = Trainer(
        batch_size,
        learning_rate,
        # adap_momentum,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    # Model 2
    """""
    trainer2 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
"""
    trainer1.train()
#    trainer2.train()

    create_plots(trainer1, "task3_model_3_trainacc")
   # create_plots(trainer2, "task3_model_2"_trainacc)

if __name__ == "__main__":
    main()