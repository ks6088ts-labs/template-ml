import logging
from typing import Annotated

import matplotlib.pyplot as plt
import torch
import typer

# Import packages
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from template_ml.loggers import get_logger

app = typer.Typer(
    add_completion=False,
    help="PyTorch Playground CLI",
)

logger = get_logger(__name__)


def set_verbose_logging(
    verbose: bool,
):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)


# Define model architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Class labels for FashionMNIST
CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@app.command(help="Download and prepare FashionMNIST dataset")
def download_data(
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", help="Directory to save dataset"),
    ] = "data",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    """Download training and test data from FashionMNIST dataset."""
    set_verbose_logging(verbose)

    # Download training data
    training_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )
    logger.info(f"Training data downloaded: {len(training_data)} samples")

    # Download test data
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    logger.info(f"Test data downloaded: {len(test_data)} samples")
    print(f"✅ Dataset downloaded successfully to '{data_dir}'")


@app.command(help="Visualize sample images from the dataset")
def show_samples(
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", help="Directory containing the dataset"),
    ] = "data",
    num_samples: Annotated[
        int,
        typer.Option("--num-samples", "-n", help="Number of samples to display"),
    ] = 9,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    """Display sample images from the training dataset."""
    set_verbose_logging(verbose)

    training_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=ToTensor(),
    )

    labels_map = {i: label for i, label in enumerate(CLASSES)}

    cols = 3
    rows = (num_samples + cols - 1) // cols
    figure = plt.figure(figsize=(8, 8))

    for i in range(1, min(num_samples, len(training_data)) + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


@app.command(help="Train a neural network model on FashionMNIST")
def train(
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", help="Directory containing the dataset"),
    ] = "data",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size for training"),
    ] = 64,
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs"),
    ] = 5,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Learning rate"),
    ] = 1e-3,
    model_path: Annotated[
        str,
        typer.Option("--model-path", "-m", help="Path to save the trained model"),
    ] = "model.pth",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    """Train a neural network on the FashionMNIST dataset."""
    set_verbose_logging(verbose)

    # Load data
    training_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=ToTensor(),
    )

    # Create data loaders
    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
    )

    # Get device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Create model
    model = NeuralNetwork().to(device)
    logger.info(f"Model architecture:\n{model}")

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=learning_rate,
    )

    def train_epoch(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss_value, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

    def test_epoch(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Training loop
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        test_epoch(test_dataloader, model, loss_fn)
    print("Done!")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved PyTorch Model State to {model_path}")


@app.command(help="Evaluate a trained model")
def evaluate(
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", help="Directory containing the dataset"),
    ] = "data",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size for evaluation"),
    ] = 64,
    model_path: Annotated[
        str,
        typer.Option("--model-path", "-m", help="Path to the trained model"),
    ] = "model.pth",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    """Evaluate the trained model on test dataset."""
    set_verbose_logging(verbose)

    # Load test data
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=ToTensor(),
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
    )

    # Get device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Load model
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Evaluate
    loss_fn = nn.CrossEntropyLoss()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size

    print(f"\n{'=' * 50}")
    print("Test Results:")
    print(f"  Accuracy: {(100 * accuracy):>0.1f}%")
    print(f"  Average loss: {test_loss:>8f}")
    print(f"{'=' * 50}\n")


@app.command(help="Make predictions on test samples")
def predict(
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", help="Directory containing the dataset"),
    ] = "data",
    model_path: Annotated[
        str,
        typer.Option("--model-path", "-m", help="Path to the trained model"),
    ] = "model.pth",
    num_samples: Annotated[
        int,
        typer.Option("--num-samples", "-n", help="Number of samples to predict"),
    ] = 5,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    """Make predictions on random test samples."""
    set_verbose_logging(verbose)

    # Load test data
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=ToTensor(),
    )

    # Get device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Load model
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Make predictions
    print(f"\n{'=' * 50}")
    print("Predictions:")
    print(f"{'=' * 50}")

    for i in range(num_samples):
        idx = torch.randint(len(test_data), size=(1,)).item()
        x, y = test_data[idx][0], test_data[idx][1]

        with torch.no_grad():
            x = x.to(device)
            pred = model(x)
            predicted_class = CLASSES[pred[0].argmax(0)]
            actual_class = CLASSES[y]

        status = "✅" if predicted_class == actual_class else "❌"
        print(f"{status} Sample {i + 1}: Predicted: '{predicted_class}', Actual: '{actual_class}'")

    print(f"{'=' * 50}\n")


@app.command(help="Show dataset information")
def info(
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", help="Directory containing the dataset"),
    ] = "data",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = True,
):
    """Display information about the dataset."""
    set_verbose_logging(verbose)

    # Load datasets
    training_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=ToTensor(),
    )

    # Get sample shape
    sample_x, sample_y = next(iter(DataLoader(test_data, batch_size=1)))

    print(f"\n{'=' * 50}")
    print("FashionMNIST Dataset Information:")
    print(f"{'=' * 50}")
    print(f"Training samples: {len(training_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Image shape: {sample_x.shape}")
    print(f"Label shape: {sample_y.shape}")
    print(f"Number of classes: {len(CLASSES)}")
    print("\nClasses:")
    for i, class_name in enumerate(CLASSES):
        print(f"  {i}: {class_name}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    assert load_dotenv(
        override=True,
        verbose=True,
    ), "Failed to load environment variables"
    app()
