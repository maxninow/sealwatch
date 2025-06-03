"""Implementation of Xunet.

Based on IEEE Signal Processing Letter 2016 paper:
Guanshuo Xu, Han-Zhou Wu, Yun-Qing Shi
CNN tailored to steganalysis, with facilitated statistical modeling.

Inspired by implementation by Brijesh Singh.

Author: Max Ninow
Affiliation: University of Innsbruck
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from sklearn.base import BaseEstimator, ClassifierMixin
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from .dataset import DatasetLoad
from .model import XuNet


class XuNetTrainer(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 checkpoints_dir="./checkpoints/",
                 batch_size=32,
                 num_epochs=50,
                 lr=0.001,
                 factor=0.1,
                 patience=10,
                 min_lr=1e-6,
                 device=None):
        """
        Initializes the XuNetTrainer class.

        Args:
            checkpoints_dir (str): Directory to save model checkpoints.
            batch_size (int): Batch size for training and testing.
            num_epochs (int): Number of training epochs.
            lr (float): Initial learning rate.
            factor (float): Factor by which the learning rate is reduced.
            patience (int): Number of epochs with no improvement before reducing the learning rate.
            min_lr (float): Minimum learning rate.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.checkpoints_dir = checkpoints_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_accuracy = Accuracy(task="binary").to(self.device)
        self.val_accuracy = Accuracy(task="binary").to(self.device)

        # Initialize model, loss, optimizer, and scheduler
        self.model = XuNet().to(self.device)
        self.loss_fn = nn.NLLLoss()
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.factor, patience=self.patience, min_lr=self.min_lr
        )
        self.train_accuracy = Accuracy(task="binary").to(self.device)
        self.val_accuracy = Accuracy(task="binary").to(self.device)

        self.transform = transforms.ToTensor()

        # Create checkpoints directory if it doesn't exist
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

    def run_epoch(self, model, dataloader, loss_fn, optimizer, device, metric, epoch, num_epochs, train=True):
        model.train() if train else model.eval()
        epoch_loss = 0
        metric.reset()
        progress_bar = tqdm(dataloader, desc=f"{'Train' if train else 'Validation'} [{epoch}/{num_epochs}]")
        with torch.set_grad_enabled(train):
            for batch in progress_bar:
                images = torch.cat((batch["cover"], batch["stego"]), 0).to(self.device)
                labels = torch.cat((batch["label"][0], batch["label"][1]), 0).to(self.device)

                if train:
                    optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                metric.update(outputs.argmax(dim=1), labels)

                # Update progress bar
                progress_bar.set_postfix(
                    loss=loss.item(), acc=metric.compute().item()
                )

        return epoch_loss / len(dataloader), metric.compute()

    def fit(
            self,
            cover_path,
            stego_path,
            valid_cover_path,
            valid_stego_path,
            train_size,
            val_size):
        """
        Trains the model using the provided training and validation datasets.

        Args:
            cover_path (str): Path to the directory containing cover images for training.
            stego_path (str): Path to the directory containing stego images for training.
            valid_cover_path (str): Path to the directory containing cover images for validation.
            valid_stego_path (str): Path to the directory containing stego images for validation.
            train_size (int): Number of training pairs to train on.
            val_size (int): Number of validation pairs to train on.

        Returns:
            self: The trained XuNetTrainer instance.
        """

        # Dataset and DataLoader
        train_data = DatasetLoad(
            cover_path, stego_path, train_size, transform=transforms.ToTensor()
        )
        val_data = DatasetLoad(
            valid_cover_path, valid_stego_path, val_size, transform=transforms.ToTensor()
        )

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        # Training loop
        best_val_loss = float("inf")
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.run_epoch(
                                                self.model,
                                                train_loader,
                                                self.loss_fn,
                                                self.optimizer,
                                                self.device,
                                                self.train_accuracy,
                                                epoch,
                                                self.num_epochs,
                                                train=True)
            val_loss, val_acc = self.run_epoch(
                                                self.model,
                                                val_loader,
                                                self.loss_fn,
                                                self.optimizer,
                                                self.device,
                                                self.val_accuracy,
                                                epoch,
                                                self.num_epochs,
                                                train=False)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Log and print epoch summary
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}/{self.num_epochs}] Summary: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Learning Rate: {current_lr:.6f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, "best_model.pt"))

        print("Training complete.")
        return self

    def predict(self, image_path: str) -> str:
        """
        Predicts whether a given image is stego or cover.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: "stego" if the image is predicted as stego, "cover" otherwise.
        """
        # Load and preprocess the image
        image = Image.open(image_path)
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        # Set the model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            prediction = output.argmax(dim=1)

        # Map the prediction to "stego" or "cover"
        return "stego" if prediction == 1 else "cover"

    def load_model(self, checkpoint_path_in=None):
        """
        Loads the model weights from a checkpoint. If no path is provided,
        loads the default model from the package folder.

        Args:
            checkpoint_path_in (str, optional): Path to the checkpoint file.
                                                If None, loads the default model.
        """
        # Default path to the model in the package folder
        if checkpoint_path_in is None:
            package_dir = os.path.dirname(os.path.abspath(__file__))  # Get the package folder
            checkpoint_path = os.path.join(package_dir, "default_model.pt")  # Default model path
        else:
            checkpoint_path = checkpoint_path_in

        # Check if the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # Load the model weights
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)

        if checkpoint_path_in is None:
            print("Default model loaded")
        else:
            print(f"Model loaded from {checkpoint_path}")

    def test(self, test_cover_path, test_stego_path, test_size=100):
        """
        Tests the model on a given test dataset.

        Args:
            test_cover_path (str): Path to the directory containing cover images for testing.
            test_stego_path (str): Path to the directory containing stego images for testing.
            test_size (int, optional): Number of samples to use for testing. Defaults to 100.

        Returns:
            tuple: Average test loss and test accuracy.
        """
        # Load the test dataset
        test_data = DatasetLoad(
            test_cover_path, test_stego_path, size=test_size, transform=transforms.ToTensor()
        )
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        # Set the model to evaluation mode
        self.model.eval()
        test_loss = 0
        self.val_accuracy.reset()

        # Progress bar for testing
        progress_bar = tqdm(test_loader, desc="Testing")
        with torch.no_grad():
            for batch in progress_bar:
                images = torch.cat((batch["cover"], batch["stego"]), 0).to(self.device)
                labels = torch.cat((batch["label"][0], batch["label"][1]), 0).to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()
                self.val_accuracy.update(outputs.argmax(dim=1), labels)

                # Update progress bar
                progress_bar.set_postfix(
                    loss=loss.item(), acc=self.val_accuracy.compute().item()
                )

        # Compute average loss and accuracy
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = self.val_accuracy.compute().item()

        return avg_test_loss, test_accuracy
