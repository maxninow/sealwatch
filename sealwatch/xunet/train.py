import os
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
        Initializes the XuNet classifier.

        Args:
            checkpoints_dir (str): Directory to save model checkpoints.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.
            factor (float): Factor for learning rate reduction.
            patience (int): Patience for learning rate scheduler.
            min_lr (float): Minimum learning rate.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.checkpoints_dir = checkpoints_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model, loss, optimizer, and scheduler
        self.model = XuNet().to(self.device)
        self.loss_fn = nn.NLLLoss()
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.factor, patience=self.patience, min_lr=self.min_lr
        )
        self.train_accuracy = Accuracy(task="binary").to(self.device)
        self.val_accuracy = Accuracy(task="binary").to(self.device)

        self.transform = transforms.ToTensor()                         # Convert to PyTorch tensor

        # Create checkpoints directory if it doesn't exist
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

    def train_one_epoch(self, model, dataloader, loss_fn, optimizer, device, metric, epoch, num_epochs):
        model.train()
        epoch_loss = 0
        metric.reset()
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]")
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            metric.update(outputs.argmax(dim=1), batch_y)

            # Update progress bar
            progress_bar.set_postfix(
                loss=loss.item(), acc=metric.compute().item()
            )

        return epoch_loss / len(dataloader), metric.compute()

    def validate_one_epoch(self, model, dataloader, loss_fn, device, metric, epoch, num_epochs):
        model.eval()
        epoch_loss = 0
        metric.reset()
        progress_bar = tqdm(dataloader, desc=f"Validation [{epoch}/{num_epochs}]")
        with torch.no_grad():
            for batch_X, batch_y in progress_bar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                epoch_loss += loss.item()
                metric.update(outputs.argmax(dim=1), batch_y)

                # Update progress bar
                progress_bar.set_postfix(
                    loss=loss.item(), acc=metric.compute().item()
                )

        return epoch_loss / len(dataloader), metric.compute()

    def fit(
        cover_path,
        stego_path,
        valid_cover_path,
        valid_stego_path,):
        """
        Trains the model using the provided training and validation data.

        Args:
            X_train (np.ndarray): Training features (images).
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features (images).
            y_val (np.ndarray): Validation labels.

        Returns:
            self
        """
        # Set up logging
        logging.basicConfig(filename="training.log", level=logging.INFO)

        # Dataset and DataLoader
        train_data = DatasetLoad(
            cover_path, stego_path, self.train_size, transform=transforms.ToTensor()
        )
        val_data = DatasetLoad(
            valid_cover_path, valid_stego_path, self.val_size, transform=transforms.ToTensor()
        )

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        # Training loop
        best_val_loss = float("inf")
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(self.model, train_loader, self.loss_fn, self.optimizer, self.device, self.train_accuracy, epoch, self.num_epochs)
            val_loss, val_acc = self.validate_one_epoch(self.model, val_loader, self.loss_fn, self.device, self.val_accuracy, epoch, self.num_epochs)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Log and print epoch summary
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch}/{self.num_epochs}] Summary: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}, "
                f"Learning Rate: {current_lr:.6f}"
            )
            logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}, "
                         f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}, LR={current_lr:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, "best_model.pt"))

        print("Training complete.")
        return self

    def predict(self, image_path):
        """
        Predicts whether a given image is stego or cover.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: "stego" if the image is predicted as stego, "cover" otherwise.
        """
        # Load and preprocess the image
        image = Image.open(image_path)
        image = self.transform(image)  # Apply the same transform used during training
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        # Set the model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            prediction = output.argmax(dim=1)  # Get the predicted class (0 or 1)

        # Map the prediction to "stego" or "cover"
        return "stego" if prediction == 1 else "cover"

    def load_model(self, checkpoint_path_in=None):
        """
        Loads the model weights from a checkpoint. If no path is provided, 
        loads the default model from the package folder.

        Args:
            checkpoint_path (str, optional): Path to the checkpoint file. 
                                            If None, loads the default model.
        """
        # Default path to the model in the package folder
        if checkpoint_path_in is None:
            package_dir = os.path.dirname(os.path.abspath(__file__))  # Get the package folder
            checkpoint_path = os.path.join(package_dir, "default_model.pt")  # Default model path

        # Check if the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # Load the model weights
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        
        if checkpoint_path_in is None:
            print(f"Default model loaded")
        else:    
            print(f"Model loaded from {checkpoint_path}")


    def test(self, test_cover_path, test_stego_path, test_size=100):
        """
        Tests the model on a given test dataset.

        Args:
            test_cover_path (str): Path to the directory containing cover images for testing.
            test_stego_path (str): Path to the directory containing stego images for testing.

        Returns:
            dict: A dictionary containing test loss and test accuracy.
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