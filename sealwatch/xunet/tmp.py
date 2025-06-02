from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
import os


class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model,
        loss_fn,
        optimizer_class,
        lr=0.001,
        batch_size=16,
        num_epochs=50,
        factor=0.1,
        patience=10,
        min_lr=1e-6,
        checkpoints_dir="./checkpoints/",
        device=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.checkpoints_dir = checkpoints_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None
        self.is_fitted_ = False

    def fit(self, X, y, X_val=None, y_val=None):
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        # Validation data (optional)
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        # Create DataLoader for training
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.factor, patience=self.patience, min_lr=self.min_lr
        )

        # Metrics
        train_accuracy = Accuracy(task="binary").to(self.device)
        val_accuracy = Accuracy(task="binary").to(self.device) if val_loader else None

        # Training loop
        best_val_loss = float("inf")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        for epoch in range(1, self.num_epochs + 1):
            # Training phase
            self.model.train()
            train_loss = 0
            train_accuracy.reset()
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_accuracy.update(outputs.argmax(dim=1), batch_y)

            train_loss /= len(train_loader)
            train_acc = train_accuracy.compute().item()

            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0
                val_accuracy.reset()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = self.loss_fn(outputs, batch_y)
                        val_loss += loss.item()
                        val_accuracy.update(outputs.argmax(dim=1), batch_y)

                val_loss /= len(val_loader)
                val_acc = val_accuracy.compute().item()

                # Scheduler step
                self.scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, "best_model.pt"))

                # Print epoch summary
                print(
                    f"Epoch [{epoch}/{self.num_epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
            else:
                # Print epoch summary (no validation)
                print(
                    f"Epoch [{epoch}/{self.num_epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("You must fit the model before making predictions.")

        # Convert data to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Move model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()

        return predictions

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("You must fit the model before making predictions.")

        # Convert data to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Move model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities