import os
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

from .dataset import DatasetLoad
from .model import XuNet
from tqdm import tqdm

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, metric, epoch, num_epochs):
    model.train()
    epoch_loss = 0
    metric.reset()
    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}]")
    for batch in progress_bar:
        images = torch.cat((batch["cover"], batch["stego"]), 0).to(device)
        labels = torch.cat((batch["label"][0], batch["label"][1]), 0).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        metric.update(outputs.argmax(dim=1), labels)

        # Update progress bar
        progress_bar.set_postfix(
            loss=loss.item(), acc=metric.compute().item()
        )

    return epoch_loss / len(dataloader), metric.compute()


def validate_one_epoch(model, dataloader, loss_fn, device, metric, epoch, num_epochs):
    model.eval()
    epoch_loss = 0
    metric.reset()
    progress_bar = tqdm(dataloader, desc=f"Validation [{epoch}/{num_epochs}]")
    with torch.no_grad():
        for batch in progress_bar:
            images = torch.cat((batch["cover"], batch["stego"]), 0).to(device)
            labels = torch.cat((batch["label"][0], batch["label"][1]), 0).to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()
            metric.update(outputs.argmax(dim=1), labels)

            # Update progress bar
            progress_bar.set_postfix(
                loss=loss.item(), acc=metric.compute().item()
            )

    return epoch_loss / len(dataloader), metric.compute()


def train_model(
    cover_path,
    stego_path,
    valid_cover_path,
    valid_stego_path,
    checkpoints_dir="./checkpoints/",
    batch_size=16,
    num_epochs=50,
    train_size=1000,
    val_size=200,
    lr=0.001,
    factor=0.1,
    patience=10,
    min_lr=1e-6,
):

    if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
            
    # Set up logging
    logging.basicConfig(filename="training.log", level=logging.INFO)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    train_data = DatasetLoad(
        cover_path, stego_path, train_size, transform=transforms.ToTensor()
    )
    val_data = DatasetLoad(
        valid_cover_path, valid_stego_path, val_size, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer, scheduler
    model = XuNet().to(device)
    #model.apply(weights_init)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr)

    # Metrics
    train_accuracy = Accuracy(task="binary").to(device)
    val_accuracy = Accuracy(task="binary").to(device)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, train_accuracy, epoch, num_epochs)
        val_loss, val_acc = validate_one_epoch(model, val_loader, loss_fn, device, val_accuracy, epoch, num_epochs)

        # Scheduler step
        scheduler.step(val_loss)

        # Log and print epoch summary
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{num_epochs}] Summary: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"Learning Rate: {current_lr:.6f}"
        )
        logging.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}, "
                     f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}, LR={current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best_model.pt"))

    print("Training complete.")