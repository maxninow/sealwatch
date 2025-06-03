import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import Accuracy
from tqdm import tqdm

from .dataset import DatasetLoad
from .model import XuNet

def test_model(
    test_cover_path,
    test_stego_path,
    model_path="./checkpoints/best_model.pt",
    batch_size=16,
    test_size=200,
):
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    test_data = DatasetLoad(
        test_cover_path, test_stego_path, test_size, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load model
    model = XuNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Metric
    test_accuracy = Accuracy(task="binary").to(device)

    # Testing loop
    test_loss = 0
    test_accuracy.reset()
    loss_fn = torch.nn.NLLLoss()
    progress_bar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch in progress_bar:
            images = torch.cat((batch["cover"], batch["stego"]), 0).to(device)
            labels = torch.cat((batch["label"][0], batch["label"][1]), 0).to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            test_accuracy.update(outputs.argmax(dim=1), labels)

            # Update progress bar
            progress_bar.set_postfix(
                loss=loss.item(), acc=test_accuracy.compute().item()
            )

    # Final results
    avg_test_loss = test_loss / len(test_loader)
    final_test_accuracy = test_accuracy.compute().item()

    return avg_test_loss, final_test_accuracy