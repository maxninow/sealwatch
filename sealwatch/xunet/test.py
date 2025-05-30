"""This module is used to test the XuNet model."""
from glob import glob
from tqdm import tqdm
import torch
import numpy as np
import imageio as io
from .model import XuNet

def test_model(
    test_batch_size = 40,
    test_cover_path = "./data/boss/split_te_hill_cover/*.png",
    test_stego_path = "./data/boss/split_te_hill_stego/*.png",
    chkpt = "./checkpoints/net_65.pt"
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cover_image_names = glob(test_cover_path)
    stego_image_names = glob(test_stego_path)

    if len(cover_image_names) == 0 or len(stego_image_names) == 0:
        raise ValueError("No images found in the specified paths.")

    cover_labels = np.zeros((len(cover_image_names)))
    stego_labels = np.ones((len(stego_image_names)))

    model = XuNet().to(device)

    ckpt = torch.load(chkpt, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    # pylint: disable=E1101
    images = torch.empty((test_batch_size, 1, 512, 512), dtype=torch.float, device=device)
    # pylint: enable=E1101
    test_accuracy = []

    for idx in tqdm(range(0, len(cover_image_names), test_batch_size // 2), desc="testing"):
        cover_batch = cover_image_names[idx : idx + test_batch_size // 2]
        stego_batch = stego_image_names[idx : idx + test_batch_size // 2]

        batch = []
        batch_labels = []

        xi = 0
        yi = 0
        for i in range(2 * len(cover_batch)):
            if i % 2 == 0:
                batch.append(stego_batch[xi])
                batch_labels.append(1)
                xi += 1
            else:
                batch.append(cover_batch[yi])
                batch_labels.append(0)
                yi += 1
        # pylint: disable=E1101
        for i in range(test_batch_size):
            images[i, 0, :, :] = torch.tensor(io.imread(batch[i]), device=device)
        image_tensor = images.to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
        # pylint: enable=E1101
        outputs = model(image_tensor)
        prediction = outputs.data.max(1)[1]

        accuracy = (
            prediction.eq(batch_labels.data).sum()
            * 100.0
            / (batch_labels.size()[0])
        )
        test_accuracy.append(accuracy.item())

    print(f"test_accuracy = {sum(test_accuracy)/len(test_accuracy):.2f}")
