"""Example for the gradient exercise.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import conseal as cl
import numpy as np
from PIL import Image
import torch

if __name__ == '__main__':

    # load model
    from b0 import B0, SqueezeExcite, InvertedResidual, DepthwiseSeparableConv
    device = torch.device('cpu')
    model = torch.load(
        'models/B0-LSBM_0.05_lsb-250224215825/model_best.pth',
        map_location=device,
        weights_only=False,
    )

    # load cover
    x0 = np.array(Image.open('data/1.png'))
    # simulate stego embedding
    x1 = x0 + cl.lsb.simulate(
        x0,
        alpha=.05,
        n=x0.size,
        modify=cl.LSB_MATCHING,
        seed=12345,
    ).astype('int8').astype('float32')

    # apply detector to the cover
    x = x0[None, None].astype('float32') / 255.
    x = torch.from_numpy(x).to(device)
    logit0 = model(x)
    pred0 = torch.nn.functional.softmax(logit0, dim=1)[:, 1]

    # apply detector to the stego
    x = x1[None, None].astype('float32') / 255.
    x = torch.from_numpy(x).to(device)
    logit1 = model(x)
    pred1 = torch.nn.functional.softmax(logit1, dim=1)[:, 1]

    print(pred0, pred1)  # softmax outputs
