# EfficientNet B0 for Steganalysis

This is a tool for catching secret messages in images. No steganography will pass through unnoticed anymore!


## Setup

First, create a virtual environment for Python.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Next, install poetry and let it set up the environment.

```bash
pip install poetry
poetry env use "$(pwd)/.venv/bin/python"
poetry update
```


## Usage

Run the example inference of the network with

```python
python src/example.py
```

Note: the first run is slow due to JIT compilation.


## Structure

- data
    - 1.png: example image from the BOSSBase
    - split_tr.csv: list of training images (from BOSSBase) used to train the network. Do not use these images for your experiments.
- models
    - model_best.pth: trained model. It was trained on permuted LSBM steganography at $\alpha=0.05$, without coding ($e=2$).
- src: source code
    - b0.py: Python code of the model
    - example.py: example code running the inference on the example image


