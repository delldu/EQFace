"""Create model."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 08月 05日 星期四 20:52:22 CST
# ***
# ************************************************************************************/
#

import math
import os
import pdb  # For debug
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

from model_helper import ResNet, FaceQuality


class EQFaceModel(nn.Module):
    """EQFace Model."""

    def __init__(self):
        """Init model."""
        super(EQFaceModel, self).__init__()
        self.backbone = ResNet(num_layers=100, feature_dim=512)
        self.quality = FaceQuality(512 * 7 * 7)

    def forward(self, x):
        """Forward."""

        # Convert value from [0, 1.0] to [-1.0, 1.0]
        x = 2.0 * (x - 0.5)
        fc = self.backbone(x)

        return self.quality(fc).clamp(0.0, 1.0)


def model_load(model, path):
    """Load model."""

    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()

    for n, p in state_dict.items():
        if n.startswith("module."):
            n = n[7:]
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""

    torch.save(model.state_dict(), path)


def get_model(checkpoint):
    """Create model."""

    model = EQFaceModel()

    if not os.path.exists(checkpoint):
        model_load(model.backbone, "models/backbone.pth")
        model_load(model.quality, "models/quality.pth")
        model_save(model, checkpoint)
    else:
        model_load(model, checkpoint)

    return model


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""

        self.reset()

    def reset(self):
        """Reset average."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag=""):
    """Trainning model ..."""

    total_loss = Counter()

    model.train()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Loss Function
    # ***
    # ************************************************************************************/
    #
    loss_function = nn.L1Loss()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            predicts = model(images)

            loss = loss_function(predicts, targets)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss="{:.6f}".format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=""):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Loss Function
    # ***
    # ************************************************************************************/
    #
    loss_function = nn.L1Loss()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                predicts = model(images)

            loss = loss_function(predicts, targets)
            loss_value = loss.item()

            valid_loss.update(loss_value, count)
            t.set_postfix(loss="{:.6f}".format(valid_loss.avg))
            t.update(count)


def model_device():
    """Please call after model_setenv."""

    return torch.device(os.environ["DEVICE"])


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random

    random.seed(42)
    torch.manual_seed(42)

    # Set default device to avoid exceptions
    if os.environ.get("DEVICE") != "cuda" and os.environ.get("DEVICE") != "cpu":
        os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    if os.environ["DEVICE"] == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])


# model = get_model("models/Face_Quality.pth")
# print(model)
