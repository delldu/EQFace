"""Model predict."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 08月 05日 星期四 20:52:22 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os
import pdb  # For debug

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import get_model, model_device, model_setenv


if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/Face_Quality.pth",
        help="checkpint file",
    )
    parser.add_argument("--input", type=str, default="images/*.png", help="input image")
    args = parser.parse_args()

    model_setenv()

    model = get_model(args.checkpoint)
    device = model_device()
    model = model.to(device)
    model.eval()

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = sorted(glob.glob(args.input))
    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB").resize((112, 112))
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        print("{:.4f} {}".format(output_tensor[0, 0].item(), filename))
