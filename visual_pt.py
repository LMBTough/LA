import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch
import numpy as np
import os
from tqdm import tqdm

os.makedirs('visualized_imgs', exist_ok=True)

argparser = argparse.ArgumentParser()
argparser.add_argument('--pt_path', type=str, default='data/img_batch.pt')

args = argparser.parse_args()
img_batch = torch.load(args.pt_path)
img_batch = img_batch.permute(0, 2, 3, 1).cpu().numpy()
img_batch = img_batch * 255
img_batch = img_batch.astype(np.uint8)
for i in tqdm(range(img_batch.shape[0])):
    img = Image.fromarray(img_batch[i])
    img.save(f"visualized_imgs/{i}.png")