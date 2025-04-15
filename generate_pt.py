import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch
import numpy as np
import os
from tqdm import tqdm

os.makedirs('data', exist_ok=True)

argparser = argparse.ArgumentParser()
argparser.add_argument('--img_path', type=str, default='visualized_imgs')

args = argparser.parse_args()
# img_batch = torch.load(args.pt_path)
# img_batch = img_batch.permute(0, 2, 3, 1).cpu().numpy()
# img_batch = img_batch * 255
# img_batch = img_batch.astype(np.uint8)
# for i in tqdm(range(img_batch.shape[0])):
#     img = Image.fromarray(img_batch[i])
#     img.save(f"visualized_imgs/{i}.png")
img_batch = []
for i in range(1000):
    img = Image.open(f"{args.img_path}/{i}.png")
    img = np.array(img)
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1)
    img_batch.append(img)
img_batch = torch.stack(img_batch)
torch.save('data/img_batch.pt', img_batch)