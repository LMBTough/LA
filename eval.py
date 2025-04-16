from evaluation import CausalMetric
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50,inception_v3,vgg16,maxvit_t
from tqdm import tqdm
import torch
import numpy as np
import argparse
import torch
import os
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(3407)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inception_v3')
parser.add_argument('--attr_method', type=str, default='agi')
args = parser.parse_args()
perfix = "scores"
os.makedirs(perfix,exist_ok=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(f"{perfix}/{args.model}_{args.attr_method}_scores.npz"):
        with torch.no_grad():
            img_batch = torch.load("data/img_batch.pt").float()
            target_batch = torch.load("data/label_batch.pt")
            model = eval(f"{args.model}(pretrained=True).eval().to(device)")
            sfmx = nn.Softmax(dim=-1)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            norm_layer = transforms.Normalize(mean, std)
            model = nn.Sequential(norm_layer, model, sfmx).to(device)
            deletion = CausalMetric(model, 'del', 224, torch.zeros_like,reverse=False)
            insertion = CausalMetric(model, 'ins', 224, torch.zeros_like,reverse=False)
            attribution = np.load(f"attributions/{args.model}_{args.attr_method}_attributions.npy")
            scores = {'del': deletion.evaluate(
                img_batch, attribution, 100), 'ins': insertion.evaluate(img_batch, attribution, 100)}
            scores['ins'] = np.array(scores['ins'])
            scores['del'] = np.array(scores['del'])
            np.savez(f"{perfix}/{args.model}_{args.attr_method}_scores.npz", **scores)