from evaluation import CausalMetric
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, inception_v3, vgg16, maxvit_t
from tqdm import tqdm
import torch
import numpy as np
import argparse
import os
import random
import csv

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(3407)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inception_v3')
parser.add_argument('--attr_method', type=str, default='la')
parser.add_argument('--spatial_range', type=int, default=20)
parser.add_argument('--max_iter', type=int, default=20)
parser.add_argument('--samples_number', type=int, default=20)
parser.add_argument('--prefix', type=str, default='scores', 
                    help='Folder used to save scores .npz files')
parser.add_argument('--attr_prefix', type=str, default='attributions', 
                    help='Folder used to save attributions .npy files')
parser.add_argument('--csv_path', type=str, default='results.csv', 
                    help='Path to output CSV file')

args = parser.parse_args()

prefix = args.prefix

os.makedirs(prefix, exist_ok=True)

attr_prefix = args.attr_prefix

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz_path = f"{prefix}/{args.model}_{args.attr_method}_spatial-range-{args.spatial_range}_max-iter-{args.max_iter}_sampling-times-{args.samples_number}_scores.npz"
    
    if not os.path.exists(npz_path):
        with torch.no_grad():
            img_batch = torch.load("data/img_batch.pt").float()
            target_batch = torch.load("data/label_batch.pt")
            model = eval(f"{args.model}(pretrained=True).eval().to(device)")
            sfmx = nn.Softmax(dim=-1)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            norm_layer = transforms.Normalize(mean, std)
            model = nn.Sequential(norm_layer, model, sfmx).eval().to(device)
            deletion = CausalMetric(model, 'del', 224, torch.zeros_like, reverse=False)
            insertion = CausalMetric(model, 'ins', 224, torch.zeros_like, reverse=False)
            if args.attr_method == 'la':
                attribution = np.load(f"{attr_prefix}/{args.model}_{args.attr_method}_spatial-range-{args.spatial_range}_max-iter-{args.max_iter}_sampling-times-{args.samples_number}_attributions.npy")
            else:
                attribution = np.load(f"{attr_prefix}/{args.model}_{args.attr_method}_attributions.npy")
            
            scores = {
                'del': deletion.evaluate(img_batch, attribution, 100),
                'ins': insertion.evaluate(img_batch, attribution, 100)
            }
            scores['ins'] = np.array(scores['ins'])
            scores['del'] = np.array(scores['del'])
            np.savez(npz_path, **scores)

    data = np.load(npz_path)
    insertion_mean = np.mean(data['ins'])
    deletion_mean = np.mean(data['del'])

    file_exists = os.path.isfile(args.csv_path)

with open(args.csv_path, mode='a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    if not file_exists:
        writer.writerow(['model', 'attr_method', 'spatial_range', 'max-iter', 'samples_number', 'insertion', 'deletion'])

    if args.attr_method == 'la':
        writer.writerow([
            args.model,
            args.attr_method,
            args.spatial_range,
            args.max_iter,
            args.samples_number,
            round(insertion_mean, 6),
            round(deletion_mean, 6)
        ])
    else:
        writer.writerow([
            args.model,
            args.attr_method,
            '-',
            '-',
            '-',
            round(insertion_mean, 6),
            round(deletion_mean, 6)
        ])
