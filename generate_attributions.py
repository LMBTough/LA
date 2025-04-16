import torch.nn as nn
from torchvision import transforms as T
import torch
import numpy as np
import argparse
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import os
from torchvision.models import maxvit_t,inception_v3,resnet50,vgg16
from saliency.saliency_zoo import *
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
parser.add_argument('--spatial_range', type=int, default=10)
parser.add_argument('--samples_number', type=int, default=20)
args = parser.parse_args()

perfix = "attributions"
os.makedirs(perfix,exist_ok=True)

attr_methods_with_softmax = ["mfaba","agi","attexplore", "la"]

if args.attr_method == "deeplift":
    from resnet_mod import resnet50
    from vgg16_mod import vgg16


if __name__ == "__main__":
    import os
    if not os.path.exists(f"{perfix}/{args.model}_{args.attr_method}_attributions.npy"):
        attr_method = eval(args.attr_method)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_batch = torch.load("data/img_batch.pt").float()
        target_batch = torch.load("data/label_batch.pt")

        dataset = TensorDataset(img_batch, target_batch)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = eval(f"{args.model}(pretrained=True).eval().to(device)")
        sfmx = nn.Softmax(dim=-1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm_layer = T.Normalize(mean, std)
        starts_with = False
        for attr_name in attr_methods_with_softmax:
            if args.attr_method.startswith(attr_name):
                starts_with = True
                break
        if starts_with:
            model = nn.Sequential(norm_layer, model).eval().to(device)
        else:
            model = nn.Sequential(norm_layer, model, sfmx).eval().to(device)
        if args.attr_method.startswith('fast_ig') or args.attr_method.startswith('guided_ig') or args.attr_method.startswith('big'):
            batch_size = 1
        elif args.attr_method.startswith('ig') or args.attr_method.startswith('ampe') or args.attr_method.startswith('eg') or args.attr_method.startswith("sg") or args.attr_method.startswith("deeplift"):
            batch_size = 4
        elif args.attr_method.startswith('agi') or args.attr_method.startswith('mfaba') or args.attr_method.startswith('sm') or args.attr_method.startswith('la'):
            batch_size = 64
        if args.model == "maxvit_t":
            if args.attr_method != "eg":
                if batch_size > 2:
                    batch_size = batch_size // 2
        attributions = []
        for i in tqdm(range(0, len(img_batch), batch_size)):
            img = img_batch[i:i+batch_size].to(device)
            target = target_batch[i:i+batch_size].to(device)
            if args.attr_method == "eg":
                attribution = attr_method(model, dataloader, img, target)
            elif args.attr_method == "la":
                attribution = attr_method(model, img, target, epsilon=args.spatial_range, max_iter=args.samples_number)
            else:
                attribution = attr_method(model, img, target)
            attributions.append(attribution)
        attributions = np.concatenate(attributions, axis=0)
        np.save(f"{perfix}/{args.model}_{args.attr_method}_attributions.npy", attributions)