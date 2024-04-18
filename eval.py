from evaluation import CausalMetric
from evaluation_gai import CausalMetric as CausalMetric_gai
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50,inception_v3,vgg16,mobilenet_v2,maxvit_t,vit_b_16
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
parser.add_argument('--model', type=str, default='inception_v3',
                    choices=["inception_v3", "resnet50", "vgg16", "mobilenet_v2", "maxvit_t", "vit_b_16"])
parser.add_argument('--attr_method', type=str, default='agi')
parser.add_argument("--dataset",type=str,default="isa",choices=["attack","isa"])
parser.add_argument("--single_softmax",action="store_true")
parser.add_argument("--eval_method",type=str,default="before",choices=["before","now","now_gai"])
args = parser.parse_args()
perfix = f"scores_{args.dataset}_{args.eval_method}"
os.makedirs(perfix,exist_ok=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(f"{perfix}/{args.model}_{args.attr_method}{'_singlesoftmax' if args.single_softmax else ''}_scores.npz"):
        with torch.no_grad():
            if args.eval_method == "now" or args.eval_method == "now_gai":
                CausalMetric = CausalMetric_gai
            if args.dataset == "attack":
                img_batch = torch.load("img_batch.pt")
                target_batch = torch.load("label_batch.pt")
            else:
                img_batch = torch.load("ISA_dataset/img_batch.pt")
                target_batch = torch.load("ISA_dataset/label_batch.pt")
            model = eval(f"{args.model}(pretrained=True).eval().to(device)")
            sm = nn.Softmax(dim=-1)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            norm_layer = transforms.Normalize(mean, std)
            model = nn.Sequential(norm_layer, model, sm).to(device)
            baseline = torch.load(f"baselines/{args.model}.pt").to(device)
            substate_fn = lambda x: baseline.repeat(x.shape[0], 1, 1, 1)
            deletion = CausalMetric(model, 'del', 224, substrate_fn=substate_fn if args.eval_method == "now_gai" else torch.zeros_like,reverse=False)
            insertion = CausalMetric(model, 'ins', 224, substrate_fn=substate_fn if args.eval_method == "now_gai" else torch.zeros_like,reverse=False)
            attribution = np.load(f"attributions_{args.dataset}/{args.model}_{args.attr_method}_attributions{'_singlesoftmax' if args.single_softmax else ''}.npy")
            scores = {'del': deletion.evaluate(
                img_batch, attribution, 100), 'ins': insertion.evaluate(img_batch, attribution, 100)}
            scores['ins'] = np.array(scores['ins'])
            scores['del'] = np.array(scores['del'])
            np.savez(f"{perfix}/{args.model}_{args.attr_method}{'_singlesoftmax' if args.single_softmax else ''}_scores.npz", **scores)