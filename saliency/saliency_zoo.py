from saliency.core import *
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
import random
from torch.autograd import Variable as V
from saliency.agis import *
from saliency.la import *
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fast_ig(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    method = FastIG(model)
    result = method(data, target).squeeze()
    return np.expand_dims(result, axis=0)



def guided_ig(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    class_idx_str = 'class_idx_str'

    def call_model_function(images, call_model_args=None, expected_keys=None):
        target_class_idx = call_model_args[class_idx_str]
        images = torch.from_numpy(images).float().to(device)
        images = images.requires_grad_(True)
        output = model(images)
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(
            outputs, images, grad_outputs=torch.ones_like(outputs))[0]
        gradients = grads.cpu().detach().numpy()
        return {'INPUT_OUTPUT_GRADIENTS': gradients}

    im = data.squeeze().cpu().detach().numpy()
    call_model_args = {class_idx_str: target}
    baseline = np.zeros(im.shape)
    method = GuidedIG()

    result =  method.GetMask(
        im, call_model_function, call_model_args, x_steps=15, x_baseline=baseline)
    return np.expand_dims(result, axis=0)

def big(model, data, target, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map


def mfaba(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def ig(model, data, target, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradient(model)
    return ig(data, target, gradient_steps=gradient_steps)


def sm(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sm = SaliencyGradient(model)
    return sm(data, target)

    
def sg(model, data, target,stdevs=0.15, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sg = SmoothGradient(model,stdevs=stdevs)
    return sg(data, target, gradient_steps=gradient_steps)

def deeplift(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    dl = DL(model)
    return dl(data, target)


def attexplore(model, data, target, data_min=0, data_max=1, epsilon=16,N=20,num_steps=10, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = ATTEXPLORE(model)
    epsilon = epsilon / 255
    attack = FGSMGradSSA(
        epsilon=epsilon, data_min=data_min, data_max=data_max,N=N)
    _, _, _, hats, grads = attack(
        model, data, target,num_steps=num_steps, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map


from saliency.core import AttributionPriorExplainer
def eg(model,dataloader,data,target,*args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    APExp = AttributionPriorExplainer(dataloader.dataset, 4,k=1)
    attr_eg = APExp.shap_values(model,data).cpu().detach().numpy()
    return attr_eg