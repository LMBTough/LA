from saliency.core import FastIG,FastIGKL, GuidedIG,negflux_pgd_step, BIG, FGSM,FGSMKL, MFABA, SaliencyGradient,SmoothGradient,DL,FGSMGrad,IntegratedGradient,FGSMGradALPHA,dct_2d,idct_2d,DI,gkern,exp,AMPE,FGSMGradSSA
from captum.attr import Saliency
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

def fast_ig_kl(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    method = FastIGKL(model)
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

def guided_ig_gai(model, data, target,model_name):
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
    baseline = torch.load(f'baselines/{model_name}.pt').cpu().detach().squeeze().numpy()
    method = GuidedIG()

    result =  method.GetMask(
        im, call_model_function, call_model_args, x_steps=15, x_baseline=baseline)
    return np.expand_dims(result, axis=0)

def negflux(model, data,target, epsilon=0.1, max_iter=20, topk=20):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = range(0,999,int(1000/topk))
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    step_grad = 0 
    # we randomly sample k adversarial samples
    for l in selected_ids:
        targeted = torch.tensor([l]*data.shape[0]).to(device)

        delta = negflux_pgd_step(data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy() # / topk

    # Return prediction, original image, and heatmap
    return adv_ex

from saliency.core import negflux_pgd_step_kl
def negflux_kl(model, data,target, epsilon=0.1, max_iter=20, topk=20):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = range(0,999,int(1000/topk))
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    step_grad = 0 
    # we randomly sample k adversarial samples
    for l in selected_ids:
        targeted = torch.tensor([l]*data.shape[0]).to(device)

        delta = negflux_pgd_step_kl(data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy() # / topk

    # Return prediction, original image, and heatmap
    return adv_ex

from saliency.core import BIGKL
def big(model, data, target, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map

def big_kl1(model, data, target, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSMKL(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map

def big_kl2(model, data, target, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIGKL(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map

def big_kl3(model, data, target, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSMKL(eps, data_min, data_max) for eps in epsilons]
    big = BIGKL(model, attacks, class_num)
    attribution_map, success = big(model, data, target, gradient_steps)
    return attribution_map

def mfaba_alpha(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    attack = FGSMGradALPHA(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def mfaba_alpha_rgb(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attribution_maps = None
    for ch in ["r","g","b"]:
        mfaba = MFABA(model)
        attack = FGSMGradALPHA(
            epsilon=epsilon, data_min=data_min, data_max=data_max, attack_ch=ch)
        _, _, _, hats, grads = attack(
            model, data, target, use_sign=use_sign, use_softmax=use_softmax)
        attribution_map = list()
        for i in range(len(hats)):
            attribution_map.append(mfaba(hats[i], grads[i]))
        if attribution_maps is None:
            attribution_maps = np.concatenate(attribution_map, axis=0)
        else:
            attribution_maps += np.concatenate(attribution_map, axis=0)
    attribution_maps /= 3
    return attribution_maps

def mfaba_smooth(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=True, use_softmax=True):
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

def mfaba_smooth_no_earlystop(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax,early_stop=False)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

from saliency.core import FGSMGradKL
def mfaba_smooth_kl(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    attack = FGSMGradKL(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    _, _, _, hats, grads = attack(
        model, data, target, use_sign=use_sign, use_softmax=use_softmax)
    attribution_map = list()
    for i in range(len(hats)):
        attribution_map.append(mfaba(hats[i], grads[i]))
    attribution_map = np.concatenate(attribution_map, axis=0)
    return attribution_map

def mfaba_sharp(model, data, target, data_min=0, data_max=1, epsilon=0.3 * 255,use_sign=False, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = MFABA(model)
    attack = FGSMGrad(
        epsilon=epsilon, data_min=data_min, data_max=data_max)
    input_baseline, success, _, hats, grads = attack(
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

from saliency.core import IntegratedGradientKL
def ig_kl(model, data, target, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradientKL(model)
    return ig(data, target, gradient_steps=gradient_steps)

def ig_gai_kl(model, data, target,model_name, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradientKL(model)
    baseline = torch.load(f'baselines/{model_name}.pt').to(data.device)
    bs = data.shape[0]
    baseline = baseline.repeat(bs, 1, 1, 1)
    return ig(data, target, gradient_steps=gradient_steps, baselines=baseline)

def ig_gai(model, data, target,model_name, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradient(model)
    baseline = torch.load(f'baselines/{model_name}.pt').to(data.device)
    bs = data.shape[0]
    baseline = baseline.repeat(bs, 1, 1, 1)
    return ig(data, target, gradient_steps=gradient_steps, baselines=baseline)

def sm(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sm = SaliencyGradient(model)
    return sm(data, target)

def sm_kl(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    data = V(data.detach(), requires_grad=True)
    output = model(data)
    loss = -F.kl_div(output.log(), torch.ones_like(output) / 1000, reduction="batchmean")
    loss.backward()
    return abs(data.grad.cpu().detach().numpy())

def sm_sum(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    data = V(data.detach(), requires_grad=True)
    output = model(data)
    output.sum().backward()
    return abs(data.grad.cpu().detach().numpy())
    
    
def sm_sum_notarget(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    data = V(data.detach(), requires_grad=True)
    output = model(data)
    # 去除target
    cost = 0
    for i in range(len(target)):
        cost += output[i][:target[i]].sum() + output[i][target[i]+1:].sum()
    cost.backward()
    return abs(data.grad.cpu().detach().numpy())
    
def sg(model, data, target,stdevs=0.15, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sg = SmoothGradient(model,stdevs=stdevs)
    return sg(data, target, gradient_steps=gradient_steps)

def deeplift(model, data, target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    dl = DL(model)
    return dl(data, target)

def saliencymap(model,data,target):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    saliencymap = Saliency(model)
    return saliencymap.attribute(data, target).cpu().detach().numpy()

def isa(model,x,label,step_size=5000,add_steps=8,minus_steps=8,alpha=0.004,factor=1.3):
    mask = torch.ones_like(x,dtype=torch.long) # 初始化mask保留所有像素
    importance = torch.zeros_like(x.unsqueeze(0)) # 初始化importance
    n_steps = np.array(x.size()[1:]).prod() // step_size + 1
    removed_count = 0
    for i in tqdm(range(n_steps)):
        combine, combine_flatten = exp(model, x, label,mask, add_steps=add_steps, minus_steps=minus_steps, alpha=alpha, lambda_r=0.01, method="total*delta")
        combine[mask.float().cpu() == 0] = -np.inf # 将已经去掉的设置成无限小
        combine_flatten = np.concatenate([c.flatten()[np.newaxis,:] for c in combine])
        # combine_flatten = combine.flatten()
        
        if removed_count + step_size > combine_flatten.shape[-1]:
            step_size = len(combine_flatten) - removed_count
        m = np.zeros_like(combine_flatten)
        temp = np.argsort(combine_flatten)[:,removed_count:removed_count+step_size]
        for t in range(len(temp)):
            m[t,temp[t]] = 1 
        m = m.reshape(combine.shape).astype(bool)
        a = combine[m]
        # raise NotImplementedError
        if len(a) == 0:
            break
        a = a - a.min(axis=0)
        a = a / (a.max(axis=0)+1e-6) * factor
        importance[:,m.squeeze()] = i + torch.from_numpy(a).cuda() # 设置重要度，从1开始
        m = ~m # 由于m中True是去掉的所以得取反
        m = m.astype(int)

        mask = mask * torch.from_numpy(m).long().to(device) # 把去掉的设置为0
        removed_count += step_size
    # importance[importance == 0] = importance.max() + 1 # 最后剩余的设置为最大的
    # print(torch.sum(importance[importance == 0]))
    # print(torch.sum(importance[importance != 0]))
    return importance.cpu().detach().numpy().squeeze()


def ampe(model, data, target, data_min=0, data_max=1, epsilon=16,N=20,num_steps=10, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = AMPE(model)
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

from saliency.core import FGSMGradSSAKL
def ampe_kl(model, data, target, data_min=0, data_max=1, epsilon=16,N=20,num_steps=10, use_sign=True, use_softmax=True):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    mfaba = AMPE(model)
    epsilon = epsilon / 255
    attack = FGSMGradSSAKL(
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