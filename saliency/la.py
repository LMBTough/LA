import random
import torch
import numpy as np
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def la(model, data, target, epsilon=10, max_iter=30, select_num=20):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected = np.random.choice(1000, select_num, replace=False)
    output = model(data)
    init_pred = output.argmax(-1)
    epsilon = data.clone() / epsilon
    attribution_result = torch.zeros_like(data)
    for l in selected[:-1]:
        targeted = torch.tensor([l] * data.shape[0]).to(device)
        targeted = torch.where(targeted == init_pred, l + 1, targeted) if l < 999 else torch.where(targeted == init_pred, l - 1, targeted)
        ori_image = data.clone()
        adv_image = data.clone()
        result = torch.zeros_like(data)
        for _ in range(max_iter):
            adv_image = torch.autograd.Variable(adv_image, requires_grad=True)
            output = model(adv_image)
            prob = F.softmax(output, dim=-1)
            loss_target = prob[:, targeted].sum()
            model.zero_grad()
            loss_target.backward(retain_graph=True)
            target_grad = adv_image.grad.data.detach().clone()
            
            loss_untarget = prob[:, init_pred].sum()
            model.zero_grad()
            adv_image.grad.zero_()
            loss_untarget.backward()
            untarget_grad = adv_image.grad.data.detach().clone()
            
            delta = epsilon * target_grad.sign()
            adv_image = ori_image + delta
            adv_image = torch.clamp(adv_image, min=0, max=1)
            delta = adv_image - ori_image
            delta = - untarget_grad * delta
            result += delta
            
        attribution_result += result
        
        
    ori_image = data.clone()
    adv_image = data.clone()
    result = torch.zeros_like(data)
    for _ in range(max_iter):
        adv_image = torch.autograd.Variable(adv_image, requires_grad=True)
        output = model(adv_image)
        prob = F.softmax(output, dim=1)
        loss_untarget = prob[:, init_pred].sum()
        model.zero_grad()
        loss_untarget.backward()
        untarget_grad = adv_image.grad.data.detach().clone()
        adv_image.grad.zero_()
        
        delta = - epsilon * untarget_grad.sign()
        adv_image = ori_image + delta
        adv_image = torch.clamp(adv_image, min=0, max=1)
        delta = adv_image - ori_image
        delta = - untarget_grad * delta
        result += delta
        
    attribution_result += result
    attribution_result = attribution_result.squeeze().detach().cpu().numpy()
    return attribution_result
