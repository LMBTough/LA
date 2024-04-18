import torch
import numpy as np


def caculate_total(model, x, label, mask, steps=5, alpha=0.0025, lambda_r=0.01, op="add", add_mask=True ):
    total = None
    x = torch.nn.Parameter(data=x*mask, requires_grad=True)
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    x.requires_grad_(True)
    model.zero_grad()
    start_loss = None
    end_loss = None
    for _ in range(steps): 
        outputs = model(x+torch.from_numpy(np.random.normal(size = x.shape, loc=0.0, scale=0.05)).float().cuda())
        model.zero_grad()
        loss = loss_func(outputs, label)
        regularization_loss = torch.norm(x)
        if op == "minus":
            loss = loss + lambda_r * regularization_loss
            loss.backward(retain_graph=True)
            grad = x.grad * mask
            # print(torch.sign(grad) == 0)
            if add_mask:
                gd_mask = (torch.sign(grad) >0)
                x = x - alpha * torch.sign(grad) * gd_mask
            else:
                x = x - alpha * torch.sign(grad)
        elif op == "add":
            loss = loss - lambda_r * regularization_loss
            loss.backward(retain_graph=True)
            grad = x.grad
            if add_mask:
                gd_mask = (torch.sign(grad) < 0)
                x = x + alpha * torch.sign(grad) * gd_mask
            else:
                x = x + alpha * torch.sign(grad)
        x = x.detach().requires_grad_(True)
        if total == None:
            total = ((alpha * torch.sign(grad)) * grad).detach()
        else:
            total += ((alpha * torch.sign(grad)) * grad).detach()
        model.zero_grad()
        if start_loss == None:
            start_loss = loss.item()
        end_loss = loss.item()
    if op == "minus":
        return total, start_loss - end_loss
    elif op == "add":
        return total, end_loss - start_loss


def caculate_combine(total, x, use_total=True, use_x=True):
    if use_total and use_x:
        total = total + total.min()
        combine = torch.abs(total * x)
    elif use_total:
        combine = total + total.min()
        # combine = torch.abs(total)
    elif use_x:  # taylor
        combine = torch.abs(x)
    combine = combine.cpu().detach().numpy()
    combine_flatten = combine.flatten()
    return combine, combine_flatten


def get_result(model, x, pos, combine, combine_flatten, delta):
    threshold = np.sort(combine_flatten)[pos]
    delta_ = delta.clone()
    delta_[combine < threshold] = 0
    result = model(x+delta_).argmax(-1)
    return result, torch.norm(delta_).item(), delta_


def exp(model, x, label, mask, add_steps=5, minus_steps=0, alpha=0.025, lambda_r=0.01, method="total*delta"):
    if add_steps != 0:
        total_add,add_weight = caculate_total(
            model, x, label, mask=mask, steps=add_steps, op="add", alpha=alpha, lambda_r=lambda_r, add_mask=False)
    if minus_steps != 0:
        total_minus,minus_weight = caculate_total(
            model, x, label, mask=mask, steps=minus_steps, op="minus", alpha=alpha, lambda_r=lambda_r, add_mask=False)
    if add_steps != 0 and minus_steps != 0:
        total = total_add / add_weight + total_minus / minus_weight
    elif add_steps != 0:
        total = total_add
    elif minus_steps != 0:
        total = total_minus
    if method == "total*delta":
        combine, combine_flatten = caculate_combine(
            total, x, use_total=True, use_x=True)
    return combine, combine_flatten