import torch
import torch.nn as nn
import torch.nn.functional as F
def get_grad(image, model, init_pred):
    perturbed_image = image.data.detach().clone()
    perturbed_image.requires_grad = True
    output = model(perturbed_image)
    # loss_lab = output[0, init_pred.item()]
    loss_lab = 0
    for i in range(len(init_pred)):
        loss_lab += output[i, init_pred[i].item()].sum()
    model.zero_grad()
    if perturbed_image.grad is not None:
        perturbed_image.grad.zero_()
    loss_lab.backward()
    data_grad_lab = perturbed_image.grad.data.detach().clone()
    return data_grad_lab

def get_grad_kl(image, model, init_pred):
    perturbed_image = image.data.detach().clone()
    perturbed_image.requires_grad = True
    output = model(perturbed_image)
    # loss_lab = output[0, init_pred.item()]
    # loss_lab = 0
    # for i in range(len(init_pred)):
    #     loss_lab += output[i, init_pred[i].item()].sum()
    loss_lab = F.kl_div(output.log(), torch.ones_like(output) / 1000, reduction="batchmean")
    model.zero_grad()
    if perturbed_image.grad is not None:
        perturbed_image.grad.zero_()
    loss_lab.backward()
    data_grad_lab = perturbed_image.grad.data.detach().clone()
    return data_grad_lab


def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    # data_grad_lab, is_adv = get_grad(image, model, init_pred)
    perturbed_image = image.clone()
    # abs_pos = F.normalize(torch.empty_like(perturbed_image).uniform_())
    # perturbed_image = perturbed_image + epsilon * abs_pos
    perturbed_image = perturbed_image + epsilon * torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon).sign()
    perturbed_image = torch.clamp(perturbed_image, min=0, max=1).detach()

    # c_delta = 0 # cumulative delta
    # data_grad_lab = 0
    data_grad_lab = get_grad(perturbed_image, model, init_pred)
    c_delta = (perturbed_image - image) * data_grad_lab
    # 为什么这里取平方后结果会变好？
    return c_delta*c_delta


def pgd_step_kl(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    # data_grad_lab, is_adv = get_grad(image, model, init_pred)
    perturbed_image = image.clone()
    # abs_pos = F.normalize(torch.empty_like(perturbed_image).uniform_())
    # perturbed_image = perturbed_image + epsilon * abs_pos
    perturbed_image = perturbed_image + epsilon * torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon).sign()
    perturbed_image = torch.clamp(perturbed_image, min=0, max=1).detach()

    # c_delta = 0 # cumulative delta
    # data_grad_lab = 0
    data_grad_lab = get_grad_kl(perturbed_image, model, init_pred)
    c_delta = (perturbed_image - image) * data_grad_lab
    # 为什么这里取平方后结果会变好？
    return c_delta*c_delta