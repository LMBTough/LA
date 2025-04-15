
import numpy as np
from captum.attr import IntegratedGradients
import torch
from torchvision.models import resnet50, inception_v3,vgg16
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class FGSM:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=50, alpha=0.001):
        dt = data.clone().detach().requires_grad_(True)
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            loss = self.criterion(output, target)
            loss.backward()
            data_grad_sign = dt.grad.data.sign()
            adv_data = dt + alpha * data_grad_sign
            total_grad = adv_data - data
            total_grad = torch.clamp(
                total_grad, -self.epsilon/255, self.epsilon/255)
            dt.data = torch.clamp(
                data + total_grad, self.data_min, self.data_max)
        adv_pred = model(dt).argmax(-1)
        success = adv_pred != target
        return dt, success, adv_pred
    


def take_closer_bd(x, y, cls_bd, dis2cls_bd, boundary_points, boundary_labels):
    """Compare and return adversarial examples that are closer to the input

    Args:
        x (np.ndarray): Benign inputs
        y (np.ndarray): Labels of benign inputs
        cls_bd (None or np.ndarray): Points on the closest boundary
        dis2cls_bd ([type]): Distance to the closest boundary
        boundary_points ([type]): New points on the closest boundary
        boundary_labels ([type]): Labels of new points on the closest boundary

    Returns:
        (np.ndarray, np.ndarray): Points on the closest boundary and distances
    """
    if cls_bd is None:
        cls_bd = boundary_points
        dis2cls_bd = np.linalg.norm(np.reshape((boundary_points - x),
                                               (x.shape[0], -1)),
                                    axis=-1)
        return cls_bd, dis2cls_bd
    else:
        d = np.linalg.norm(np.reshape((boundary_points - x), (x.shape[0], -1)),
                           axis=-1)
        for i in range(cls_bd.shape[0]):
            if d[i] < dis2cls_bd[i] and y[i] != boundary_labels[i]:
                dis2cls_bd[i] = d[i]
                cls_bd[i] = boundary_points[i]
    return cls_bd, dis2cls_bd


def boundary_search(model, attacks, data, target, class_num=10,
                    num_steps=50, alpha=0.001):
    dis2cls_bd = np.zeros(data.shape[0]) + 1e16
    bd = None
    batch_boundary_points = None
    batch_success = None
    boundary_points = list()
    success_total = 0
    for attack in attacks:
        c_boundary_points, c_success, _ = attack(
            model, data, target, num_steps=num_steps, alpha=alpha)
        c_boundary_points = c_boundary_points
        batch_success = c_success
        success_total += torch.sum(batch_success.detach())
        if batch_boundary_points is None:
            batch_boundary_points = c_boundary_points.detach(
            ).cpu()
            batch_success = c_success.detach().cpu()
        else:
            for i in range(batch_boundary_points.shape[0]):
                if not batch_success[i] and c_success[i]:
                    batch_boundary_points[
                        i] = c_boundary_points[i]
                    batch_success[i] = c_success[i]
    boundary_points.append(batch_boundary_points)
    boundary_points = torch.cat(boundary_points, dim=0).to(device)
    y_pred = model(boundary_points).cpu().detach().numpy()
    x = data.cpu().detach().numpy()
    y = target.cpu().detach().numpy()
    y_onehot = np.eye(class_num)[y]
    bd, _ = take_closer_bd(x, y, bd,
                           dis2cls_bd, boundary_points.cpu(),
                           np.argmax(y_pred, -1))
    cls_bd = None
    dis2cls_bd = None
    cls_bd, dis2cls_bd = take_closer_bd(x, y_onehot, cls_bd,
                                        dis2cls_bd, bd, None)
    return cls_bd, dis2cls_bd, batch_success


class BIG:
    def __init__(self, model, attacks, class_num=10):
        self.model = model
        self.attacks = attacks
        self.class_num = class_num
        self.saliency = IntegratedGradients(model)

    def __call__(self, model, data, target, gradient_steps=50):
        cls_bd, _, success = boundary_search(
            model, self.attacks, data, target, self.class_num)
        attribution_map = self.saliency.attribute(
            data, target=target, baselines=cls_bd.to(device), n_steps=gradient_steps, method="riemann_trapezoid")
        return attribution_map.cpu().detach().numpy(), success


# @get_time("BIG")
# def big_pipeline(model, data, target, data_min, data_max, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=10, gradient_steps=50):
#     attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
#     big = BIG(model, attacks, class_num)
#     attribution_map, success = big(model, data, target, gradient_steps)
#     return attribution_map, success
