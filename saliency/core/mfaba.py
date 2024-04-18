import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class FGSMGradNew:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=50, alpha=0.001, use_softmax=False):
        x = data.clone().detach().requires_grad_(True)
        for _ in range(num_steps):
            x = torch.autograd.Variable(x.data, requires_grad=True)
            output = model(x)
            if use_softmax:
                output = F.softmax(output,dim=-1)
            tgt_out = torch.gather(output, 1, target.view(-1, 1))
            grad = torch.autograd.grad(tgt_out, x, grad_outputs=torch.ones_like(tgt_out), create_graph=True, retain_graph=False)[0]
            print(grad.shape)
            x_new = x + alpha * grad.sign()
            loss_new = grad.norm(2).sum()
            grad_new = torch.autograd.grad(loss_new, x)[0]
            


class FGSMGradKL:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=50, alpha=0.001, early_stop=True, use_sign=False, use_softmax=False):
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            if use_softmax:
                output = F.softmax(output,dim=-1)
            kl_loss = F.kl_div(output.log(),torch.ones_like(output)/1000,reduction="batchmean")
            kl_loss.backward()
            grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(grad[i:i+1].clone())
            if use_sign:
                data_grad = dt.grad.detach().sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon/255, self.epsilon/255)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            dt.grad = None
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        dt.grad = None
        adv_pred = model(dt)
        if use_softmax:
            adv_pred = F.softmax(adv_pred,dim=-1)
        kl_loss = F.kl_div(adv_pred.log(),torch.ones_like(adv_pred)/1000,reduction="batchmean")
        kl_loss.backward()
        grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads

class FGSMGrad:
    def __init__(self, epsilon, data_min, data_max):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max

    def __call__(self, model, data, target, num_steps=50, alpha=0.001, early_stop=True, use_sign=False, use_softmax=False):
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            if use_softmax:
                tgt_out = torch.diag(
                    F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            tgt_out.sum().backward()
            grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(grad[i:i+1].clone())
            if use_sign:
                data_grad = dt.grad.detach().sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon/255, self.epsilon/255)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            dt.grad = None
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        dt.grad = None
        adv_pred = model(dt)
        if use_softmax:
            tgt_out = torch.diag(F.softmax(adv_pred, dim=-1)
                                 [:, target_clone]).unsqueeze(-1)
        else:
            tgt_out = torch.diag(adv_pred[:, target_clone]).unsqueeze(-1)
        tgt_out.sum().backward()
        grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads



class FGSMGradALPHA:
    def __init__(self, epsilon, data_min, data_max,attack_ch="all"):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.attack_ch = attack_ch

    def __call__(self, model, data, target, num_steps=50, alpha=0.001, early_stop=False, use_sign=False, use_softmax=False):
        dt = data.clone().detach().requires_grad_(True)
        alpha = dt.detach() / num_steps
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in range(num_steps):
            output = model(dt)
            model.zero_grad()
            if use_softmax:
                tgt_out = torch.diag(
                    F.softmax(output, dim=-1)[:, target]).unsqueeze(-1)
            else:
                tgt_out = torch.diag(output[:, target]).unsqueeze(-1)
            tgt_out.sum().backward()
            grad = dt.grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(grad[i:i+1].clone())
            if self.attack_ch == "all":
                grad_mask = torch.ones_like(dt)
            elif self.attack_ch == "r":
                grad_mask = torch.zeros_like(dt)
                grad_mask[:,0:1,:,:] = 1
            elif self.attack_ch == "g":
                grad_mask = torch.zeros_like(dt)
                grad_mask[:,1:2,:,:] = 1
            elif self.attack_ch == "b":
                grad_mask = torch.zeros_like(dt)
                grad_mask[:,2:3,:,:] = 1    
            if use_sign:
                data_grad = dt.grad.detach().sign()
                
                
                adv_data = dt - alpha * data_grad * grad_mask
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon/255, self.epsilon/255)
                dt.data = torch.clamp(
                    data + total_grad, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            else:
                data_grad = grad / \
                    grad.view(grad.shape[0], -1).norm(dim=1,
                                                      keepdim=True).view(-1, 1, 1, 1)
                adv_data = dt - alpha * data_grad * 100 * grad_mask
                dt.data = torch.clamp(
                    adv_data, self.data_min, self.data_max)
                for i, idx in enumerate(leave_index):
                    hats[idx].append(dt[i:i+1].data.clone())
            dt.grad = None
            if early_stop:
                adv_pred = model(dt)
                adv_pred_argmax = adv_pred.argmax(-1)
                removed_index = np.where((adv_pred_argmax != target).cpu())[0]
                keep_index = np.where((adv_pred_argmax == target).cpu())[0]
                if len(keep_index) == 0:
                    break
                if len(removed_index) > 0:
                    dt = dt[keep_index, :].detach().requires_grad_(True)
                    data = data[keep_index, :]
                    target = target[keep_index]
                    leave_index = leave_index[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        dt.grad = None
        adv_pred = model(dt)
        if use_softmax:
            tgt_out = torch.diag(F.softmax(adv_pred, dim=-1)
                                 [:, target_clone]).unsqueeze(-1)
        else:
            tgt_out = torch.diag(adv_pred[:, target_clone]).unsqueeze(-1)
        tgt_out.sum().backward()
        grad = dt.grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads

class MFABA:
    def __init__(self, model):
        self.model = model

    def __call__(self, hats, grads):
        # core algorithm
        t_list = hats[1:] - hats[:-1]
        grads = grads[:-1]
        # grads = (grads[:-1] + grads[1:]) / 2
        total_grads = -torch.sum(t_list * grads, dim=0)
        attribution_map = total_grads.unsqueeze(0)
        return attribution_map.detach().cpu().numpy()

