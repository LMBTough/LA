import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable as V
from scipy import stats as st
features = None
import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import numpy as np
import torch
def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)

def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)

"""Translation-Invariant https://arxiv.org/abs/1904.02884"""
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

"""Input diversity: https://arxiv.org/abs/1803.06978"""
def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret

def DIResize(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    ret = F.interpolate(ret, size=[img_size, img_size], mode='bilinear', align_corners=False)
    return ret



def hook_feature(module, input, output):
    global features
    features = output


def image_transform(x):
    return x


def get_NAA_loss(adv_feature, base_feature, weights):
    gamma = 1.0
    attribution = (adv_feature - base_feature) * weights
    blank = torch.zeros_like(attribution)
    positive = torch.where(attribution >= 0, attribution, blank)
    negative = torch.where(attribution < 0, attribution, blank)
    # Transformation: Linear transformation performs the best
    balance_attribution = positive + gamma * negative
    loss = torch.sum(balance_attribution) / \
        (base_feature.shape[0]*base_feature.shape[1])
    return loss


def normalize(grad, opt=2):
    if opt == 0:
        nor_grad = grad
    elif opt == 1:
        abs_sum = torch.sum(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        nor_grad = grad/abs_sum
    elif opt == 2:
        square = torch.sum(torch.square(grad), dim=(1, 2, 3), keepdim=True)
        nor_grad = grad/torch.sqrt(square)
    return nor_grad


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / \
        (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
    stack_kern = np.expand_dims(stack_kern, 3)
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, kern_size):
    x = torch.nn.functional.pad(
        x, (kern_size, kern_size, kern_size, kern_size), "constant", 0)
    x = torch.nn.functional.conv2d(
        x, stack_kern, stride=1, padding=0, groups=3)
    return x


"""Input diversity: https://arxiv.org/abs/1803.06978"""


def input_diversity(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize,
                        size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(
        x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(),
                            size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(),
                             size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(
    ), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret


P_kern, kern_size = project_kern(3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_kernel = gkern(7, 3)


class FGSMGradSSA:
    def __init__(self, epsilon, data_min, data_max,N=20):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.N = N

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        image_width = 224
        momentum = 1.0
        alpha = self.epsilon / num_steps
        grad = 0
        rho = 0.5
        # N = 20
        N = self.N
        sigma = 16
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in tqdm(range(num_steps)):
            model.zero_grad()
            noise = 0
            for n in tqdm(range(N)):
                gauss = torch.randn(dt.size()[0], 3, image_width, image_width) * (sigma / 255)
                gauss = gauss.cuda()
                dt_dct = dct_2d(dt + gauss).cuda()
                mask = (torch.rand_like(dt) * 2 * rho + 1 - rho).cuda()
                dt_idct = idct_2d(dt_dct * mask)
                dt_idct = V(dt_idct, requires_grad = True)

                # DI-FGSM https://arxiv.org/abs/1803.06978
                output_v3 = model(DIResize(dt_idct))
                if use_softmax:
                    output_v3 = F.softmax(output_v3, dim=-1)
                loss = F.cross_entropy(output_v3, target)
                loss.backward()
                noise += dt_idct.grad.data
            noise = noise / N
            # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
            noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

            # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
            noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            noise = momentum * grad + noise
            grad = noise
            grad = grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(grad[i:i+1].clone())
            if use_sign:
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
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
                    grad = grad[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(dt)
        model.zero_grad()
        grad = 0
        noise = 0
        for n in range(N):
            gauss = torch.randn(dt.size()[0], 3, image_width, image_width) * (sigma / 255)
            gauss = gauss.cuda()
            dt_dct = dct_2d(dt + gauss).cuda()
            mask = (torch.rand_like(dt) * 2 * rho + 1 - rho).cuda()
            dt_idct = idct_2d(dt_dct * mask)
            dt_idct = V(dt_idct, requires_grad = True)

            # DI-FGSM https://arxiv.org/abs/1803.06978
            output_v3 = model(DIResize(dt_idct) if "MaxVit" in str(model) else DI(dt_idct))
            if use_softmax:
                output_v3 = F.softmax(output_v3, dim=-1)
            loss = F.cross_entropy(output_v3, target_clone)
            loss.backward()
            noise += dt_idct.grad.data
        noise = noise / N
        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        grad = grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads

import torch.nn.functional as F
class FGSMGradSSAKL:
    def __init__(self, epsilon, data_min, data_max,N=20):
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        self.data_min = data_min
        self.data_max = data_max
        self.N = N

    def __call__(self, model, data, target, num_steps=10, early_stop=False, use_sign=False, use_softmax=False):
        image_width = 224
        momentum = 1.0
        alpha = self.epsilon / num_steps
        grad = 0
        rho = 0.5
        # N = 20
        N = self.N
        sigma = 16
        dt = data.clone().detach().requires_grad_(True)
        target_clone = target.clone()
        hats = [[data[i:i+1].clone()] for i in range(data.shape[0])]
        grads = [[] for _ in range(data.shape[0])]
        leave_index = np.arange(data.shape[0])
        for _ in tqdm(range(num_steps)):
            model.zero_grad()
            noise = 0
            for n in tqdm(range(N)):
                gauss = torch.randn(dt.size()[0], 3, image_width, image_width) * (sigma / 255)
                gauss = gauss.cuda()
                dt_dct = dct_2d(dt + gauss).cuda()
                mask = (torch.rand_like(dt) * 2 * rho + 1 - rho).cuda()
                dt_idct = idct_2d(dt_dct * mask)
                dt_idct = V(dt_idct, requires_grad = True)

                # DI-FGSM https://arxiv.org/abs/1803.06978
                # output_v3 = model(DI(dt_idct))
                # 如果模型是maxvit_t，需要将DI改为DIResize
                output_v3 = model(DIResize(dt_idct) if "maxvit_t" in str(model) else DI(dt_idct))
                if use_softmax:
                    output_v3 = F.softmax(output_v3, dim=-1)
                # loss = F.cross_entropy(output_v3, target)
                loss = -F.kl_div(output_v3.log(), torch.ones_like(output_v3) / 1000, reduction="batchmean")
                loss.backward()
                noise += dt_idct.grad.data
            noise = noise / N
            # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
            noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

            # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
            noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            noise = momentum * grad + noise
            grad = noise
            grad = grad.detach()
            for i, idx in enumerate(leave_index):
                grads[idx].append(grad[i:i+1].clone())
            if use_sign:
                data_grad = grad.sign()
                adv_data = dt - alpha * data_grad
                total_grad = adv_data - data
                total_grad = torch.clamp(
                    total_grad, -self.epsilon, self.epsilon)
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
                    grad = grad[keep_index]
        dt = [hat[-1] for hat in hats]
        dt = torch.cat(dt, dim=0).requires_grad_(True)
        adv_pred = model(dt)
        model.zero_grad()
        grad = 0
        noise = 0
        for n in range(N):
            gauss = torch.randn(dt.size()[0], 3, image_width, image_width) * (sigma / 255)
            gauss = gauss.cuda()
            dt_dct = dct_2d(dt + gauss).cuda()
            mask = (torch.rand_like(dt) * 2 * rho + 1 - rho).cuda()
            dt_idct = idct_2d(dt_dct * mask)
            dt_idct = V(dt_idct, requires_grad = True)

            # DI-FGSM https://arxiv.org/abs/1803.06978
            output_v3 = model(DIResize(dt_idct) if "MaxVit" in str(model) else DI(dt_idct))
            if use_softmax:
                output_v3 = F.softmax(output_v3, dim=-1)
            # loss = F.cross_entropy(output_v3, target_clone)
            loss = -F.kl_div(output_v3.log(), torch.ones_like(output_v3) / 1000, reduction="batchmean")
            loss.backward()
            noise += dt_idct.grad.data
        noise = noise / N
        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        grad = grad.detach()
        for i in range(grad.shape[0]):
            grads[i].append(grad[i:i+1].clone())
        hats = [torch.cat(hat, dim=0) for hat in hats]
        grads = [torch.cat(grad, dim=0) for grad in grads]
        success = adv_pred.argmax(-1) != target_clone
        return dt, success, adv_pred, hats, grads


class AMPE:
    def __init__(self, model):
        self.model = model

    def __call__(self, hats, grads):
        t_list = hats[1:] - hats[:-1]
        grads = grads[:-1]
        total_grads = -torch.sum(t_list * grads, dim=0)
        attribution_map = total_grads.unsqueeze(0)
        return attribution_map.detach().cpu().numpy()
