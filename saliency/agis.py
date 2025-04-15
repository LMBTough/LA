from saliency.core import pgd_step
import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def agi(model, data, target, epsilon=0.05, max_iter=20, topk=20):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = random.sample(list(range(0, 999)), topk)
    output = model(data)
    init_pred = output.argmax(-1)

    top_ids = selected_ids  # only for predefined ids
    step_grad = 0
    for l in top_ids:
        targeted = torch.tensor([l] * data.shape[0]).to(device)
        if l < 999:
            targeted[targeted == init_pred] = l + 1
        else: 
            targeted[targeted == init_pred] = l - 1

        delta, perturbed_image = pgd_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()  # / topk
    return adv_ex
