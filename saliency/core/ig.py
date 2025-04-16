#!/usr/bin/env python3
import typing
from typing import Any, Callable, List, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.batching import _batch_attribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
from captum.log import log_usage
from torch import Tensor


class IntegratedGradients(GradientAttribution):

    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = True,
    ) -> None:

        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs


    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[False] = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        *,
        return_convergence_delta: Literal[True],
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]:
        ...

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        _validate_input(inputs, baselines, n_steps, method)


        attributions = self._attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            method=method,
        )

        return _format_output(is_inputs_tuple, attributions)

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
    ) -> Tuple[Tensor, ...]:
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)

        scaled_features_tpl = torch.cat(scaled_features_tpl, 0)
        # output = self.forward_func(scaled_features_tpl)
        output = list()
        bs = 4
        for i in range(0, scaled_features_tpl.shape[0], bs):
            output.append(self.forward_func(scaled_features_tpl[i:i+bs]))
        output = torch.cat(output, 0)
        output_tgt = torch.gather(output, 1, expanded_target.unsqueeze(1))
        grads = (torch.autograd.grad(output_tgt, scaled_features_tpl, create_graph=True,retain_graph=False,grad_outputs=torch.ones_like(output_tgt))[0].detach(),)
        
        
        # grads = self.gradient_func(
        #     forward_fn=self.forward_func,
        #     inputs=scaled_features_tpl,
        #     target_ind=expanded_target,
        #     additional_forward_args=input_additional_args,
        # )
        # print(grads[1].shape)
        # raise ValueError

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )
        return attributions

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs




class IntegratedGradient:
    def __init__(self, model):
        self.model = model
        self.saliency = IntegratedGradients(model)

    def __call__(self, data, target, gradient_steps=50,baselines=None):
        attribution_map = self.saliency.attribute(data,
                                                  target=target.squeeze(),
                                                  baselines=baselines,
                                                  n_steps=gradient_steps,
                                                  method="riemann_trapezoid")
        return attribution_map.detach().cpu().numpy()
    
