from typing import Tuple, Type, Optional, Callable
import torch
from torch import Tensor
from botorch.sampling import MCSampler, SobolQMCNormalSampler

from .base import BaseMCAcquisitionStrategy, MCAcquisitionFunction
from .utility_function import BaseMCUtilityFunction, UncertaintyUtility, QuantitativeImprovement
from ..surrogate_models import BaseSurrogate
from ..aggregation_functions import BaseAggregation, Mean
from .utils import create_all_permutations

from .sequential_acquisition import SequentialAcquisition

class QSequentialAcquisition(BaseMCAcquisitionStrategy):
    """
    Implementation of the acquisition strategy in which the next X and the next W are picked sequentially. In the first \
    step, the next X is picked by maximizing a conventional BO acquisition function. In the second step, the next W is \
    picked by a maximum uncertainty criterion.

    This implements the algorithm described by Angello et al. (Science 2022, https://doi.org/10.1126/science.adc8743).

    Args:
        x_bounds: A tensor (2 x d) of bounds for each dimension of the input space.
        x_options: A tensor (num_choices x m), where `m` is the representation of chemicals, of options for discrete
                   optimization.
        w_options: A tensor (r x w) of possible objective function parameters, where `r` is the number of objective
                   functions that can be evaluated, and `w` is the number of parameters per objective function.
        aggregation_function: The aggregation metric to use for the acquisition strategy.
        x_utility: The utility function to use for the X optimization step.
        x_utility_kwargs: Keyword arguments to pass to the X utility function.
        w_utility: The utility function to use for the W optimization step.
        w_utility_kwargs: Keyword arguments to pass to the W utility function.
        maximization: If True, the objective should be maximized. Otherwise, it should be minimized.
    """

    def __init__(
            self,
            x_bounds: Optional[Tensor] = None,
            x_options: Optional[Tensor] = None,
            w_options: Tensor = None,
            aggregation_function: BaseAggregation = Mean(),
            sample_reduction: Callable = torch.mean,
            q_reduction: Callable = torch.amax,
            sampler_type: Type[MCSampler] = SobolQMCNormalSampler,
            num_mc_samples: int = 3,
            maximization: bool = True,
            x_utility: Type[BaseMCUtilityFunction] = QuantitativeImprovement,
            x_utility_kwargs: dict = None,
            w_utility: Type[BaseMCUtilityFunction] = UncertaintyUtility,
            w_utility_kwargs: dict = None,
            utility: Type[BaseMCUtilityFunction] = QuantitativeImprovement,
            utility_kwargs: dict = None,
    ):

        super().__init__(
            x_bounds=x_bounds,
            x_options=x_options,
            w_options=w_options,
            aggregation_function=aggregation_function,
            sample_reduction=sample_reduction,
            q_reduction=q_reduction,
            sampler_type=sampler_type,
            num_mc_samples=num_mc_samples,
            maximization=maximization,
            _x_utility_type=x_utility,
            _x_utility_kwargs=x_utility_kwargs or {},
            _w_utility_type=w_utility,
            _w_utility_kwargs=w_utility_kwargs or {},
            _utility=utility,
            _utility_kwargs=utility_kwargs
        )

    def get_recommendation(
            self,
            model: BaseSurrogate,
            acquisition_strategy: BaseMCAcquisitionStrategy,
            q: int = 1,
            **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Identify the next X and W to evaluate by first maximizing the acquisition function of the aggregation metric \
        (over the input feature space X), and then maximizing the acquisition function over all options of W.

        Args:
            model: Trained surrogate model.
            acquisition_strategy: Acquisition Strategy that is called for a single recommendation
            q: The number of points to return.

        Returns:
            Tensor: The next X to evaluate (shape: `q x d`).
            Tensor: The index of the next W to evaluate (shape: `q`).
        """
        if not model.trained:
            raise ValueError("The surrogate model must be trained before calling the acquisition strategy.")

        next_x_total = torch.empty((0, self.x_options.shape[1]))
        next_w_idx_total = torch.empty((0), dtype=torch.int)
        for i in range(q):
            next_x, next_w_idx = acquisition_strategy.get_recommendation(model, **kwargs)
            next_w = self.w_options[next_w_idx]
            next_y = model.posterior(torch.cat((next_x, next_w), dim=-1)).mean

            next_x_total = torch.cat((next_x_total, next_x), dim=0)
            next_w_idx_total = torch.cat((next_w_idx_total, next_w_idx), dim=0)

            if i != q-1:
                model = model.condition_on_observations(next_x, next_y, next_w)

        return next_x_total, next_w_idx_total
