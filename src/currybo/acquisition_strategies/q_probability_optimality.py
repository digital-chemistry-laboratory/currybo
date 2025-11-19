from typing import Tuple, Type, Optional, Callable
import torch
from torch import Tensor
from botorch.sampling import MCSampler, SobolQMCNormalSampler

from .base import BaseMCAcquisitionStrategy, MCAcquisitionFunction
from .utility_function import BaseMCUtilityFunction, UncertaintyUtility, QuantitativeImprovement
from ..surrogate_models import BaseSurrogate
from ..aggregation_functions import BaseAggregation, Mean
from .utils import create_all_permutations


class QProbabilityOfOptimality(BaseMCAcquisitionStrategy):
    """
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
            **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Identify the next X and W to evaluate by first maximizing the acquisition function of the aggregation metric \
        (over the input feature space X), and then maximizing the acquisition function over all options of W.

        Args:
            model: Trained surrogate model.
            q: The number of points to return.

        Returns:
            Tensor: The next X to evaluate (shape: `q x d`).
            Tensor: The index of the next W to evaluate (shape: `q`).
        """
        if not model.trained:
            raise ValueError("The surrogate model must be trained before calling the acquisition strategy.")

        if 'cli_args' not in kwargs:
            raise ValueError("Please pass `cli_args` as a kwarg into `run_recommendation`")

        cli_args = kwargs.get('cli_args')

        if cli_args['qpo_num_samples'] is None:
            raise ValueError("Please specify --qpo-num-samples")

        """
        This code is not intuitive, so here's an explanation.

        The problem is that, for each combination of x and w, we need to store how many samples
        have this speicific combination as a maximum. For this, we need a hash table that stores this
        number of "wins".

        The problem is now that we need some kind of non-list index for this hashtable. For w, this is not
        a problem - we can just take the w_idx. For x, however, we don't have access to the x index, just its
        value. In fact, the index for a maximum x is so deep within BoTorch methods that it's completely unreasonable
        to patch the BoTorch classes for this. 

        This is why we take the fingerprint of the x value, stringify all values and separate them with a _. Afterwards, we add 
        -{w index} at the back (example: 1_0_0_0_..._1_0-7). This stringified value works as an index for the hash table and 
        allows us to convert it back into meaningful x and w data.

        NOTE: This assumes that the fingerprint values are positive integers incl. 0
        """

        stringify = lambda x, w: '_'.join(map(lambda i: str(int(i)), x)) + ';' + str(int(w))

        w_idx_options = torch.arange(self.w_options.shape[0]).unsqueeze(dim=1)
        all_xw, all_xw_idx = create_all_permutations(self.x_options, w_idx_options)
        x_fingerprint_len = self.x_options.shape[1]
        hashtable = {k: 0 for k in [stringify(option[:x_fingerprint_len], option[-1]) for option in all_xw]}

        for s in range(cli_args['qpo_num_samples']):
            if self.sampler_type is not None:
                sampler = self.sampler_type(sample_shape=torch.Size([1]))
            else:
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]))

            next_x, next_w_idx = acquisition_strategy.get_recommendation(model, **kwargs)

            next_stringified = stringify(next_x[0], next_w_idx[0])

            hashtable[next_stringified] += 1
        
        hashtable_sorted = {k: v for k, v in sorted(hashtable.items(), key=lambda i: i[1], reverse=True)}
        batch_keys = list(hashtable_sorted.keys())[:q]

        next_w_idx = torch.tensor(list(map(lambda xw: int(xw.split(';')[1]), batch_keys)), dtype=torch.int64)
        
        fingerprints_lst = list(map(lambda xw: list(map(int, xw.split(';')[0].split('_'))), batch_keys))
        next_x = torch.tensor(fingerprints_lst, dtype=torch.float64)

        return next_x, next_w_idx
