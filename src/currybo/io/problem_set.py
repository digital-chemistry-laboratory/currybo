import itertools
import numpy as np
import torch

from torch import Tensor
from typing import Dict, List, Union

from .dataset import Dataset

class ProblemSet:
    """
    Class that describes a set of related discrete test problems from the same reaction dataset.

    Args:
        noise_std: Standard deviation of the observation noise. If a list is provided, specifies separate noise standard
                   deviations for each objective in a multiobjective problem. Argument as specified in botorch's
                   `BaseTestProblem` class.
        negate: If True, negate the function. Argument as specified in botorch's `BaseTestProblem` class.
        x_options: A dictionary of the different options in the optimization domain. 
                    Keys are the different dimensions, values a list of options for each dimension.
        w_options: A dictionary of the different options in the parameter domain. 
                    Keys are the different parameters, values a list of options for each parameter.
    """
    def __init__(
        self,
        dataset: Dataset,
        noise_std: Union[None, float, List[float]] = None,
        negate: bool = False,
    ):
        
        self.negate = negate
        self.noise_std = noise_std
        self.dataset = dataset

        self._setup_problems(dataset.x_options, dataset.w_options, dataset.x_data, dataset.w_data, dataset.y_data)


    def _setup_problems(
            self, 
            x_options: Dict[str, List[Tensor]] = None,
            w_options: Dict[str, List[Tensor]] = None,
            x_data: Dict[str, List[Tensor]] = None,
            w_data: Dict[str, List[Tensor]] = None,
            y_data: Dict[str, List[Tensor]] = None,
        ):

        """
        Creates the options tensors for the x and w domain.

        Args:
            x_options: A dictionary of the different options in the optimization domain. 
                    Keys are the different dimensions, values a list of options for each dimension.
            w_options: A dictionary of the different options in the parameter domain. 
                    Keys are the different parameters, values a list of options for each parameter.
        """

        for var in ['x_options', 'w_options']:
            combinations = list(itertools.product(*eval(var).values()))
            options_combinations = [torch.cat(combo) for combo in combinations]
            options_tensor = torch.stack(options_combinations)
            setattr(self, var, options_tensor)

        for var in ['x_data', 'w_data']:
            combinations = list(zip(*eval(var).values()))
            options_combinations = [torch.cat(combo) for combo in combinations]
            options_tensor = torch.stack(options_combinations)
            setattr(self, var, options_tensor)

        self.y_data = y_data


    def __len__(self) -> int:
        """
        Returns the number of problems in the family.
        """
        return self.w_options.shape[0]
