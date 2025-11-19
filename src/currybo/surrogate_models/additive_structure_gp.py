from typing import Type, Optional, List, Dict, Any

from gauche.kernels.fingerprint_kernels import TanimotoKernel
from gpytorch.kernels import RBFKernel
import torch
import sys

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.fit import fit_gpytorch_mll

from gpytorch.kernels.kernel import Kernel
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood

from currybo.io import Dataset
from currybo.surrogate_models.base import BaseSurrogate


class AdditiveStructureGP(BaseSurrogate, SingleTaskGP):
    """
    A GP model that uses a sum of multiple Kernels to model the features and the objective function parameters separately.

    Args:
        train_X (torch.Tensor): A `n x d` tensor of training features
        train_W (torch.Tensor): A `n x w` tensor of objective function parameters for each training data point
        train_Y (torch.Tensor): A `n x t` tensor of training observations.
        x_kernel (Optional[Kernel]): A kernel to use for the features.
        w_kernel (Kernel): A kernel to use for the objective function parameters.
        likelihood (Likelihood): A likelihood to use for the model.
        normalize_inputs (bool): True if the input data should be normalized (using a botorch InputTransform).
        standardize_outcomes (bool): True if the output data should be standardized (using a botorch OutcomeTransform).

    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_W: torch.Tensor,
        train_Y: torch.Tensor,
        kernel: Type[Kernel],
        likelihood: Type[Likelihood],
        dataset: Dataset,
        normalize_inputs: bool = True,
        standardize_outcomes: bool = True,
        cli_args: Dict = {},
        # x_kernels: Optional[List[Type[Kernel]]],
        # x_kernel_kwargs: List[Dict[str, Any]],
        # w_kernels: Optional[List[Type[Kernel]]],
        # w_kernel_kwargs: List[Dict[str, Any]],
        # likelihood: Type[Likelihood],
        # normalize_inputs: Optional[bool] = True,
        # standardize_outcomes: Optional[bool] = True
    ):

        train_inputs = torch.cat((train_X, train_W), dim=-1)

        input_transform = Normalize(train_inputs.shape[-1]) if normalize_inputs else None
        outcome_transform = Standardize(m=train_Y.shape[-1]) if standardize_outcomes else None

        # Preliminary transformation of the input data to set the _aug_batch_shape attribute
        with torch.no_grad():
            train_X_transformed = self.transform_inputs(train_inputs, input_transform=input_transform)
        self._set_dimensions(train_X=train_X_transformed, train_Y=train_Y)

        smiles_kernel = cli_args['surrogate_kwargs']['smiles-kernel'] if "smiles-kernel" in cli_args['surrogate_kwargs'] else TanimotoKernel
        array_kernel = cli_args['surrogate_kwargs']['array-kernel'] if "array-kernel" in cli_args['surrogate_kwargs'] else RBFKernel
        scalar = cli_args['surrogate_kwargs']['scalar-kernel'] if "scalar-kernel" in cli_args['surrogate_kwargs'] else RBFKernel

        kernels = []
        for i in cli_args['conditions']:
            if(i['type'] == "smiles"):
                length = dataset.x_options[i['name']][0].shape[0]
                k = eval(cli_args['surrogate_kwargs'][i['name']]) if i['name'] in cli_args['surrogate_kwargs'] else TanimotoKernel
                kernels.append({
                    'name': i['name'],
                    'length': length,
                    'kernel': k
                })
            elif(i['type'] == "array"):
                length = len(dataset.x_options[i['name']][0])
                k = eval(cli_args['surrogate_kwargs'][i['name']]) if i['name'] in cli_args['surrogate_kwargs'] else RBFKernel
                kernels.append({
                    'name': i['name'],
                    'length': length,
                    'kernel': k
                })
            else:
                k = eval(cli_args['surrogate_kwargs'][i['name']]) if i['name'] in cli_args['surrogate_kwargs'] else RBFKernel
                kernels.append({
                    'name': i['name'],
                    'length': 1,
                    'kernel': k
                })

        for i in cli_args['substrates']:
            if(i['type'] == "smiles"):
                length = dataset.w_options[i['name']][0].shape[0]
                k = eval(cli_args['surrogate_kwargs'][i['name']]) if i['name'] in cli_args['surrogate_kwargs'] else TanimotoKernel
                kernels.append({
                    'name': i['name'],
                    'length': length,
                    'kernel': k
                })
            elif(i['type'] == "array"):
                length = len(dataset.w_options[i['name']][0])
                k = eval(cli_args['surrogate_kwargs'][i['name']]) if i['name'] in cli_args['surrogate_kwargs'] else RBFKernel
                kernels.append({
                    'name': i['name'],
                    'length': length,
                    'kernel': k
                })
            else:
                k = eval(cli_args['surrogate_kwargs'][i['name']]) if i['name'] in cli_args['surrogate_kwargs'] else RBFKernel
                kernels.append({
                    'name': i['name'],
                    'length': 1,
                    'kernel': k
                })

        combined_kernel = kernels[0]['kernel'](active_dims=list(range(kernels[0]['length'])))
        last = kernels[0]['length']
        for k in kernels[1:]:
            dims = list(range(last, last + k['length']))
            new_kernel = k['kernel'](active_dims = dims)
            last += k['length'] 
            combined_kernel += new_kernel

        covar_module = ScaleKernel(
            combined_kernel,
            batch_shape=self._aug_batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

        SingleTaskGP.__init__(
            self,
            train_X=train_inputs,
            train_Y=train_Y,
            likelihood=likelihood(),
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform
        )

        self._subset_batch_dict = {
            "mean_module.raw_constant": -1,
            "covar_module.raw_outputscale": -1,
        }

    def fit(self) -> None:
        """Fit the surrogate model to the training data."""
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        fit_gpytorch_mll(mll)
        self.trained = True

    def posterior(self, X: torch.Tensor, W: Optional[torch.Tensor] = None) -> GPyTorchPosterior:
        """
        Get the posterior distribution at a set of points.

        Args:
            X (torch.Tensor): A `m x d` tensor of points at which to evaluate the posterior.
            W (torch.Tensor): A `m x w` tensor of objective function parameters for each point in `X`.

        Returns:
            A GPyTorchPosterior object representing the posterior distribution at the given points.
        """
        if W is None:
            test_inputs = X
        else:
            test_inputs = torch.cat((X, W), dim=-1)
        return SingleTaskGP.posterior(self, test_inputs)

    def condition_on_observations(
            self, X: torch.Tensor, Y: torch.Tensor, W: Optional[torch.Tensor] = None
    ) -> BaseSurrogate:
        """
        Return a new model that is conditioned on new observed data points (X, W; Y).

        Args:
            X (torch.Tensor): A `n x d` tensor of design points to condition on.
            Y (torch.Tensor): A `n x t` tensor of observed outcomes corresponding to `X`.
            W (torch.Tensor): A `n x w` tensor of objective function parameters for each point in `X`.
        """
        if W is None:
            new_inputs = X
        else:
            new_inputs = torch.cat((X, W), dim=-1)

        return SingleTaskGP.condition_on_observations(self, X=new_inputs, Y=Y)
