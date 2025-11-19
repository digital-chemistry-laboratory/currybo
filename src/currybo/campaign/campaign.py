from typing import Type, Tuple, Union, Optional, Dict

import time
import numpy as np
from pathlib import Path

import torch
from torch import Tensor

from currybo.surrogate_models import BaseSurrogate
from currybo.acquisition_strategies import BaseMCAcquisitionStrategy, JointLookaheadAcquisition
#from currybo.test_functions import AnalyticalProblemSet, DiscreteProblemSet, MixedProblemSet
from currybo.acquisition_strategies.utility_function import Random

from currybo.io import ProblemSet

class GeneralBOCampaign(object):
    """
    Class for running a general Bayesian Optimization (BO) campaign, which coordinates the entire optimization process 
    including initializing surrogate models, handling acquisition strategies, updating observations, and saving/loading
    campaign data.

    Attributes:
        observations_x (Tensor): Observed input data points.
        observations_w_idx (Tensor): Indices representing the contextual variables or auxiliary data.
        observations_w (Tensor): Observed contextual data values.
        observations_y (Tensor): Observed function evaluations.
        optimum_x (Tensor): Current optimuml input values.
        loaded_from_file (bool): Flag indicating whether the campaign was loaded from a file.
    """

    observations_x: Tensor = None
    observations_w_idx: Tensor = None
    observations_w: Tensor = None
    observations_y: Dict[str, Tensor] = None
    optim_x: Tensor = None
    loaded_from_file: bool = False
    cli_args: Dict = {}

    def __init__(self):

        self.observations = None

        self._surrogate_type = None
        self._surrogate_kwargs = None
        self._acquisition_strategy = None
        self._batch_strategy = None
        self._problem = None
        self._test_problem = None

    @property
    def surrogate_type(self) -> Type[BaseSurrogate]:
        """
        Get the surrogate model type used in the optimization process.

        Raises:
            AttributeError: If the surrogate type has not been set.
        """
        if self._surrogate_type is None:
            raise AttributeError("Surrogate model has not been set.")
        return self._surrogate_type

    @surrogate_type.setter
    def surrogate_type(self, surrogate_type: Type[BaseSurrogate]) -> None:
        """
        Set the surrogate model type for the campaign.
        
        Args:
            surrogate_type (Type[BaseSurrogate]): A surrogate model class that inherits from BaseSurrogate.
        """
        self._surrogate_type = surrogate_type

    @property
    def surrogate_kwargs(self) -> dict:
        """
        Get the keyword arguments for the surrogate model.

        Raises:
            AttributeError: If surrogate kwargs have not been set.
        """
        if self._surrogate_kwargs is None:
            raise AttributeError("Surrogate model keyword arguments have not been set.")
        return self._surrogate_kwargs

    @surrogate_kwargs.setter
    def surrogate_kwargs(self, surrogate_kwargs: dict) -> None:
        """
        Set the keyword arguments for initializing the surrogate model.
        
        Args:
            surrogate_kwargs (dict): Dictionary of keyword arguments for surrogate model initialization.
        """
        self._surrogate_kwargs = surrogate_kwargs

    @property
    def acquisition_strategy(self) -> BaseMCAcquisitionStrategy:
        """
        Get the acquisition strategy used in the optimization process.

        Raises:
            AttributeError: If the acquisition strategy has not been set.
        """
        if self._acquisition_strategy is None:
            raise AttributeError("Acquisition strategy has not been set.")
        return self._acquisition_strategy

    @acquisition_strategy.setter
    def acquisition_strategy(self, acquisition_strategy: BaseMCAcquisitionStrategy) -> None:
        """
        Set the acquisition strategy for the campaign.

        Args:
            acquisition_strategy (BaseMCAcquisitionStrategy): An acquisition strategy that inherits from BaseMCAcquisitionStrategy.
        """
        self._acquisition_strategy = acquisition_strategy

    @property
    def batch_strategy(self) -> BaseMCAcquisitionStrategy:
        """
        Get the batch strategy used in the optimization process.

        Raises:
            AttributeError: If the batch strategy has not been set.
        """
        if self._batch_strategy is None:
            raise AttributeError("Batch strategy has not been set")
        return self._batch_strategy

    @batch_strategy.setter
    def batch_strategy(self, batch_strategy: BaseMCAcquisitionStrategy) -> None:
        """
        Set the batch strategy for the campaign.

        Args:
            batch_strategy (BaseMCAcquisitionStrategy): A batch strategy that inherits from BaseMCAcquisitionStrategy
        """
        self._batch_strategy = batch_strategy

    @property
    def problem(self):
        """
        Get the optimization problem set for the campaign.
        """
        return self._problem

    @problem.setter
    def problem(self, problem: any):
        """
        Set the optimization problem set for the campaign.

        Args:
            problem (ProblemSet): The problem set to optimize.
        
        Raises:
            ValueError: If the problem set is not one of the specified types.
        """
        self._problem = problem

    #@property
    #def test_problem(self):
    #    return self._test_problem

    #@test_problem.setter
    #def test_problem(self, test_problem: any):
    #    self._test_problem = test_problem
    #    if isinstance(self._problem, AnalyticalProblemSet):
    #        self._test_problem_type = "continuous"
    #    elif isinstance(self._problem, DiscreteProblemSet):
    #        self._test_problem_type = "discrete"
    #    elif isinstance(self._problem, MixedProblemSet):
    #        self._test_problem_type = "mixed"
    #    else:
    #        raise ValueError("Problem Set needs to be either continuous, discrete or mixed!")

    #def _generate_seed_data(self, num_seeds: int = 1, random_seed: int = 12) -> Tuple[Tensor, Tensor]:
    #    """
    #    Generate initial seed data for the optimization campaign based on the problem type.

    #    Args:
    #        num_seeds (int): Number of seed data points to generate.
    #        random_seed (int): Random seed for reproducibility.

    #    Returns:
    #        Tuple[Tensor, Tensor]: Generated seed data points and their respective indices.
    #    """
    #    torch.manual_seed(random_seed)
    #    np.random.seed(random_seed)
    #    random.seed(random_seed)

    #    if self._problem_type == "continuous":
    #        seed_x = [torch.FloatTensor(num_seeds).uniform_(*bounds) for bounds in self.problem.bounds.transpose(0, 1)]
    #        seed_w_idx = torch.randint(0, len(self.problem), (num_seeds,))

    #        return torch.stack(seed_x, dim=-1).type(torch.get_default_dtype()), seed_w_idx
    #    elif self._problem_type == "discrete":
    #        seed_x_idx = torch.randint(0, self.problem.x_options.shape[0], (num_seeds,))
    #        seed_x = self.problem.x_options[seed_x_idx]
    #        seed_w_idx = torch.randint(0, len(self.problem), (num_seeds,))

    #        return seed_x, seed_w_idx
    #    else:
    #        seed_continuous = torch.rand(num_seeds, self.problem.continuous_bounds.shape[1]) * (self.problem.continuous_bounds[1] - self.problem.continuous_bounds[0]) + self.problem.continuous_bounds[0]
    #        seed_x_discrete_idx = torch.randint(0, self.problem.x_options.shape[0], (num_seeds,))
    #        seed_x_discrete = self.problem.x_options[seed_x_discrete_idx]
    #        seed_w_idx = torch.randint(0, len(self.problem), (num_seeds,))
    #        seed_x = torch.cat([seed_continuous, seed_x_discrete], dim=1)

    #        seed_w_idx = torch.randint(0, len(self.problem), (num_seeds,))

    #        return seed_x, seed_w_idx

    #def _update_observations(self, next_x: Tensor, next_w_idx: Tensor, **kwargs) -> None:
    #    """
    #    Updates the observations with the new data. Observes the true function value at the new point(s).

    #    Args:
    #        next_x (Tensor): The new X data to observe.
    #        next_w_idx (Tensor): The new W index data to observe.
    #    """
    #    self.observations_x = torch.cat([self.observations_x, next_x], dim=0)
    #    self.observations_w_idx = torch.cat([self.observations_w_idx, next_w_idx], dim=0)
    #    self.observations_w = torch.cat([self.observations_w, self.problem.w_options[next_w_idx]], dim=0)

    #    objective_obs = self.problem.evaluate_true(next_x, next_w_idx)
    #    observations_y = dict_cat(self.observations_y, objective_obs)
    #    self.observations_y = observations_y

    #    # we get an X value for each objective --> dict
    #    new_optimum_x = self.get_optimum(**kwargs)
    #    #optimum_x = dict_cat(self.optimum_x, new_optimum_x)
    #    optimum_x = self.get_optimum(**kwargs)
    #    self.optimum_x = optimum_x

    #    # self.generality_train_set = torch.cat([self.generality_train_set, self.evaluate_generalizability(optimum=self.optimum_x[-1], test_problem=self.problem)], dim = 0)
    #    self.generality_train_set = dict_cat(self.generality_train_set, self.evaluate_generalizability(optimum=self.optimum_x[-1], test_problem=self.problem))
    #    if self.test_problem is not None:
    #        self.generality_test_set = dict_cat(self.generality_test_set, self.evaluate_generalizability(optimum=self.optimum_x[-1], test_problem=self.test_problem))
    #        #self.generality_test_set = torch.cat([self.generality_test_set, self.evaluate_generalizability(optimum=self.optimum_x[-1], test_problem=self.test_problem)], dim = 0)

    def run_optimization(
            self,
            cli_args: Dict = None,
    ) -> None:
        """
        Run the general Bayesian optimization campaign.

        Args:
            budget: Experimental budget (number of experiments to be performed)
            num_seeds: Number of random seed data points to generate before starting the optimization campaign
            batch_size: Number of experiments to generate per iteration
        
        Args:
            budget (int): Experimental budget (number of experiments to be performed).
            num_seeds (int): Number of random seed data points to generate before starting the optimization campaign.
            batch_size (int): Number of experiments to generate per iteration.
            random_seed (int): Random seed for reproducibility.
            single_substrate (bool): If True, only evaluates the first substrate.
            complete_monitoring (bool): If True, evaluates all substrates for one x.
            save_file (str): Optional path to save the campaign's progress.
        """
        self.observations_x = self.problem.x_data
        self.observations_w = self.problem.w_data
        self.observations_y = self.problem.y_data
        self.cli_args = cli_args
        self.optimum_x = self.get_optimum(cli_args = cli_args)
        #self.global_optimum = self.get_global_optimum(test_problem = self.problem)

        #seed_x, seed_w_idx = self._generate_seed_data(num_seeds = num_seeds, random_seed = random_seed)

        #self.observations_x = seed_x
        #self.observations_w_idx = seed_w_idx
        #self.observations_w = self.problem.w_options[seed_w_idx]
        #self.observations_y = self.problem.evaluate_true(seed_x, seed_w_idx)
        #self.optimum_x = self.get_optimum(cli_args=cli_args)
        #self.global_optimum = self.get_global_optimum(test_problem=self.problem)
        
        # for the optimum x: take the latest point for each objective and evaluate it for generalizability
        #self.generality_train_set = self.evaluate_generalizability(optimum = self.optimum_x[-1], test_problem = self.problem)
        #if self.test_problem is not None:
        #    self.generality_test_set = self.evaluate_generalizability(optimum = self.optimum_x[-1], test_problem = self.problem)
        #self.generality_train_set = {obj: self.evaluate_generalizability(optimum=dict_get_index(self.optimum_x, -1)[obj], test_problem=self.problem) for obj in self.optimum_x.keys()}
        #if self.test_problem is not None:
        #    self.generality_test_set = {obj: self.evaluate_generalizability(optimum=dict_get_index(self.optimum_x, -1)[obj], test_problem=self.test_problem) for obj in self.optimum_x.keys()}

        #num_iter = -(-budget // batch_size) # same as rounding up, without importing math
        #for _ in atpbar(range((self.observations_x.shape[0] - num_seeds) // batch_size, num_iter), name=f"Job {job+1}/{cli_args['jobs']}"):
        # while self.observations_x.shape[0] - 1 < budget:
            #if(cli_args['verbose']):
            #    print(f"Job {job+1}/{cli_args['jobs']}, Iteration {self.observations_x.shape[0]}", flush=True)

        if not isinstance(self.acquisition_strategy, JointLookaheadAcquisition):
            if self.acquisition_strategy._w_utility_type == Random:
                self.acquisition_strategy._w_utility_kwargs["random_seed"] = (cli_args['seed'] + 1) * (self.observations_x.shape[0] - 1)

        # print(f"--- STEP {self.observations_x.shape[0]} ---", flush=True)
        start_time = time.time()
        y_comb = torch.stack([self.observations_y[obj].T[0] for obj in self.observations_y.keys()], dim=1)
        surrogate = self.surrogate_type(
            train_X=self.observations_x,
            train_W=self.observations_w,
            train_Y=y_comb,
            cli_args=cli_args,
            dataset=self._problem.dataset,
            **self.surrogate_kwargs
        )
        surrogate.fit()

        # We get a recommendation from the batch strategy, which in turn calls the acquisition function q times.
        next_x, next_w_idx = self.batch_strategy.get_recommendation(surrogate, q = cli_args['batch_size'], acquisition_strategy = self.acquisition_strategy, cli_args=cli_args)
        # For single substrate benchmark, only ever measure the first substrate
        #if single_substrate:
        #    next_w_idx = torch.zeros_like(next_w_idx)
        # For complete monitoring benchmark, measure all substrates on one x
        #if complete_monitoring:
        #    for j in range(len(self.problem)):
        #        next_w_idx = torch.tensor([j])
        #        self._update_observations(next_x, next_w_idx, cli_args=cli_args)
        #    if save_file is not None:
        #        self.save(save_file)
        #    continue
        #self._update_observations(next_x, next_w_idx, cli_args=cli_args)

        #if save_file is not None:
        #    self.save(save_file)

        end_time = time.time()

        next_w = [self.problem.w_options[i] for i in next_w_idx]
        next_w_t = torch.stack(next_w)
        estimate = surrogate.posterior(next_x, next_w_t)

        optimum_y = self.evaluate_generalizability(self.optimum_x, self.problem.w_options, surrogate)
        output = self.problem.dataset.generate_output(next_x, next_w, self.optimum_x, optimum_y, estimate, cli_args)
        #self.problem.dataset.print_output(output, cli_args)

        return output
        # print(f"Time: {end_time - start_time}", flush=True)

        #bar(cli_args['batch_size'])

    def get_optimum(self, **kwargs) -> Tensor:
        """
        Return the current optimum as the x value which maximizes the mean of the predictive posterior distribution of
        the aggregation score. This will also use MOBO down the road.

        Returns:
            Tensor: The current optimum (Shape: `1, d`).
        """
        # NOTE: We train the surrogate model on multiple columns of y here.
        y_stacked = torch.cat(list(self.observations_y.values()), dim=1)
        surrogate = self.surrogate_type(
            train_X = self.observations_x,
            train_W = self.observations_w,
            train_Y = y_stacked,
            cli_args = self.cli_args,
            dataset=self._problem.dataset,
            **self.surrogate_kwargs
        )

        surrogate.fit()

        return self.acquisition_strategy.get_final_recommendation(surrogate, **kwargs)
        #def get_optimum(objective):
        #    surrogate = self.surrogate_type(
        #        train_X=self.observations_x,
        #        train_W=self.observations_w,
        #        train_Y=self.observations_y[objective],
        #        **self.surrogate_kwargs
        #    )

        #    surrogate.fit()

        #    return self.acquisition_strategy.get_final_recommendation(surrogate, **kwargs)

        #return {obj: get_optimum(obj) for obj in self.observations_y.keys()}

    def evaluate_generalizability(self, optimum: Tensor, w_options: Tensor, surrogate: BaseSurrogate) -> Tensor:
        """
        Evaluate the generalizability (i.e. the aggregation metric) of a set of x values
        on the held-out test problem set.

        This function acts on any number of X points.

        Args:
            optimum (Tensor): Tensor of x values to evaluate (Shape: `n, d`).
            test_problem: Test problem set to evaluate generalizability.

        Returns:
            Tensor: The generalizability of the x values (Shape: `n`).  # TODO: Right output?
        """
        if optimum.ndimension() == 1:
            optimum = optimum.unsqueeze(0)

        y_points = {obj: [] for obj in self.observations_y.keys()}
        for i in range(len(w_options)):
            ev = surrogate.posterior(optimum, w_options[i].unsqueeze(0))
            for j, obj in enumerate(y_points.keys()):
                mean = ev.mean[0][j].unsqueeze(0)
                y_points[obj].append(mean)

        y_true = {obj: torch.cat(y_points[obj], dim=0) for obj in y_points.keys()}
        y_true = {obj: y_true[obj].reshape(len(w_options), optimum.shape[0]).permute(1, 0) for obj in y_points.keys()}
        y_agg = {obj: self.acquisition_strategy.aggregation_function(y_true[obj]).unsqueeze(1) for obj in y_points.keys()}
        return y_agg
    #
    #def get_global_optimum(self, test_problem: ProblemSet):
    #    """
    #    Gets the global optimum for a specified problem set.

    #    Args:
    #        test_problem: The problem set for which to find the global optimum.
    #    """

    #    generalizability = self.evaluate_generalizability(optimum=test_problem.x_options, test_problem=test_problem)
    #    
    #    # TODO: This needs some BoTier magic
    #    #global_optimum_X = test_problem.x_options[torch.argmax(generalizability)]
    #    #global_optimum = self.evaluate_generalizability(optimum=global_optimum_X, test_problem=test_problem)

    #    return torch.tensor([0])

    def save(self, file: Union[str, Path]) -> None:
        """
        Saves the campaign to a file.

        Args:
            file: The file to save the campaign to.
        """
        torch.save(
            {
                "observations_x": getattr(self, "observations_x", None),
                "observations_w_idx": getattr(self, "observations_w_idx", None),
                "observations_w": getattr(self, "observations_w", None),
                "observations_y": getattr(self, "observations_y", None),
                "optimum_x": getattr(self, "optimum_x", None),
                "global_optimum": getattr(self, "global_optimum", None),
                "generality train set": getattr(self, "generality_train_set", None),
                "generality test set": getattr(self, "generality_test_set", None),
                "x_options": getattr(self.problem, "x_options", None) if hasattr(self, "problem") else None,
            },
            file
        )

    def load_from_file(self, file: Union[str, Path]) -> None:
        """
        Loads the campaign from a specified file.

        Args:
            file: The file from which to load the campaign from.
        """

        if not file.endswith('.pt'):
            print(f"Error: Loaded file {file} needs to be a .pt file")
            return

        try:
            loaded_campaign = torch.load(file)
        except FileNotFoundError:
            print(f"Error: File {file} does not exist!")
            return
        except Exception as e:
            print(f"Error loading campaign: {e}")
            return

        # Required keys with their default values
        default_values = {
            "observations_x": None,
            "observations_w_idx": None,
            "observations_w": None,
            "observations_y": None,
            "optimum_x": None,
            "global_optimum": None,
            "generality train set": None,
            "generality test set": None,
        }

        # Assign values from loaded data, or default if key is missing
        for key, default in default_values.items():
            setattr(self, key.replace(" ", "_"), loaded_campaign.get(key, default))

        self.loaded_from_file = True
