import argparse
import time
import os
import random
import torch
import sys
import multiprocessing

import numpy as np
import tomllib as toml

from typing import Dict, Tuple, List

from importlib import resources as impresources

from joblib import Parallel, delayed
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from gauche.kernels.fingerprint_kernels import TanimotoKernel
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood

from currybo.campaign import GeneralBOCampaign
from currybo.surrogate_models import SimpleGP, AdditiveStructureGP
from currybo.acquisition_strategies import Random, SimpleRegret, UncertaintyUtility, QuantileUtility, QuantitativeImprovement, QualitativeImprovement
from currybo.acquisition_strategies import QSequentialAcquisition, QProbabilityOfOptimality
from currybo.acquisition_strategies import SequentialAcquisition, SequentialLookaheadAcquisition, JointLookaheadAcquisition
from currybo.aggregation_functions import Mean, Sigmoid, MSE, Min
from currybo.acquisition_strategies.utility_function import UncertaintyUtility, QuantileUtility, Random

from operator import itemgetter
from currybo.io import Dataset, ProblemSet, print_output
#from currybo.test_functions import ChemistryDatasetLoader, DiscreteProblemSet, presets, load_proxy_model
#from currybo.test_functions import CernakLoader, DenmarkLoader, DenmarkMOBOLoader, DoyleLoader, DeoxyfluorinationLoader, BorylationLoader

from currybo.utils import keyval_to_dict

from currybo.visualizer import visualize_surface

import warnings
warnings.filterwarnings('ignore')

multiprocessing.set_start_method('fork', force=True)

def parse_arguments(interface: Dict, name: str, desc: str, omit_key: str = "omit_core"):
    argparse_kwargs = {
        "help": {},
        "type": {
            "eval": True,
        },
        "choices": {},
        "action": {},
        "default": {},
        "nargs": {},
    }

    parser = argparse.ArgumentParser(prog = name, description = desc)

    for argument in interface:
        # these are strings for the parser itself
        if argument in ["name", "desc"]:
            continue

        params = interface[argument]

        if omit_key in params and params[omit_key]:
            continue

        kwarg_keys = [key for key in params if key in argparse_kwargs]
        def parse_kwarg(key):
            if "eval" in argparse_kwargs[key] and argparse_kwargs[key]:
                return eval(params[key])
            return params[key]
        kwargs = {key: parse_kwarg(key) for key in kwarg_keys}

        parser.add_argument(params['arg'], **kwargs)

    return vars(parser.parse_args())


def load_interface():
    interface_path = impresources.files(__package__) / 'interface.toml'
    with open(interface_path, "rb") as f:
        interface = toml.load(f)
    return interface


def cli():
    interface = load_interface()
    args = parse_arguments(interface, "currybo", "Find general parameters in synthesis using Bayesian Optimization")

    # Data Sources
    # --> specify data file or take a preset (Denmark, Cernak, etc.)
    if args['measurements'] is None or args['options'] is None:
        raise ValueError("Please specify --measurements and --options")

    if not 'conditions' in args or not 'substrates' in args or not 'targets' in args:
        raise ValueError("Please specify all --conditions, --substrates and --targets")

    args['conditions'] = [keyval_to_dict(val) for val in args['conditions']]
    args['targets'] = [keyval_to_dict(val) for val in args['targets']]
    args['substrates'] = [keyval_to_dict(val) for val in args['substrates']]
    args['objectives'] = [keyval_to_dict(val) for val in args['objectives']]

    # kwargs to utilities
    args['x_utility_kwargs'] = keyval_to_dict(args['x_utility_kwargs'])
    args['w_utility_kwargs'] = keyval_to_dict(args['w_utility_kwargs'])
    args['utility_kwargs'] = keyval_to_dict(args['utility_kwargs'])
    args['surrogate_kwargs'] = keyval_to_dict(args['surrogate_kwargs'])
    args['aggregation_kwargs'] = keyval_to_dict(args['aggregation_kwargs'])

    run(args)


def run(args):
    dataset = load_dataset(args)

    result = run_campaign(dataset, args)

    if not args['silent']:
        print_output(result, dataset, args) 

    return result

def load_dataset(cli_args: Dict) -> Tuple[Dataset, List[int]]:
    input_params = cli_args['conditions']
    output_params = cli_args['targets']
    w_columns = cli_args['substrates']

    ds = Dataset()
    ds.load_data(cli_args['measurements'], cli_args['options'])
    ds.preprocess_data(input_params, output_params, w_columns)

    return ds


#def run_parallel_campaign(dataset: ChemistryDatasetLoader, cli_args: Dict, samples: List[int]):
#    os.makedirs("runs", exist_ok=True)
#
#    res = [None for i in range(cli_args['jobs'])]
#
#    # do not use parallel features for only one worker. easier for debugging.
#    if(cli_args['workers'] > 1):
#        with flushing(), ProcessPoolExecutor(max_workers=cli_args['workers'], initializer=register_reporter, initargs=(find_reporter(),)) as executor:
#            for i in range(cli_args['jobs']):
#                res[i] = executor.submit(run_campaign, dataset=dataset, job=i, cli_args=cli_args, samples=samples)
#    else:
#        for i in range(cli_args['jobs']):
#            res[i] = run_campaign(dataset=dataset, job=i, cli_args=cli_args, samples=samples)
#
#    return res
     

def run_campaign(dataset: Dataset, cli_args: Dict):
    torch.manual_seed(cli_args['seed'])
    np.random.seed(cli_args['seed'])
    random.seed(cli_args['seed'])

    campaign = GeneralBOCampaign()

    # this will convert all x_ and w_options to tensors that include all possible combinations
    campaign.problem = ProblemSet(dataset)

    campaign.surrogate_type = eval(cli_args['surrogate'])
    campaign.surrogate_kwargs = {"kernel": eval(cli_args['kernel']), "likelihood": eval(cli_args['likelihood'])}

    campaign.acquisition_strategy = eval(cli_args['acquisition'])(
        x_bounds=None,
        x_options=campaign.problem.x_options,
        w_options=campaign.problem.w_options,
        aggregation_function=eval(cli_args['aggregation'])(**cli_args['aggregation_kwargs']),
        x_utility=eval(cli_args['x_utility']),
        x_utility_kwargs=cli_args['x_utility_kwargs'],
        w_utility=eval(cli_args['w_utility']),
        w_utility_kwargs=cli_args['w_utility_kwargs'],
        utility=eval(cli_args['utility']),
        utility_kwargs=cli_args['utility_kwargs'],
        maximization=True,
        cli_args=cli_args
    )

    campaign.batch_strategy = eval(cli_args['batch_strategy'])(
        x_bounds=None,
        x_options=campaign.problem.x_options,
        w_options=campaign.problem.w_options,
        aggregation_function=eval(cli_args['aggregation'])(**cli_args['aggregation_kwargs']),
        x_utility=eval(cli_args['x_utility']),
        x_utility_kwargs=cli_args['x_utility_kwargs'],
        w_utility=eval(cli_args['w_utility']),
        w_utility_kwargs=cli_args['w_utility_kwargs'],
        utility=eval(cli_args['utility']),
        utility_kwargs=cli_args['utility_kwargs'],
        maximization=True
    )

    result = campaign.run_optimization(cli_args=cli_args)

    return result

if __name__ == "__main__":
    cli()
