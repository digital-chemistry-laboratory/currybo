from ctypes import ArgumentError
import numpy as np
import os
import pandas as pd
import re
import torch
import json
import re
import sys

from torch import Tensor
from rdkit import Chem
from rdkit import DataStructs
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from typing import List, Dict
from botorch.posteriors import GPyTorchPosterior

class Dataset:
    def __init__(self):
        self.x_options = {}
        self.w_options = {}
        self.smiles = {}
        self.x_data = {}
        self.w_data = {}
        self.y_data = {}
        self.dataset = None
        self.options = None

    def smiles_to_fingerprint(self, smiles: str, radius: int=2, n_bits: int=1024):
        """
        Convert SMILES to fingerprints.

        Args:
            smiles (str): SMILES string to convert.
            radius (int): Radius of the Morgan fingerprint.
            n_bits (int): Length of the Morgan fingerprint.

        Returns:
            torch.tensor: Bit fingerprint as torch tensor.
        """
        if smiles is np.nan:
            return torch.zeros(n_bits, dtype=torch.float64)
        RDLogger.DisableLog('rdApp.*')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(n_bits, dtype=torch.float64)
        #fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = mfpgen.GetFingerprint(mol)
        return torch.tensor(fp, dtype=torch.float64)


    def fingerprint_to_smiles(self, fingerprint: Tensor, name: str, options: List[Tensor]):
        for i, fp in enumerate(options[name]):
            match = True
            for bit_a, bit_b in zip(fp, fingerprint):
                if bit_a != bit_b:
                    match = False
                    break

            if match:
                return self.smiles[name][i]

        raise ValueError("A fingerprint could not be assigned to a SMILES string")


    def generate_output(self, next_x: List[Tensor], next_w: List[Tensor], optimum_x: Tensor, optimum_y: Dict, posterior: GPyTorchPosterior, cli_args):
        output = {
            'estimated_current_optimum': {
                'point': {},
                'value': {}
            },
            'next_points': []
        }

        for o in cli_args['targets']:
            output['estimated_current_optimum']['value'][o['name']] = optimum_y[o['name']].item()

        n_bits = next_w[0].shape[0]
        num_smiles = 0
        for i in cli_args['substrates']:
            if i['type'] == "smiles":
                num_smiles += 1
            elif i['type'] == "array":
                n_bits -= len(self.w_options[i['name']][0])
            else:
                n_bits -= 1
        if num_smiles != 0:
            n_bits //= num_smiles

        if n_bits == 0:
            n_bits = next_x[0].shape[0]
            num_smiles = 0
            for i in cli_args['conditions']:
                if i['type'] == "smiles":
                    num_smiles += 1
                elif i['type'] == "array":
                    n_bits -= len(self.x_options[i['name']][0])
                else:
                    n_bits -= 1
            if num_smiles != 0:
                n_bits //= num_smiles

        for i in range(cli_args['batch_size']):
            point = {}
            obj = {}    

            w_tensor = next_w[i]
            w_idx = 0

            x_tensor = next_x[i]
            x_idx = 0

            for w in cli_args['substrates']:
                if w['type'] == "smiles":
                    smiles = self.fingerprint_to_smiles(w_tensor[w_idx:w_idx+n_bits], w['name'], self.w_options)
                    point[w['name']] = smiles
                    w_idx += n_bits
                elif w['type'] == "array":
                    array_bits = len(self.w_options[w['name']][0])
                    point[w['name']] = " ".join([str(int(j) if j % 1 == 0 else j) for j in w_tensor[w_idx:w_idx+array_bits].tolist()])
                    w_idx += array_bits
                elif w['type'] == "scalar":
                    point[w['name']] = w_tensor[w_idx].item()
                    w_idx += 1

            for x in cli_args['conditions']:
                if x['type'] == "smiles":
                    smiles = self.fingerprint_to_smiles(x_tensor[x_idx:x_idx+n_bits], x['name'], self.x_options)
                    point[x['name']] = smiles

                    smiles_opt = self.fingerprint_to_smiles(optimum_x[0][x_idx:x_idx+n_bits], x['name'], self.x_options)
                    output['estimated_current_optimum']['point'][x['name']] = smiles_opt

                    x_idx += n_bits
                elif x['type'] == "array":
                    array_bits = len(self.w_options[x['name']][0])
                    point[x['name']] = " ".join([str(int(j) if j % 1 == 0 else j) for j in x_tensor[x_idx:x_idx+array_bits].tolist()])
                    point['estimated_current_optimum']['point'][x['name']] = " ".join([str(int(j) if j % 1 == 0 else j) for j in optimum_x[0][x_idx:x_idx+array_bits].tolist()])
                    x_idx += array_bits
                elif x['type'] == "scalar":
                    point[x['name']] = x_tensor[x_idx].item()
                    output['estimated_current_optimum']['point'][x['name']] = optimum_x[0][x_idx].item() 
                    x_idx += 1

            obj['point'] = point

            val = {}
            for j, o in enumerate(cli_args['targets']):
                val[o['name']] = {}
                val[o['name']]['mean'] = posterior.mean[i][j].detach().item()
                val[o['name']]['stdev'] = np.sqrt(posterior.variance[i][j].detach().item())

            obj['value'] = val

            output['next_points'].append(obj)

        return output


    def load_data(self, data_path: str, options_path: str):
        """
        Read csv from dataset.
        """
        self.dataset = pd.read_csv(data_path, index_col = False)
        self.options = pd.read_csv(options_path, index_col = False)
        self.clean_data()


    def clean_data(self):
        """
        Clean the pandas dataframes according to the following rules:
        - Options should be unique
        - Dataset cannot have NaN
        - Get Labels and remove them
        - Store labels somewhere
        """
        if(self.options is None or self.dataset is None):
            return
        
        # store a dict with all labels
        labels = {}
        for col in self.options:
            for val in self.options[col]:
                if val != val:
                    continue
                if ";" in str(val):
                    spl = val.split(";")
                    label = re.sub(r"(^\s|\s$)", "", spl[0])
                    value = re.sub(r"(^\s|\s$)", "", spl[1])
                    if(value in labels and labels[value] is not label):
                        raise ArgumentError(f"The Value `{value}` is used multiple times in your options, and the labels do not match (`{label}` vs. `{labels[value]}`)") 
                    labels[value] = label
        self.labels = labels

        # remove the labels and make each entry unique
        for col in self.options:
            opt = list(set([re.sub(r"^.*; ?", "", str(c)) for c in self.options[col].dropna()]))
            self.options[col] = np.nan
            self.options[col][0:len(opt)] = opt
        self.options.dropna(how = "all", inplace=True)


        # replace all labels by their actual value
        num_rows = len(self.dataset[self.options.columns[0]])
        for col in self.options:
            # check if all columns have the same number of values
            if len(self.dataset[col]) != num_rows:
                raise ArgumentError(f"Your Measurement Dataset does not have a consistent number of rows")
            for i, val in enumerate(self.dataset[col]):
                cleaned = re.sub(r" ?;.*$", "", str(val))
                hits = [value for value in labels if labels[value] == cleaned]
                if len(hits) == 0:
                    continue
                if len(hits) != 1:
                    raise ArgumentError(f"The Label `{cleaned}` is used multiple times in your options, and the values do not match ([{','.join(hits)}])")
                self.dataset[col][i] = hits[0]



    def preprocess_data(self, variable_names: List[Dict[str, str]], targets: List[Dict[str, str]], w_columns: List[Dict[str, str]]):
        """
        Process data from csv into options dictionary.

        Args:
            variable_names (List[Dict[str, str]]): List of names of variable in the optimization domain. Example input: [{ "name": "Catalyst", "type": "smiles" }, { "name": "Temperature", "type": "scalar" }]
            targets (List[Dict[str, str]]): Target variables.
            w_columns (List[Dict[str, str]]): Columns that should be treated as W.
        """
        if not set([v['name'] for v in variable_names]) <= set(self.dataset.columns) or len(variable_names) == 0:
            raise ValueError(f"The input columns (min. 1) you entered are invalid. Possible: {set(self.dataset.columns)}; input: {set([v['name'] for v in variable_names])}")

        if not set([t['name'] for t in targets]) <= set(self.dataset.columns) or len(targets) == 0:
            raise ValueError(f"The target columns (min. 1) you entered are invalid. Possible: {set(self.dataset.columns)}; input: {set([t['name'] for t in targets])}")

        if not set([w['name'] for w in w_columns]) <= set(self.dataset.columns) or len(w_columns) == 0:
            raise ValueError(f"The w columns (min. 1) you entered are invalid. Possible: {set(self.dataset.columns)}; input: {set([w['name'] for w in w_columns])}")

        def parse_column(var):
            if var['type'] == 'smiles':
                unique_smiles = self.options[var['name']].dropna().unique().tolist()
                self.smiles[var['name']] = unique_smiles
                return [self.smiles_to_fingerprint(smiles) for smiles in unique_smiles]
            elif var['type'] == 'scalar':
                values = self.options[var['name']].dropna().sort_values().unique().tolist()
                return torch.tensor([[float(val)] for val in values], dtype=torch.float64)
            elif var['type'] == 'array':
                values = self.options[var['name']].dropna().sort_values().unique().tolist()
                return [torch.Tensor([float(i) for i in re.split(r"\s+", val)]) for val in values]
            else:
                raise ValueError(f"The variable type {var['type']} does not exist. Please choose from {{'smiles', 'scalar', 'array'}}")

        def parse_data(var):
            if var['type'] == 'smiles':
                smiles = self.dataset[var['name']].dropna().tolist()
                return [self.smiles_to_fingerprint(sm) for sm in smiles]
            elif var['type'] == 'scalar':
                values = self.dataset[var['name']].dropna().tolist()
                return torch.tensor([[np.float64(val)] for val in values], dtype=torch.float64)
            elif var['type'] == 'array':
                values = self.dataset[var['name']].dropna().tolist()
                return [torch.Tensor([float(i) for i in re.split(r"\s+", val)]) for val in values]
            else:
                raise ValueError(f"The variable type {var['type']} does not exist. Please choose from {{'smiles', 'scalar', 'array'}}")

        for variable in variable_names:
            self.x_options[variable['name']] = parse_column(variable)
            self.x_data[variable['name']] = parse_data(variable)

        for w in w_columns:
            self.w_options[w['name']] = parse_column(w)
            self.w_data[w['name']] = parse_data(w)
        
        for y in targets:
            self.y_data[y['name']] = parse_data(y)
