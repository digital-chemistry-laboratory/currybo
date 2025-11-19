![CurryBO-logo](https://raw.githubusercontent.com/digital-chemistry-laboratory/currybo/refs/heads/main/CurryBO-logo.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](./LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)

[![arXiv](https://img.shields.io/badge/arXiv-2502.18966-b31b1b.svg)](https://arxiv.org/abs/2502.18966)



# CurryBO: Bayesian optimization over curried function spaces

`CurryBO` is a pure python package that allows to conduct Bayesian Optimization in the search for general (i.e. transferable) parameters that *work well* across multiple related tasks.

## Installation

To install the package, simply:
```
pip install currybo
```

## Usage

If you are interested in using CurryBO to predict your experiments based on your own measurements, you can simply call
```
currybo \
--measurements [your-data-file.csv] --options [your-options-file.csv] \
--substrates "name=[Your-substrate-class-name],type=smiles" \
--conditions "name=[Your-condition-class-name],type=smiles" \
--targets "name=[Your-target-name],type=scalar" \
--objectives "name=[Your-target-name],abs_threshold=[Your-target-threshold],maximize=True" \
```

If you do not want to code, please visit the [CurryBO Website](https://currybo.ethz.ch), where we built a web-based application so that everyone can optimize for general reaction conditions.

If you are interested in using CurryBO to predict your next experiments, please refer to the [currybo-benchmarks repository](https://github.com/digital-chemistry-laboratory/currybo-benchmarks.git), which contains datasets to reproduce the experiments in this work.

## Citation

If you use CurryBO in your research, please cite the corresponding paper:

```
@misc{schmid_one_2025,
	title = {One {Set} to {Rule} {Them} {All}: {How} to {Obtain} {General} {Chemical} {Conditions} via {Bayesian} {Optimization} over {Curried} {Functions}},
	url = {http://arxiv.org/abs/2502.18966},
	doi = {10.48550/arXiv.2502.18966},
	publisher = {arXiv},
	author = {Schmid, Stefan P. and Rajaonson, Ella Miray and Ser, Cher Tian and Haddadnia, Mohammad and Leong, Shi Xuan and Aspuru-Guzik, Alán and Kristiadi, Agustinus and Jorner, Kjell and Strieth-Kalthoff, Felix},
	month = feb,
	year = {2025},
	note = {arXiv:2502.18966 [cs]},
}
```
