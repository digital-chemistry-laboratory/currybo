import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings 
import glob

from genbo.campaign import GeneralBOCampaign

warnings.filterwarnings('ignore') 

def nanstd(x):
    return torch.sqrt(torch.nanmean(torch.pow(x - torch.nanmean(x, dim=0), 2), dim = 0))

def create_plot(datasets, args, gap=True):
    fig, ax = plt.subplots()

    for q, dataset in zip(args['batch_sizes'], datasets):
        generality_train_set = [campaign.generality_train_set for campaign in dataset]
        # MAX_BUDGET = q * generality_train_set[0].shape[0]
        # generality_train_set = [tensor[:(MAX_BUDGET+1), :] for tensor in generality_train_set]
        generality_stacked = torch.stack(generality_train_set)
        if gap:
            global_optimum_set = [campaign.global_optimum for campaign in dataset]
            global_optimum_stacked = torch.cat(global_optimum_set)
            generality_gap = generality_stacked.squeeze()
            diff = global_optimum_stacked - generality_gap[:, 0].unsqueeze(1)

            if torch.isclose(diff, torch.zeros_like(diff), atol=0.01).any():
                zeros = torch.isclose(diff, torch.zeros_like(diff), atol=0.01).all(dim=1)
                generality_gap = generality_gap[~zeros]
                global_optimum_stacked = global_optimum_stacked[~zeros]

            gap_value = (generality_gap - generality_gap[:, 0].unsqueeze(1)) / (global_optimum_stacked - generality_gap[:, 0].unsqueeze(1))
            gap_value = gap_value.unsqueeze(-1)
            mean_gap = torch.mean(gap_value, dim=0)
            std_error_gap = torch.std(gap_value, dim=0) / np.sqrt(gap_value.size(0))
            y_mean = mean_gap.view(-1).numpy()
            y_std = std_error_gap.view(-1).numpy()

            ax.plot(np.arange(y_mean.shape[0]) * q, y_mean, label=dataset[0].label)
            ax.fill_between(np.arange(y_mean.shape[0]) * q, (y_mean - y_std), (y_mean + y_std), alpha=0.3)
            ax.set_ylabel(r'GAP ($\uparrow$)', size=10)
        else:
            mean_tensor = torch.mean(generality_stacked, dim=0)
            std_error_tensor = torch.std(generality_stacked, dim=0) / np.sqrt(generality_stacked.size(0))
            y_mean = mean_tensor.view(-1).numpy()
            y_std = std_error_tensor.view(-1).numpy()

            ax.plot(np.arange(y_mean.shape[0]) * q, y_mean, label=dataset[0].label)
            ax.fill_between(np.arange(y_mean.shape[0]) * q, (y_mean - y_std), (y_mean + y_std), alpha=0.3)
            ax.set_ylabel(args['ylabel'], fontsize=16)

    ax.set_xlim(0, datasets[0][0].generality_train_set.shape[0] * args['batch_sizes'][0])

    ax.set_xlabel(args['xlabel'], fontsize=16)

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args['path'], args['output']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Key to define analytical problem set.")
    parser.add_argument('--path', type=str, default=".", help="base path for all datasets")
    parser.add_argument('--datasets', nargs="+", type=str, help="GLOB strings of datasets to use. Each Data set is one line in the plot")
    parser.add_argument('--labels', nargs="+", type=str, help="Labels for the datasets")
    parser.add_argument('--xlabel', type=str, default="Observations", help="xlabel")
    parser.add_argument('--ylabel', type=str, default=r"$\phi(f(\hat{\mathbf{x}}; \mathbf{w}), \mathcal{W})$ ($\uparrow$)", help="ylabel")
    parser.add_argument('--output', type=str, default="plot.pdf", help="plot file name")
    parser.add_argument('--batch-sizes', type=int, nargs="+", help="batch sizes for datasets")
    parser.add_argument('--not-gap', default=False, action="store_true", help="do not use GAP")
    args = vars(parser.parse_args())

    def get_files(globstr, label):
        files = glob.glob(os.path.join(args['path'], globstr))
        return files, label

    files = list(map(get_files, args['datasets'], args['labels']))

    def get_dataset(file, label):
        campaign = GeneralBOCampaign()
        campaign_load = torch.load(file)
        campaign.observations_x = campaign_load['observations_x']
        campaign.observations_y = campaign_load['observations_y']
        campaign.observations_w = campaign_load['observations_w']
        campaign.observations_w_idx = campaign_load['observations_w_idx']
        campaign.optimum_x = campaign_load['optimum_x']
        campaign.global_optimum = campaign_load['global_optimum']
        campaign.generality_train_set = campaign_load['generality train set']
        campaign.generality_test_set = campaign_load['generality test set']
        campaign._problem_type = 'discrete'
        campaign.label = label
        campaign.ylabel = args['ylabel']
        return campaign

    def get_datasets(f, label):
        return list(map(get_dataset, f, [label for i in range(len(f))]))

    datasets = [get_datasets(i, j) for i, j in files]


    create_plot(datasets, args)
    #for j, dataset in enumerate(["Denmark"]):
        #create_plot(first_element = campaign_dicts[j], file_name =f"../runs/chemistry_methods_{dataset}{'_enhance' if args.enhanced else ''}.pdf", gap = True, MAX_BUDGET = 10)
