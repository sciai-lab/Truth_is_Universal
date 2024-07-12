import torch as t
import pandas as pd
import numpy as np
import os
from glob import glob
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25

def collect_acts(dataset_name, model_family, model_size,
                  model_type, layer, center=True, scale=False, device='cpu'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = os.path.join(ROOT, 'acts', model_family, model_size, model_type, dataset_name)
    activation_files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
    acts = [t.load(os.path.join(directory, f'layer_{layer}_{i}.pt'), map_location=device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)] 
    try:
        acts = t.cat(acts, dim=0).to(device)
    except:
        raise Exception("No activation vectors could be found for the dataset " 
                        + dataset_name + ". Please generate them first using generate_acts.")
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts

def cat_data(d):
    """
    Given a dict of datasets (possible recursively nested), returns the concatenated activations and labels.
    """
    all_acts, all_labels = [], []
    for dataset in d:
        if isinstance(d[dataset], dict):
            if len(d[dataset]) != 0: # disregard empty dicts
                acts, labels = cat_data(d[dataset])
                all_acts.append(acts), all_labels.append(labels)
        else:
            acts, labels = d[dataset]
            all_acts.append(acts), all_labels.append(labels)
    try:
        acts, labels = t.cat(all_acts, dim=0), t.cat(all_labels, dim=0)
    except:
        raise Exception("No activation vectors could be found for this dataset. Please generate them first using generate_acts.")
    return acts, labels

class DataManager:
    """
    Class for storing activations and labels from datasets of statements.
    """
    def __init__(self):
        self.data = {
            'train' : {},
            'val' : {}
        } # dictionary of datasets
        self.proj = None # projection matrix for dimensionality reduction
    
    def add_dataset(self, dataset_name, model_family, model_size, model_type, layer,
                     label='label', split=None, seed=None, center=True, scale=False, device='cpu'):
        """
        Add a dataset to the DataManager.
        label : which column of the csv file to use as the labels.
        If split is not None, gives the train/val split proportion. Uses seed for reproducibility.
        """
        acts = collect_acts(dataset_name, model_family, model_size, model_type,
                             layer, center=center, scale=scale, device=device)
        df = pd.read_csv(os.path.join(ROOT, 'datasets', f'{dataset_name}.csv'))
        labels = t.Tensor(df[label].values).to(device)

        if split is None:
            self.data[dataset_name] = acts, labels

        if split is not None:
            assert 0 <= split and split <= 1
            if seed is None:
                seed = random.randint(0, 1000)
            t.manual_seed(seed)
            train = t.randperm(len(df)) < int(split * len(df))
            val = ~train
            self.data['train'][dataset_name] = acts[train], labels[train]
            self.data['val'][dataset_name] = acts[val], labels[val]

    def get(self, datasets):
        """
        Output the concatenated activations and labels for the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        If proj, projects the activations using the projection matrix.
        """
        if datasets == 'all':
            data_dict = self.data
        elif datasets == 'train':
            data_dict = self.data['train']
        elif datasets == 'val':
            data_dict = self.data['val']
        elif isinstance(datasets, list):
            data_dict = {}
            for dataset in datasets:
                if dataset[-6:] == ".train":
                    data_dict[dataset] = self.data['train'][dataset[:-6]]
                elif dataset[-4:] == ".val":
                    data_dict[dataset] = self.data['val'][dataset[:-4]]
                else:
                    data_dict[dataset] = self.data[dataset]
        elif isinstance(datasets, str):
            data_dict = {datasets : self.data[datasets]}
        else:
            raise ValueError(f"datasets must be 'all', 'train', 'val', a list of dataset names, or a single dataset name, not {datasets}")
        acts, labels = cat_data(data_dict)
        # if proj and self.proj is not None:
        #     acts = t.mm(acts, self.proj)
        return acts, labels
    
def dataset_sizes(datasets):
    """
    Computes the size of each dataset, i.e. the number of statements.
    Input: array of strings that are the names of the datasets
    Output: dictionary, keys are the dataset names and values the number of statements
    """
    dataset_sizes_dict = {}
    for dataset in datasets:
        file_path = 'datasets/' + dataset + '.csv'
        with open(file_path, 'r') as file:
            line_count = sum(1 for line in file)
        dataset_sizes_dict[dataset] = line_count - 1
    return dataset_sizes_dict

def collect_training_data(datasets, train_set_sizes, model_family, model_size, model_type, layer, center=False, device='cpu'):
    all_acts, all_labels, all_polarities = [], [], []
    
    for dataset in datasets:
        dm = DataManager()
        split = min(train_set_sizes.values()) / train_set_sizes[dataset]
        dm.add_dataset(dataset, model_family, model_size, model_type, layer, split=split, center=center, device=device)
        acts, labels = dm.get('train')
        
        polarity = -1.0 if 'neg_' in dataset else 1.0
        polarities = t.full((labels.shape[0],), polarity).to(device)
        
        all_acts.append(acts)
        all_labels.append(labels)
        all_polarities.append(polarities)
    
    return map(t.cat, (all_acts, all_labels, all_polarities))

def compute_statistics(results):
    stats = {}
    for key in results:
        means = {dataset: np.mean(values) for dataset, values in results[key].items()}
        stds = {dataset: np.std(values) for dataset, values in results[key].items()}
        stats[key] = {'mean': means, 'std': stds}
    return stats
    
def learn_truth_directions(acts, labels, polarities):
    # Check if all polarities are zero (handling both int and float) -> if yes learn only t_g
    all_polarities_zero = t.allclose(polarities, t.tensor([0.0]), atol=1e-8)
    
    if all_polarities_zero:
        X = labels.reshape(-1, 1)
    else:
        X = t.column_stack([labels, labels * polarities])

    # Compute the analytical OLS solution
    solution = t.linalg.inv(X.T @ X) @ X.T @ acts

    # Extract t_g and t_p
    if all_polarities_zero:
        t_g = solution.flatten()
        t_p = None
    else:
        t_g = solution[0, :]
        t_p = solution[1, :]

    return t_g, t_p

     




