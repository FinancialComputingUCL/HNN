from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from tqdm import tqdm
import os
from os.path import exists

from pmlb import fetch_data, classification_dataset_names,regression_dataset_names


def save_description(dataset_name,dataset):
    path = f'Results/PMLB/prediction/regression_raw/{dataset_name}'
    file_path = path + '/' + f'description.pkl'

    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        pass

    if not os.path.exists(file_path):

        with open(file_path, 'wb') as f:
            pickle.dump([len(ds),len(ds.columns)-1], f)






if __name__ == "__main__":
    #1191_BNG_pbc

    n=len(regression_dataset_names)
    pbar = tqdm(total=n, desc='Back Test Progress')

    for regression_dataset in regression_dataset_names:
        ds = fetch_data(regression_dataset)
        save_description(regression_dataset, ds)
        pbar.update(1)





