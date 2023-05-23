from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from tqdm import tqdm
import os

from pmlb import fetch_data, classification_dataset_names,regression_dataset_names
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


if __name__ == "__main__":
    model_name_ls=['LM','SDG','RF','LGBM','XGB','MLP','ANN','HNN','HNN_skip','HNN_adv','MLPres','description']
    mse_dict={}
    r2_dict={}

    n=len(model_name_ls)*len(regression_dataset_names)
    pbar = tqdm(total=n, desc='Back Test Progress', )

    for regression_dataset in regression_dataset_names:
        mse_dict[f'{regression_dataset}']={}
        r2_dict[f'{regression_dataset}'] = {}
        for model_name in model_name_ls:
            existance=True
            try:
                path=f'Results/PMLB/prediction/regression_raw/{regression_dataset}/{model_name}.pkl'
                with open(path, 'rb') as f:
                    if model_name != 'description':
                        test_y,pred_y=pickle.load(f)
                    else:
                        no_instances,no_features=pickle.load(f)
            except:
                existance=False

            if existance:
                if model_name != 'description':
                    mse_dict[f'{regression_dataset}'][f'{model_name}']=mse(test_y,pred_y)
                    r2_dict[f'{regression_dataset}'][f'{model_name}'] = r2_score(test_y, pred_y)
                else:
                    mse_dict[f'{regression_dataset}'][f'{model_name}']=[no_instances,no_features]
                    r2_dict[f'{regression_dataset}'][f'{model_name}'] = [no_instances, no_features]

            pbar.update(1)

    path = f'Results/PMLB/evaluation_matrix/'
    file_path = path + '/' + f'regression_mse.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(mse_dict, f)
    print(mse_dict)

    path = f'Results/PMLB/evaluation_matrix/'
    file_path = path + '/' + f'regression_r2.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(r2_dict, f)
    print(r2_dict)






