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

import warnings
warnings.filterwarnings('ignore')


def train_model(train_X, test_X, train_y, test_y, dataset_name, model_name, model, params):
    path = f'Results/PMLB/prediction/regression_raw/{dataset_name}'
    file_path = path + '/' + f'{model_name}.pkl'

    HNN_path=path + '/' + f'HNN.pkl'
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        pass

    if (not os.path.exists(file_path)) : # and (os.path.exists(HNN_path)):
        print(file_path)
        CV = GridSearchCV(model, params)
        CV.fit(train_X, train_y)

        pred_y = CV.best_estimator_.predict(test_X)

        with open(file_path, 'wb') as f:
            pickle.dump([test_y, pred_y], f)


def model_LM():
    from sklearn.linear_model import LinearRegression
    hyper_params = {'fit_intercept': (True,),}
    est=LinearRegression()
    return est, hyper_params

def model_SDG():
    from sklearn import linear_model
    hyper_params = [
        {
            'alpha': (1e-04,1e-03,0.01,0.1,1,10),
            'penalty': ('l2','l1','elasticnet',),
        },
    ]

    est=linear_model.SGDRegressor()
    return est, hyper_params

def model_RF():
    from sklearn import ensemble

    hyper_params = [{
        'n_estimators': (10,50, 100, 500),
        'min_weight_fraction_leaf': (0.0, 0.25, 0.5),
        'max_features': ('sqrt', 'log2', None),
    }]

    est = ensemble.RandomForestRegressor()
    return est, hyper_params

def model_LGBM():
    import lightgbm

    hyper_params = {
        'n_estimators': (10, 50, 100, 250),
        'learning_rate': (0.0001, 0.01, 0.05, 0.1, 0.2,0.5),
        'subsample': (0.75, 1,),
        'boosting_type': ('gbdt', 'dart', 'goss')
    }

    est = lightgbm.LGBMRegressor(
        max_depth=6,
        deterministic=True,
        force_row_wise=True
    )
    return est, hyper_params

def model_XGB():
    import xgboost
    hyper_params = [
        {
            'n_estimators': (10, 50, 100, 250),
            'learning_rate': (0.0001, 0.01, 0.05, 0.1, 0.2,0.5),
            'gamma': (0, 0.1, 0.2, 0.3, 0.4,),
            'subsample': (0.75, 1,),
        },
    ]

    est = xgboost.XGBRegressor(max_depth=6)
    return est, hyper_params

def model_ANN():
    from sklearn.neural_network import MLPRegressor

    hyper_params = [
        {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (30,100,50),
                                   (30,),(50,),(100,),
                                   (30,50),(50,100),(30,100),
                                   (30,50,100,50),(10,30,50,30)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001,0.01,0.1],
            'learning_rate': ['constant','adaptive'],
        },
    ]

    est = MLPRegressor(random_state=1,max_iter=500,early_stopping=True)
    return est, hyper_params



if __name__ == "__main__":
    #1191_BNG_pbc

    #model_dict={'LM':model_LM(),'SDG':model_SDG(),'RF':model_RF(),'LGBM':model_LGBM(),'XGB':model_XGB(),}
    model_dict={'LM':model_LM()}
    n=len(model_dict)*len(regression_dataset_names)
    pbar = tqdm(total=n, desc='Back Test Progress', )

    for regression_dataset in regression_dataset_names:
        X, y = fetch_data(regression_dataset, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y)
        for model_name,func in model_dict.items():
            model,params=func
            train_model(train_X, test_X, train_y, test_y, regression_dataset,model_name, model, params)

        pbar.update(1)





