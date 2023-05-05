import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import *

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = ['davis']
modelings = [GINConvNet, GATNet, GAT_GCN, GCNNet]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 512

result = []
for dataset in datasets:
    processed_data_file_test = 'data/' + dataset + '_test.csv'
    if (not os.path.isfile(processed_data_file_test)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # Create a list of weak models
        models = [model() for model in modelings]

        # Train each weak model and make predictions on the test set
        predictions = []
        i=0
        for model in models:
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model_file_name = 'pretrained/model_' + model.__class__.__name__ + '_' + dataset + '.model'
            if os.path.isfile(model_file_name):
                model.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu')), strict=False)
                y_true, y_pred = predicting(model, device, test_loader)
                predictions.append(y_pred)
                ret = [rmse(y_true, y_pred), mse(y_true, y_pred), pearson(y_true, y_pred),
                spearman(y_true, y_pred), ci(y_true, y_pred)]
                ret = [dataset, modelings[i]] + ret
                result.append(ret)
                i = i + 1
            else:
                print('model', model.__class__.__name__, 'is not available!')

        # Create an AdaBoost regressor with decision tree as base estimator
        ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=50, learning_rate=1.0)

        # Fit the AdaBoost model to the predictions
        ada.fit(np.vstack(predictions).T, y_true)

        # Make prediction on the test set using the AdaBoost model
        y_pred_ada = ada.predict(np.vstack(predictions).T)

        # Compute the evaluation metrics
        ret = [rmse(y_true, y_pred_ada), mse(y_true, y_pred_ada), pearson(y_true, y_pred_ada),
               spearman(y_true, y_pred_ada), ci(y_true, y_pred_ada)]
        ret = [dataset, 'AdaBoostRegressor'] + ret

        # Print the results
        print('Dataset: {}\nModel: {}\nRMSE: {:.4f}\nMSE: {:.4f}\nPearson Correlation: {:.4f}\nSpearman Correlation: {:.4f}\n95% CI: {:.4f}'.format(*ret))

        # Append the results to the result list
        result.append(ret)
        print(result)
        with open('result.csv','w') as f:
            f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
            for ret in result:
                f.write(','.join(map(str,ret)) + '\n')