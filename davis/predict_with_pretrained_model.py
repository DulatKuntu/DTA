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
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


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
        
        model_st = modelings[0].__name__
        print('\npredicting for ', dataset, ' using ', model_st)
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modelings[0]().to(device)
        model_file_name = 'pretrained/model_' + model_st + '_' + dataset +  '.model'
        if os.path.isfile(model_file_name):            
            model.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu')),strict =False)
            G,P = predicting(model, device, test_loader)
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
            ret =[dataset, model_st] +  [round(e,3) for e in ret]
            result += [ ret ]
            print('dataset,model,rmse,mse,pearson,spearman,ci')
            print(ret)
        else:
            print('model is not available!')
        model_st = modelings[1].__name__
        print('\npredicting for ', dataset, ' using ', model_st)
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modelings[1]().to(device)
        model_file_name = 'pretrained/model_' + model_st + '_' + dataset +  '.model'
        if os.path.isfile(model_file_name):            
            model.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu')),strict =False)
            G1,P1 = predicting(model, device, test_loader)
            ret1 = [rmse(G1,P1),mse(G1,P1),pearson(G1,P1),spearman(G1,P1),ci(G1,P1)]
            ret1 =[dataset, model_st] +  [round(e,3) for e in ret1]
        else:
            print('model is not available!')
        model_st = modelings[2].__name__
        print('\npredicting for ', dataset, ' using ', model_st)
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modelings[2]().to(device)
        model_file_name = 'pretrained/model_' + model_st + '_' + dataset +  '.model'
        if os.path.isfile(model_file_name):            
            model.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu')),strict =False)
            G2,P2 = predicting(model, device, test_loader)
            ret2 = [rmse(G2,P2),mse(G2,P2),pearson(G2,P2),spearman(G2,P2),ci(G2,P2)]
            ret2 =[dataset, model_st] +  [round(e,3) for e in ret2]
            
        else:
            print('model is not available!')
        model_st = modelings[3].__name__
        print('\npredicting for ', dataset, ' using ', model_st)
        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modelings[3]().to(device)
        model_file_name = 'pretrained/model_' + model_st + '_' + dataset +  '.model'
        if os.path.isfile(model_file_name):            
            model.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu')),strict =False)
            G3,P3 = predicting(model, device, test_loader)
            print('P3', P3)
            ret3 = [rmse(G3,P3),mse(G3,P3),pearson(G3,P3),spearman(G3,P3),ci(G3,P3)]
            ret3 =[dataset, model_st] +  [round(e,3) for e in ret3]
        else:
            print('model is not available!')
        ensemble_predictions = [(p * 0.05 + p1 * 0.4 + p_gcnNet * 0.05 + p_gat_gcn * 0.5) for p, p1, p_gcnNet, p_gat_gcn in zip(P, P1, P2, P3)]
        print('ensemble_predictions', ensemble_predictions)
        # Calculate the average of the ensemble predictions
        average_ensemble = sum(ensemble_predictions) / len(ensemble_predictions)

        print("Average ensemble:", average_ensemble)
        ret4 = [rmse(G,ensemble_predictions),mse(G,ensemble_predictions),0,spearman(G,ensemble_predictions)]
        ret4 =[dataset, 'weighted_ensemble'] +  [round(e,3) for e in ret4]
        weighted_ensemble_predictions = [(p + p1 + p_gcnNet + p_gat_gcn)/4 for p, p1, p_gcnNet, p_gat_gcn in zip(P, P1, P2, P3)]
        # Calculate the average of the ensemble predictions

        ret5 = [rmse(G,weighted_ensemble_predictions),mse(G,weighted_ensemble_predictions),0,spearman(G,weighted_ensemble_predictions)]
        ret5 =[dataset, 'ensemble'] +  [round(e,3) for e in ret5]
        result += [ ret1, ret2, ret3, ret4, ret5 ]
with open('result.csv','w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')
