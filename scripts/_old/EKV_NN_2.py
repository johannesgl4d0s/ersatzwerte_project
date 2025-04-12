# -*- coding: utf-8 -*-
"""

"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset #Dataset
import xlrd
from datetime import datetime
import matplotlib.pyplot as plt
#from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
#import seaborn as sns

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim1)
        #nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        nn.init.uniform_(self.layer_1.weight, a=0.0, b=1.0)
        self.layer_2 = nn.Linear(hidden_dim1, hidden_dim2)
        #nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        nn.init.uniform_(self.layer_2.weight, a=0.0, b=1.0)
        self.layer_3 = nn.Linear(hidden_dim2, hidden_dim3)
        #nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")
        nn.init.uniform_(self.layer_3.weight, a=0.0, b=1.0)
        self.layer_4 = nn.Linear(hidden_dim3, output_dim)
        #nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")
        nn.init.uniform_(self.layer_4.weight, a=0.0, b=1.0)
       
    def forward(self, x):
        x = torch.nn.functional.tanh(self.layer_1(x))
        x = torch.nn.functional.tanh(self.layer_2(x))
        x = torch.nn.functional.tanh(self.layer_3(x))
        x = torch.nn.functional.tanh(self.layer_4(x))
        return x
    
class NeuralNetwork2(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(NeuralNetwork2, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim1)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(hidden_dim1, hidden_dim2)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(hidden_dim2, hidden_dim3)
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")
        self.layer_4 = nn.Linear(hidden_dim3, output_dim)
        nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")
       
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.layer_1(x))
        x = torch.nn.functional.leaky_relu(self.layer_2(x))
        x = torch.nn.functional.leaky_relu(self.layer_3(x))
        x = torch.nn.functional.leaky_relu(self.layer_4(x))
        return x
    
class NeuralNetwork3(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(NeuralNetwork3, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim1)
        nn.init.uniform_(self.layer_1.weight, a=0.0, b=1.0)
        self.layer_2 = nn.Linear(hidden_dim1, hidden_dim2)
        nn.init.uniform_(self.layer_2.weight, a=0.0, b=1.0)
        self.layer_3 = nn.Linear(hidden_dim2, hidden_dim3)
        nn.init.uniform_(self.layer_3.weight, a=0.0, b=1.0)
        self.layer_4 = nn.Linear(hidden_dim3, output_dim)
        nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")
       
    def forward(self, x):
        x = torch.nn.functional.sigmoid(self.layer_1(x))
        x = torch.nn.functional.tanh(self.layer_2(x))
        x = torch.nn.functional.tanh(self.layer_3(x))
        x = torch.nn.functional.leaky_relu(self.layer_4(x))
        return x
    
def reset_model(model):
    for layer in model.children():
       if hasattr(layer, 'reset_parameters'):
           layer.reset_parameters()

# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

# Old CSV Import
def read_data_csv():
    df = pd.read_csv("C:\data_local\ML\Daten_NN.csv", header=0, delimiter=";",decimal=",")
    df = df.drop(df.columns[0], axis=1)
    return df

# Reads in the 
def read_model_data(filepath=r'C:\data_local\ML\Modelldaten.xlsb', sheetname='Modelldaten_v2'):
    df = pd.read_excel(filepath,sheetname, usecols='A:AX')
    df['Timestamp_left']=df.apply(lambda row : datetime(*xlrd.xldate_as_tuple(row['Timestamp_left'], 0)),axis=1)
    df['Timestamp_right']=df.apply(lambda row : datetime(*xlrd.xldate_as_tuple(row['Timestamp_right'], 0)),axis=1)
    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    df=df.set_index(['Timestamp_left','Timestamp_right'])    
    return df
    
def read_input_data(filepath=r'C:\data_local\ML\Prognoseersatzwerte.xlsb', sheetname='Inputdaten'):
    df = pd.read_excel(filepath,sheetname, usecols='A:E', dtype='float64', skiprows=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16])
    df.columns.values[0] = 'Timestamp_left'
    df.columns.values[1] = 'Timestamp_right'
    df['Timestamp_left']=df.apply(lambda row : datetime(*xlrd.xldate_as_tuple(row['Timestamp_left'], 0)),axis=1)
    df['Timestamp_right']=df.apply(lambda row : datetime(*xlrd.xldate_as_tuple(row['Timestamp_right'], 0)),axis=1)
    df=df.set_index(['Timestamp_left','Timestamp_right'])
    return df

def write_output_data(df, filepath=r'C:\data_local\ML\Prognose.xlsx', sheetname='Test'):
    df.to_excel(filepath,sheetname)

def print_shape_dataloader(DataLoader: torch.utils.data.DataLoader):
    for batch, (X, y) in enumerate(DataLoader):
        print(f"Batch: {batch+1}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        break
    
def get_prediction(model, input_data, dfy_min, dfy_max):
    #get prediction
    prediction=model(input_data)
    #print(prediction)
    prediction=(prediction*(dfy_max-dfy_min))+dfy_min
    return prediction
    
def plot_results(prediction_torch, df_actual: pd.DataFrame, df_column: str):
    df_plot = df_actual.copy()
    df_plot["Prognose"]=prediction_torch.cpu().detach().numpy()
    df_plot = df_plot.droplevel(1)
    plt.plot(df_plot.index.to_pydatetime(), df_plot[df_column], linewidth=1, color='blue', label="Daten")
    plt.plot(df_plot.index.to_pydatetime(), df_plot["Prognose"], linewidth=0.3, color='red', label="Prognose")    
    plt.legend()
    plt.show()    
    
def plot_stepwise_loss(num_epochs, length_step, loss_values):
    step = np.linspace(0, num_epochs, length_step)    
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(step, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    
def plot_ersatzwerte(df):
    for column in df:
        #print(column)
        df_plot = df[column]
        #print(df_plot)
        df_plot = df_plot.droplevel(1)
        #print(df_plot)
        plt.plot(df_plot.index.to_pydatetime(), df_plot, label="Ersatzwert")
        plt.legend()
        plt.show()
    
def calculate_prediction_via_nn(df, n):
    #Get input data from df
    dfx = df.drop(df.columns[n], axis=1)
    #normalize input data
    dfx = (dfx-dfx.min())/(dfx.max()-dfx.min())
    
    #get output data from df
    dfy = df.iloc[:,n]
    #normalize output data
    dfy_min=dfy.min()
    dfy_max=dfy.max()
    dfy = (dfy-dfy.min())/(dfy.max()-dfy.min())
    #create torches from dfs
    X_data = torch.from_numpy(dfx.values).float()#.to(device)
    y_data = torch.from_numpy(dfy.values).float()

    #split data into test and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.8, random_state=22)
    train_data = TensorDataset(X_train, y_train)
    #test_data = TensorDataset(X_test, y_test)
    
    #create batches
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=24)
    #test_dataloader = DataLoader(test_data, batch_size=len(test_data.tensors[0]))
    
    #print shape of training Dataset
    print_shape_dataloader(train_dataloader)
    
    #set parameters for nn
    input_dim = n
    hidden_dim1 = 16*n
    hidden_dim2 = round(hidden_dim1/2)
    hidden_dim3 = round(hidden_dim1/4)
    output_dim = 1
           
    #model = NeuralNetwork3(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
    model = NeuralNetwork2(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
    reset_model(model)
    print(model)

    #learning_rate = 0.01

    loss_fn = nn.MSELoss()

    #optimizer = optim.SGD(model.parameters(), lr=0.1)
    #optimizer = optim.Adadelta(model.parameters())#, lr=learning_rate)
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters())#, lr=learning_rate)

    num_epochs = 200
    loss_values = []        

    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()           
            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()
        if (epoch % 10) == 0:
            prediction = get_prediction(model, X_data, dfy_min, dfy_max)
            plot_results(prediction, pd.DataFrame(df.iloc[:,n], columns=[colname]), colname)
            #plot_results_v2(prediction, pd.DataFrame(df.iloc[:,n], columns=[colname]), colname)
            print(loss)
    
    plot_stepwise_loss(num_epochs, len(train_dataloader)*num_epochs, loss_values)
            
    prediction = get_prediction(model, X_data_p, dfy_min, dfy_max)
    prediction = prediction.cpu().detach().numpy()
    return prediction, model
    

if __name__ == "__main__":
    # Set Device
    device = get_device()

    ### Load the data
    df_input = read_input_data()
    df_parameters = read_model_data()
    df_output = df_parameters.loc[:,[]]
    
    #get prediction X_data
    dfx_p = df_parameters
    dfx_p = (dfx_p-dfx_p.min())/(dfx_p.max()-dfx_p.min())
    X_data_p = torch.from_numpy(dfx_p.values).float()
    
    for column in df_input:
        # Create Single timeseries input for model
        #df_temp = df_input['EXXETA.Plattform.Verbrauch.EKV.Ost'].copy()
        df = df_parameters.join(df_input[column])
        df = df.dropna()
        # And make a convenient variable to remember the number of input columns
        n = len(df.columns)-1
        colname = df.columns[n]
        #calculate prediction
        prediction, model = calculate_prediction_via_nn(df, n)        
        #add to output df
        df_output[colname] = prediction
    
    paramweights = list()
    
    for param in model.parameters():
        paramweights.append(param.data.cpu().detach().numpy())
      
    df_mininput = pd.DataFrame(paramweights[0].min(axis=0))
    df_maxinput = pd.DataFrame(paramweights[0].max(axis=0))
    
    plot_ersatzwerte(df_output)
    #write_output_data(df_output)
        
    print("Training Complete")