# -*- coding: utf-8 -*-
"""

"""
import warnings
warnings.filterwarnings("ignore")
import os
from datetime import datetime
#from filelock import FileLock
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import xlrd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
    
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, hidden_dim3=16, output_dim=1):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim1)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim1, hidden_dim2)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.layer_3 = nn.Linear(hidden_dim2, hidden_dim3)
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="relu")
        self.layer_4 = nn.Linear(hidden_dim3, output_dim)
       
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.linear(self.layer_4(x))
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

def load_data(df):
    #Get input data from df
    dfx = df.drop(df.columns[n], axis=1)
    #normalize input data
    dfx = (dfx-dfx.min())/(dfx.max()-dfx.min())
    
    #get output data from df
    dfy = df.iloc[:,n]
    #normalize output data
    dfy_min=dfy.min()
    dfy_max=dfy.max()
    #dfy = (dfy-dfy.min())/(dfy.max()-dfy.min())
    #create torches from dfs
    X_data = torch.from_numpy(dfx.values).float()#.to(device)
    y_data = torch.from_numpy(dfy.values).float()

    #split data into test and training datasets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.2, random_state=22)
    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)
    return trainset, testset, dfy_min, dfy_max

def write_output_data(df, filepath=r'C:\data_local\ML\Prognoseersatzwerte.xlsx', sheetname='Test'):
    df.to_excel(filepath,sheetname)
    
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
    
    
def train_nn(config):
    net = NeuralNetwork(config["hidden_dim1"], config["hidden_dim2"], config["hidden_dim3"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    #data_dir = os.path.abspath("./data")
    trainset, testset = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"loss": (val_loss / val_steps), "accuracy": correct / total}, checkpoint=checkpoint)
    print("Finished Training")
    

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
        prediction = calculate_prediction_via_nn(df, n)        
        #add to output df
        df_output[colname] = prediction
    
    plot_ersatzwerte(df_output)
    #write_output_data(df_output)
        
    print("Training Complete")