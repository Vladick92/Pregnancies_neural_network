import pandas as pd
import torch.nn as nn
import torch
from columns_for_preprocessing import *
from torch.utils.data import TensorDataset,DataLoader
from utils import *

df=pd.read_csv('../datasets/Dataset - Updated.csv')

# Dropping zero values from dataset
df=df.dropna(axis=0)

# Turning High and Low levels into 1 and 0 respectfully
df['Risk Level']=df['Risk Level'].map({"Low":0,'High':1})

# Splitting dataset into train,test,validation datasets
split_data(df)
train_data=pd.read_csv('../datasets/pregnancies_train_data.csv')
test_data=pd.read_csv('../datasets/pregnancies_test_data.csv')
val_data=pd.read_csv('../datasets/pregnancies_val_data.csv')

# Turning dataframe into tensors
x_train_ten=torch.tensor(train_data.drop('Risk Level',axis=1).to_numpy(),dtype=torch.float32)
y_train_ten=torch.tensor(train_data['Risk Level'].to_numpy(),dtype=torch.float32).unsqueeze(dim=1)

x_test_ten=torch.tensor(test_data.drop('Risk Level',axis=1).to_numpy(),dtype=torch.float32)
y_test_ten=torch.tensor(test_data['Risk Level'].to_numpy(),dtype=torch.float32).unsqueeze(dim=1)

x_val_ten=torch.tensor(val_data.drop('Risk Level',axis=1).to_numpy(),dtype=torch.float32)
y_val_ten=torch.tensor(val_data['Risk Level'].to_numpy(),dtype=torch.float32).unsqueeze(dim=1)

train_dataset=TensorDataset(x_train_ten,y_train_ten)
test_dataset=TensorDataset(x_test_ten,y_test_ten)

train_loader=DataLoader(train_dataset,10,True)
test_loader=DataLoader(test_dataset,5,True)

# Using nn.Sequential, for less code instead of nn.Module
torch.manual_seed(42)
model=nn.Sequential(
    nn.Linear(11,8),
    nn.ReLU(),
    nn.Linear(8,1)
)

# Function for training neural network, that returns trained network
trained=train_model(model,train_loader,test_loader)

# Function that prints metrics
evaluate_model(trained,x_val_ten,y_val_ten)

# Saving network
torch.save(trained.state_dict(),'network_weights.pth')