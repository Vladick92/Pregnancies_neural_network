import torch
import pandas as pd
from utils import *
from model_layers import *

# Loading data for making predictions
df=pd.read_csv('../datasets/pregnancies_val_data.csv')

# Dropping null values
df=df.dropna(axis=1)

# Turning dataframe values into tensors
if 'Risk Level' in df.columns:
    x_ten=torch.from_numpy(df.drop('Risk Level',axis=1).to_numpy()).type(torch.float32)
    y_ten=torch.from_numpy(df['Risk Level'].to_numpy()).type(torch.float32).unsqueeze(1)
else:
    x_ten=torch.from_numpy(df.to_numpy()).type(torch.float32)

# Loading model`s weights
model=make_model()
model.load_state_dict(torch.load('./network_weights.pth',weights_only=True))

# Getting predictions
if 'Risk Level' in df.columns:
    preds=predict(model,x_ten,y_ten)
else:
    preds=predict(model,x_ten)

# Saving predictions
preds=pd.DataFrame(preds.numpy(),columns=['Risk Level'])
preds=preds['Risk Level'].map({1:'High',0:'Low'})
preds.to_csv('./predictions.csv',index=False)