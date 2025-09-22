import torch
import pandas as pd
import torch.nn as nn
from sklearn.metrics import recall_score,log_loss,f1_score

# Loading data for making predictions
df=pd.read_csv('../datasets/pregnancies_val_data.csv')

# Dropping null values
df=df.dropna(axis=1)

# Turning dataframe values into tensors
x_values=df.drop(df.columns[-1],axis=1)
y_values=df.iloc[:,-1]
x_ten=torch.from_numpy(x_values.to_numpy()).type(torch.float32)
y_ten=torch.from_numpy(y_values.to_numpy()).type(torch.float32)

# Loading model`s weights
model=nn.Sequential(
    nn.Linear(11,8),
    nn.ReLU(),
    nn.Linear(8,1)
)
model.load_state_dict(torch.load('./network_weights.pth',weights_only=True))

# Getting predictions and metrics
model.eval()
with torch.inference_mode():
    logits=model(x_ten)
    sigms=torch.sigmoid(logits)
    preds=(sigms>=0.5).long()
    print(f'Recall score: {recall_score(y_values,preds.detach().numpy()):.3f}')
    print(f'F1 score: {f1_score(y_values,preds.detach().numpy()):.3f}')
    print(f'Log loss: {log_loss(y_values,sigms.numpy()):.3f}')

# Saving predictions
preds=pd.DataFrame(preds.numpy(),columns=['Risk Level'])
preds=preds['Risk Level'].map({1:'High',0:'Low'})
preds.to_csv('./predictions.csv',index=False)