from sklearn.model_selection import train_test_split
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.metrics import recall_score,log_loss,f1_score,confusion_matrix

def split_data(dataframe):
    df=dataframe.copy()
    x_train,x_temp,y_train,y_temp=train_test_split(df.drop('Risk Level',axis=1),df['Risk Level'],random_state=42,train_size=0.7)
    x_test,x_val,y_test,y_val=train_test_split(x_temp,y_temp,test_size=0.5,random_state=42)
    train_data=pd.concat([x_train,y_train],axis=1)
    train_data.to_csv('../datasets/pregnancies_train_data.csv',index=False)
    test_data=pd.concat([x_test,y_test],axis=1)
    test_data.to_csv('../datasets/pregnancies_test_data.csv',index=False)
    val_data=pd.concat([x_val,y_val],axis=1)
    val_data.to_csv('../datasets/pregnancies_val_data.csv',index=False)

# Using mini-batch gradient descent, because there is small number of objects in dataframe
def train_model(model_p,train_loader,test_loader):
    train_losses=[]
    test_losses=[]
    model=model_p
    loss_func=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(50):

        model.train()
        epoch_train_loss=0
        for x_train,y_train in train_loader:
            train_preds=model(x_train)
            train_loss=loss_func(train_preds,y_train)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epoch_train_loss+=train_loss.item()
        epoch_train_loss/=len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_test_loss=0
        for x_test,y_test in test_loader:
            with torch.inference_mode():
                test_preds=model(x_test)
                test_loss=loss_func(test_preds,y_test)
                epoch_test_loss+=test_loss.item()
        epoch_test_loss/=len(test_loader)
        test_losses.append(epoch_test_loss)
    print(f'Train loss: {train_losses[-1]:.2f} | Test loss: {test_losses[-1]:.2f}')
    return model

def evaluate_model(model,x_val,y_val):
    with torch.inference_mode():
        logits=model(x_val)
        sigms=torch.sigmoid(logits)
        preds=(sigms>=0.5).long()    
    print('Metrics on validation dataset')
    print(f'Recall score: {recall_score(y_val.numpy(),preds.detach().numpy()):.3f}')
    print(f'F1 score: {f1_score(y_val.numpy(),preds.detach().numpy()):.3f}')
    print(f'Log loss: {log_loss(y_val.numpy(),sigms.numpy()):.3f}')
    print(f'Confusion matrix: {'\n'}{confusion_matrix(y_val.numpy(),preds.detach().numpy())}')
