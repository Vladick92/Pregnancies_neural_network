## Business needs
This neural network predicts if pregnant woman have high risk of pregnancy based on health conditions, such as blood pressure, sugar in blood, mental health, and more. Risk level is binary column with High and Low values, respectfully. Developed neural network have two layers of neurons and one ReLU activation function between them. All stages of developing this network displayed in `pregnancies_notebook.ipynb` file.

## Requirements

    python 3.12
    pandas
    sklearn
    pytorch
    numpy

## Running: 

**For training neural network execute:**

    python train.py

Dataset looks good and dont require a lot of preproccesing. This script firstly removes all null values, and applyies categorical encoding to target column. Splits dataset into training, test and validation, then trains neural network. Also dataset isnt huge (approx 1200 rows), so mini-batch gradient descent were used. After training weights of neural network saves with `network_weights.pth` name. If you want train network on your data, just edit line 8:

    df=pd.read_csv('../datasets/{new_dataset.csv}')

**For getting predictions execute:**

    python predict.py

This script loads network, then uses it to make predictions and save it to `predictions.csv` in 'src' folder. There is only one step for preprocessing, its dropping null values. If you want to make predictions with other data edit line 7: 

    df=pd.read_csv('../datasets/{your_new_data.csv}')

**Train and test losses during training**

    Train loss: 0.10 | Test loss: 0.14

**Metrics using validational dataset(pregnancies_val_data.csv):**

    recall score: 0.949
    f1 score: 0.974
    log loss: 0.067
    confusion matrix: 
    [[116   0]
    [  3  56]]


[**Link for dataset**](https://www.kaggle.com/datasets/vmohammedraiyyan/maternal-health-and-high-risk-pregnancy-dataset)