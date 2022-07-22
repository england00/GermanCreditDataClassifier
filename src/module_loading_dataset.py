import pandas as pd
from sklearn.model_selection import train_test_split


############################################### LOADING DATASET ########################################################

def loading_dataset(path):
    # acquiring data
    names = ['Status of existing checking account', 'Duration in month', 'Credit history', 'Purpose',
             'Credit amount', 'Savings account/bonds', 'Present employment since',
             'Installment rate in percentage of disposable income', 'Personal status and sex',
             'Other debtors / guarantors', 'Present residence since', 'Property', 'Age in years',
             'Other installment plans', 'Housing', 'Number of existing credits at this bank', 'Job',
             'Number of people being liable to provide maintenance for', 'Telephone', 'Foreign worker', 'y']
    df = pd.read_csv(path, delimiter=' ', header=None, names=names)

    # splitting the dataset in training and testing
    df_tr, df_ts = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df['y'])

    return df_tr, df_ts, names
