import os
import pandas as pd
import numpy as np

from scripts.helpers import load_config

if __name__ == "__main__":
    config = load_config(os.path.join("scripts","eval","aggregate_true_loss.json"))
    
    csv_path = os.path.join("outputs","evaluation",config.lossCSVName)
    df = pd.read_csv(csv_path)

    col_dict = {}
    for i in df.columns:
        if i is not 'Unnamed: 0':
            col_dict[i] = {'mean': np.mean(df[i].values), 'std':np.std(df[i].values) }

    for key in col_dict.keys():
        print(key, col_dict[key])