import os
from data.wrangling.data_utils import split_dataset
from scripts.helpers import load_config

'''

This is meant to be ran once when the dataset is first generated,
it will create the train/val split for you and keep the same directory structure

'''

if __name__ == '__main__':
    config = load_config(os.path.join("scripts","data","split_dataset.json"))    
    
    fpath = config.dataRootDir
    split = config.splitPercentage
    split_dataset(split,fpath)


