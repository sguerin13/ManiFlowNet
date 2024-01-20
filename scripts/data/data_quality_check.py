from pyntcloud.io import ply
import torch
import sys
import os
import numpy as np
import pickle
import plotly.express as px
import pandas as pd
import data.wrangling.data_utils as DU
import plotly.graph_objects as go
import tqdm
import shutil
from scripts.helpers import load_config

if __name__ == '__main__':
    
    config = load_config(os.path.join("scripts","sim","data_quality_check.json"))
    train_path = config.trainPath
    quarantine_folders = config.quarantinePath

    sim_types = os.listdir(train_path)
    sim_folders = [os.path.join(train_path,i) for i in sim_types]

    sim_path_list = []
    for folder in sim_folders:
        simulations = os.listdir(folder)
        for simulation in simulations:
            sim_path = os.path.join(folder,simulation)
            sim_path_list.append(sim_path)


    ######################## check for negative pressures ################################
    bad_pressure_counter = 0
    quarantine_list = []
    for sim_path in tqdm.tqdm(sim_path_list):
        try:
            ply_path = os.path.join(sim_path,'pnt_cld.ply')
            # check the pressure values
            pnt_cld = DU.read_ply(ply_path)
            pressures = pnt_cld['pressure']
            if any(pressures < 0):
                bad_pressure_counter += 1
                quarantine_list.append(sim_path)
        except Exception as e:
            print(e)
            sim_type = sim_path.split('/')[-2]
            sim_no = sim_path.split('/')[-1]
            quarantine_path = os.path.join(quarantine_folders,'missing_data',sim_type)
            quarantine_dest = os.path.join(quarantine_path,sim_no)
            if not os.path.exists(quarantine_path):
                os.mkdir(quarantine_path)
            dest = shutil.move(sim_path, quarantine_dest) 

            
    print('ratio of corrupt simulations: ', bad_pressure_counter/len(sim_path_list))

    # move the files to quarantine
    for sim in quarantine_list:
        sim_type = sim.split('/')[-2]
        sim_no = sim.split('/')[-1]
        quarantine_path = os.path.join(quarantine_folders,'negative_pressure',sim_type)
        quarantine_dest = os.path.join(quarantine_path,sim_no)
        if not os.path.exists(quarantine_path):
            os.mkdir(quarantine_path)

        dest = shutil.move(sim, quarantine_dest) 
