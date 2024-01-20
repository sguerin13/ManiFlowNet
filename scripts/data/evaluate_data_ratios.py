import json
from types import SimpleNamespace
import data.wrangling.data_utils as DU
import numpy as np
import matplotlib.pyplot as plt
import os
from scripts.helpers import load_config

if __name__ == '__main__':
    config = load_config(os.path.join("scripts","data","evaluate_data_ratios.json"))

    normScalerPath = os.path.join("outputs",config.normScalerFileName)
    ndimScalerPath = os.path.join("outputs",config.ndimScalerFileName)
    mode = config.mode
    root = config.pathToTrainFolder

    if mode == 'norm':
        scaler_dict = DU.load_scaler(normScalerPath)
    else:
        scaler_dict = DU.load_scaler("ndim_scaler_ratios.pkl")

    xv,yv,zv,del_p = scaler_dict['xv'],scaler_dict['yv'],scaler_dict['zv'],scaler_dict['del_p']

    av_x,std_x,max_x = np.mean(xv), np.std(xv), max(xv)
    av_y,std_y,max_y = np.mean(yv), np.std(yv), max(yv)
    av_z,std_z,max_z = np.mean(zv), np.std(zv), max(zv)
    av_p,std_p,max_p = np.mean(del_p), np.std(del_p), max(del_p)

    print("Numbers")
    print("\tX:")
    print("\t\t Av:",av_x, ", Std:", std_x,", Max: ", max_x)
    print("\tY:")
    print("\t\t Av:",av_y, ", Std:", std_y,", Max: ", max_y)
    print("\tZ:")
    print("\t\t Av:",av_z, ", Std:", std_z,", Max: ", max_z)
    print("\tP:")
    print("\t\t Av:",av_p, ", Std:", std_p,", Max: ", max_p)


    fig =  plt.figure(figsize=(16,9))

    axs = [fig.add_subplot(221),fig.add_subplot(222),fig.add_subplot(223),fig.add_subplot(224)]

    axs[0].set_title("X Velo")
    axs[0].set_xlabel("Max/Input Ratio")
    axs[0].set_ylabel("Value")
    # axs[0].set_yscale('log')
    axs[0].hist(xv,bins=100)

    axs[1].set_title("Y Velo")
    axs[1].set_xlabel("Max/Input Ratio")
    axs[1].set_ylabel("Value")
    # axs[1].set_yscale('log')
    axs[1].hist(yv,bins=100)

    axs[2].set_title("Z Velo")
    axs[2].set_xlabel("Max/Input Ratio")
    axs[2].set_ylabel("Value")
    # axs[2].set_yscale('log')
    axs[2].hist(zv,bins=100)

    axs[3].set_title("Pressure")
    axs[3].set_xlabel("Max/Input Ratio") # either non-dim pressure, or... pressure ratio
    axs[3].set_ylabel("Value")
    # axs[3].set_yscale('log')
    axs[3].hist(del_p, bins=100)

    # Routine to find the outliers

    sim_folders = []
    for folder in os.listdir(root):
        # simple manifold, multiple_manifold, etc...
        sim_type_folder = os.path.join(root, folder)
        for sim_folder in os.listdir(sim_type_folder):
            sim_folder_path = os.path.join(sim_type_folder, sim_folder)
            sim_folders.append(sim_folder_path)

    x_out_index = []
    for i,val in enumerate(xv):
        if val > av_x + 6*std_x:
            x_out_index.append(i)

    y_out_index = []
    for i,val in enumerate(yv):
        if val > av_y + 6*std_y:
            y_out_index.append(i)

    z_out_index = []
    for i,val in enumerate(zv):
        if val > av_z + 6*std_z:
            z_out_index.append(i)

    p_out_index = []
    for i,val in enumerate(del_p):
        if val > av_p + 6*std_p:
            p_out_index.append(i)

    print(x_out_index)
    print(y_out_index)
    print(z_out_index)
    print(p_out_index)

