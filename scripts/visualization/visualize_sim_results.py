import json
import os
import random
from types import SimpleNamespace
from data.visualization.vis_functions import visualize_sim_data_magnitudes
from scripts.helpers import load_config


if __name__ == "__main__":
    config = load_config(
        os.path.join("scripts", "visualization", "visualize_sim_results.json")
    )
    train_dir = config.trainDir
    path_list = []
    for sim_type in os.listdir(train_dir):
        for sim_no in os.listdir(os.path.join(train_dir, sim_type)):
            path_list.append(os.path.join(train_dir, sim_type, sim_no))

    command = "n"
    print("enter 'exit' to quit, otherwise press any other character to continue")
    while command == "n":
        fpath = random.choice(path_list)
        fig = visualize_sim_data_magnitudes(fpath, include_wall=False)
        temp_input = str(input())
        if temp_input == "exit":
            command = "exit"
        else:
            fig
            del fig
