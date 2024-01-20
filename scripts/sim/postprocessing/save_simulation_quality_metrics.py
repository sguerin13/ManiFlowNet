import os
import re
import pickle
from tqdm import tqdm
import json
from types import SimpleNamespace

from scripts.helpers import load_config

def read_log_file(file_path,mesh_qual_list,fluid_params_list,run_time_list):
    # parse the log file

    with open(file_path,'r') as f:
        lines = f.readlines()
        
        for i in range(len(lines)):
            txt = lines[i]
            # file name
            if i == 0:
                pass

            # flow params
            if i == 1:
                flow_params_dict = {}
                Re = re.search("Re\: [0-9]*\.[0-9]*",txt).group(0)
                flow_params_dict['Re'] = float(Re.split(" ")[1])

                Eps = re.search("Eps\: \d+(\.\d+)?(e[+-]\d+)?",txt).group(0)
                flow_params_dict['Eps'] = float(Eps.split(" ")[1])

                Visc = re.search("Visc\: [0-9]*\.[0-9]*",txt).group(0)
                flow_params_dict['Visc'] = float(Visc.split(" ")[1])

                Rho = re.search("Rho\: [0-9]*\.[0-9]*",txt).group(0)
                flow_params_dict['Rho'] = float(Rho.split(" ")[1])

                P = re.search("P\: [0-9]*\.[0-9]*",txt).group(0)
                flow_params_dict['P'] = float(P.split(" ")[1])
                
                V = re.search("V\: [0-9]*\.[0-9]*",txt).group(0)
                flow_params_dict['V'] = float(V.split(" ")[1])

                D = re.search("D\: [0-9]*\.[0-9]*",txt).group(0)
                flow_params_dict['D'] = float(D.split(" ")[1])
                
                fluid_params_list.extend([flow_params_dict])

            if i == 2:
                mesh_qual_dict = {}

                Min = re.search("Min\: [0-9]*\.[0-9]*",txt).group(0)
                mesh_qual_dict['Min'] = float(Min.split(" ")[1])

                Max = re.search("Max\: [0-9]*\.[0-9]*",txt).group(0)
                mesh_qual_dict['Max'] = float(Max.split(" ")[1])

                Avg = re.search("Avg\: [0-9]*\.[0-9]*",txt).group(0)
                mesh_qual_dict['Avg'] = float(Avg.split(" ")[1])

                Std = re.search("Std\: [0-9]*\.[0-9]*",txt).group(0)
                mesh_qual_dict['Std'] = float(Std.split(" ")[1]) 

                mesh_qual_list.extend([mesh_qual_dict])

            if i == 3:
                run_time = re.search("Run Time\: [0-9]*\.[0-9]*",txt).group(0)
                if run_time is not None:
                    run_time_list.extend([float(run_time.split(" ")[2])])
                else:
                    pass
        
        if len(lines) < 4:
            run_time_list.extend([None])

    return mesh_qual_list, fluid_params_list, run_time_list

def read_res_file(file_path,residual_list):

    residual_dict = {}
    with open(file_path,'r') as f:
        lines = f.readlines()
        
        counter = 0
        # first pass, find all of the entry bounds
        for line in lines:

            # start of a region
            if re.search("^\(",line):
                value = re.search("^\(\(.*",line).group(0).split(" ")[1]
                value = value.split("\"")[1]
                # print(value)

            if re.search("^\)",line):
                # scientific notation
                last_res = re.search("^[0-9]*\t[0-9]*\.[0-9]*e\-[0-9]*",lines[counter-1])
                if last_res != None:
                    last_res = last_res.group(0)
                

                else: 
                    # decimal notation
                    last_res = re.search("^[0-9]*\t[0-9]*\.[0-9]*",lines[counter-1]).group(0)
                # print(last_res)
                last_res = last_res.split("\t")[1]
                residual_dict[value] = float(last_res)
                
            counter += 1
    
    residual_list.extend([residual_dict])
    return residual_list

def get_n_nodes(file_path, node_count_list):
    with open(file_path,'r') as f:
        lines = f.readlines()
        line = lines[-1]
        node_count_list.extend([int(re.search("^[0-9]*.*",line).group(0).split(",")[0])])
    return node_count_list

if __name__ == "__main__":
    config = load_config(os.path.join("scripts", "visualizations", "save_simulation_quality_metrics.json"))
    
    summary_data_file = config.summaryFileName
    root_dir = config.rootDataDir
    
    mesh_qual_list = []
    fluid_params_list = []
    run_time_list = []
    residual_list = []
    node_count_list = []


    # TODO: update to gather stats before the train/val split
    i = 0
    for set_type in ['train','val']:
        sim_dir = os.path.join(root_dir, set_type)
        for sim_folder in os.listdir(sim_dir):
            sim_folder_path = os.path.join(sim_dir,sim_folder)
            for folder in tqdm(os.listdir(sim_folder_path)):
                try:
                    log_file = os.path.join(os.path.join(sim_folder_path,folder),
                                            "log.txt")
                    res_file = os.path.join(os.path.join(sim_folder_path,folder),
                                            "res_file.txt")
                    sim_file = os.path.join(os.path.join(sim_folder_path,folder),
                                            "sol_file.txt")

                    mesh_qual_list, fluid_params_list, run_time_list = \
                    read_log_file(log_file,mesh_qual_list,fluid_params_list,run_time_list)

                    residual_list = read_res_file(res_file,residual_list)
                    node_count_list = get_n_nodes(sim_file,node_count_list)
                    i+=1
                except:
                    print(folder)
                    break
                # print(i)


    summary_data = {
                    'fluid_params' : fluid_params_list,
                    'mesh_quality' : mesh_qual_list,
                    'run_times'    : run_time_list,
                    'residuals'    : residual_list,
                    'node_counts'  : node_count_list,
    }

    pickle.dump(summary_data,open(os.path.join("outputs",summary_data_file),'wb'))