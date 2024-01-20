import json
import os
import re
import pickle
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def mesh_stats(mesh_qual_list):
    min_q = []
    max_q = []
    avg_q = []
    std_q = []

    for mesh_dict in mesh_qual_list:
        mesh_stats = mesh_dict
        min_q.extend([mesh_stats["Min"]])
        max_q.extend([mesh_stats["Max"]])
        avg_q.extend([mesh_stats["Avg"]])
        std_q.extend([mesh_stats["Std"]])

    min_avg_q = np.mean(min_q)
    min_std_q = np.std(min_q)
    max_avg_q = np.mean(max_q)
    max_std_q = np.std(max_q)
    avg_avg_q = np.mean(avg_q)
    avg_std_q = np.std(avg_q)
    std_avg_q = np.mean(std_q)
    std_std_q = np.std(std_q)

    eb = plt.errorbar(
        ["Min", "Max", "Avg", "Std"],
        [min_avg_q, max_avg_q, avg_avg_q, std_avg_q],
        [min_std_q, max_std_q, avg_std_q, std_std_q],
        fmt="o",
        ecolor="r",
    )

    print(
        [min_avg_q, max_avg_q, avg_avg_q, std_avg_q],
        [min_std_q, max_std_q, avg_std_q, std_std_q],
    )
    plt.title("Mesh Quality Statistics", fontsize=12)
    plt.xlabel("Ansys Mesh Quality Metric", fontsize=12)
    plt.ylabel("Number of Simulations", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def fluid_stats(fluid_params_list):
    Re = []
    Eps = []
    Visc = []
    Rho = []
    P = []
    V = []
    D = []

    for fluid_dict in fluid_params_list:
        Re.extend([fluid_dict["Re"]])
        Eps.extend([fluid_dict["Eps"]])
        Visc.extend([fluid_dict["Visc"]])
        Rho.extend([fluid_dict["Rho"]])
        P.extend([fluid_dict["P"]])
        V.extend([fluid_dict["V"]])
        D.extend([fluid_dict["D"]])

    plt.hist(Re)
    plt.title("Reynolds No", fontsize=20)
    plt.xlabel("Reynolds No Value", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=15)
    plt.xticks(rotation=30)
    plt.show()

    plt.hist(Eps)
    plt.title("Wall Roughness", fontsize=20)
    plt.xlabel("Non Dimensional Wall Roughness", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=15)
    plt.show()

    plt.hist(Visc)
    plt.title("Viscosity", fontsize=20)
    plt.xlabel("Pa-s", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=15)
    plt.show()

    plt.hist(Rho)
    plt.title("Density", fontsize=20)
    plt.xlabel("kg/m^3", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=15)
    plt.show()

    plt.hist(P)
    plt.title("Pressure", fontsize=20)
    plt.xlabel("Pa", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=15)
    plt.show()

    plt.hist(V)
    plt.title("Velocity", fontsize=20)
    plt.xlabel("m/s", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=15)
    plt.show()

    plt.hist(D)
    plt.title("Pipe Inlet Diameter", fontsize=20)
    plt.xlabel("m", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=15)
    plt.show()

def residual_stats(residual_list):
    cont = []
    x_velo = []
    y_velo = []
    z_velo = []
    k = []
    omega = []

    for res_dict in tqdm(residual_list):
        cont.extend([res_dict["continuity"]])
        x_velo.extend([res_dict["x-velocity"]])
        y_velo.extend([res_dict["y-velocity"]])
        z_velo.extend([res_dict["z-velocity"]])
        if "k" in res_dict:
            k.extend([res_dict["k"]])
        if "omega" in res_dict:
            omega.extend([res_dict["omega"]])

    print(len(list(filter(lambda c: c <= 0.00025, cont))) / len(cont))

    plt.figure(figsize=(10, 8))
    plt.hist(list(filter(lambda c: c < 0.001, cont)), bins=100)
    plt.title("Continuity Residual Error", fontsize=20)
    plt.xlabel("Error", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=15)
    plt.xticks(rotation=30)
    plt.xlim([0, 0.0010])
    plt.show()

    # plt.figure(figsize = (10,10))
    # plt.hist(x_velo,  bins = 1000)
    # plt.title("X Velocity")
    # plt.xlabel('Residual Error')
    # plt.ylabel('Number of Simulations')
    # plt.xticks(rotation=90)
    # plt.show()

    # plt.figure(figsize = (10,10))
    # plt.hist(y_velo,  bins = 1000)
    # plt.xlabel('Residual Error')
    # plt.ylabel('Number of Simulations')
    # plt.title("Y Velocity")

    # plt.xticks(rotation=90)
    # plt.show()

    # plt.figure(figsize = (10,10))
    # plt.hist(z_velo,  bins = 1000)
    # plt.xlabel('Residual Error')
    # plt.ylabel('Number of Simulations')
    # plt.title("Z Velocity")
    # plt.xticks(rotation=90)
    # plt.show()

    # plt.figure(figsize = (10,10))
    # plt.hist(k,  bins = 100)
    # plt.xlabel('Residual Error')
    # plt.ylabel('Number of Simulations')
    # plt.title("Kappa")
    # plt.xticks(rotation=90)
    # plt.show()

    # plt.figure(figsize = (10,10))
    # plt.hist(omega,  bins = 100)
    # plt.xlabel('Residual Error')
    # plt.ylabel('Number of Simulations')
    # plt.title("Omega")
    # plt.xticks(rotation=90)
    # plt.show()

def node_stats(node_count_list):
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    plt.hist(node_count_list)
    # print(np.mean(node_count_list), np.median(node_count_list), np.std(node_count_list), np.min(node_count_list), np.max(node_count_list))
    plt.title("Node Count", fontsize=20)
    plt.xlabel("Number of Nodes in Mesh", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=15)
    plt.xticks(rotation=45)
    plt.show()

def run_time_stats(run_time_list):
    rt_list = [i for i in run_time_list if i != None]
    print(
        np.min(rt_list),
        np.max(rt_list),
        np.mean(rt_list),
        np.std(rt_list),
        np.median(rt_list),
    )
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    plt.hist(rt_list, bins=50)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Seconds", fontsize=20)
    plt.ylabel("Number of Simulations", fontsize=20)
    plt.title("Run Time (s)", fontsize=20)
    plt.xticks(rotation=90)

    plt.show()


if __name__ == "__main__":
    with open(os.path.join("scripts", "visualizations", "config.json")) as f:
        config = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    stat_to_viz = config.metricToSee
    summary_data_file = config.summaryFileName
    
    # Grab simulation data
    sim_results = pickle.load(open(summary_data_file, "rb"))

    mesh_qual_list = sim_results["mesh_quality"]
    fluid_params_list = sim_results["fluid_params"]
    run_time_list = sim_results["run_times"]
    residual_list = sim_results["residuals"]
    node_count_list = sim_results["node_counts"]


    if stat_to_viz == "mesh":
        mesh_stats(mesh_qual_list)
    elif stat_to_viz == "fluid":
        fluid_stats(fluid_params_list)
    elif stat_to_viz == "residual":
        residual_stats(residual_list)
    elif stat_to_viz == "node":
        node_stats(node_count_list)
    elif stat_to_viz == "rune_time":
        run_time_stats(run_time_list)
    else:
        raise Exception("invalid metric")
