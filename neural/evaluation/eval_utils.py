import copy
import torch
import numpy as np
import os
from tqdm import tqdm
import plotly.graph_objects as go

from neural.models.convpoint.PLConvPointUNet import PLConvPointUNet
from neural.models.minkeng.PLMinkEngUNet import PLMinkEngUNet
from models.convpoint.Dataset import Dataset as CPDataset
from models.minkeng.Dataset import custom_collate_fn, Dataset as MEDataset


import pickle
from torch.utils.data import DataLoader
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import MinkowskiEngine as ME
import data.wrangling.data_utils as DU


# Plotting with a slider: # https://linuxtut.com/en/5c75a1c0cea56b6bf6cb/

#########################################
#                MINKENG                #
#########################################

def get_ME_eval_output(
    val_loader,
    net,
    val_index,
):
    for i, batch in enumerate(val_loader):
        if val_index:
            if i == val_index:
                batch = val_loader[val_index]
                x_coords, x_feats, y_feats = batch
                sparse_input = ME.SparseTensor(
                    features=x_feats,
                    coordinates=x_coords,
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    device=torch.device("cuda"),
                )
                sparse_output = ME.SparseTensor(
                    features=y_feats,
                    coordinates=x_coords,
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    device=torch.device("cuda"),
                )
                out = net.forward(sparse_input)
                # x_F = x_feats
                out_F = out.F.detach().cpu().numpy()
                out_C = out.C.detach().cpu().numpy()
                target_F = sparse_output.F.cpu().numpy()
                return out_F, out_C, target_F
            else:
                continue

        else:
            raise Exception("bad logic")

def get_ME_eval_output_for_single_sample(
    val_loader,
    net,
):
    # will only have one sample
    for i, batch in enumerate(val_loader):
        x_coords, x_feats, y_feats = batch
        sparse_input = ME.SparseTensor(
            features=x_feats,
            coordinates=x_coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=torch.device("cuda"),
        )
        sparse_output = ME.SparseTensor(
            features=y_feats,
            coordinates=x_coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device=torch.device("cuda"),
        )
        out = net.forward(sparse_input)
        out_F = out.F.detach().cpu().numpy()
        out_C = out.C.detach().cpu().numpy()
        target_F = sparse_output.F.cpu().numpy()
        return out_F, out_C, target_F

def get_ME_net(ckpt_path):
    net = PLMinkEngUNet.load_from_checkpoint(ckpt_path)
    net = net.eval()
    net.cuda(0)
    return net

def get_ME_val_dataloader(
    val_path,
    n_voxels,
    bbox_scaler,
    velo_scaler,
    pressure_scaler,
    fparams_scaler,
    context_values_included,
    sample_qty,
    rotate,
    surface_nodes,
    add_normals,
    VP_all_inputs,
    non_dimensionalize,
    sample_index=None,
):
    val_set = MEDataset(
        root=val_path,
        n_voxels=n_voxels,
        bbox_scaler=bbox_scaler,
        velo_scaler=velo_scaler,
        pressure_scaler=pressure_scaler,
        fparams_scaler=fparams_scaler,
        sample_qty=sample_qty,
        rotate=rotate,
        surface_nodes=surface_nodes,
        add_normals=add_normals,
        context_values_included=context_values_included,
        VP_all_inputs=VP_all_inputs,
        non_dimensionalize=non_dimensionalize,
    )

    if sample_index is not None:
        subset = torch.utils.data.Subset(val_set, [sample_index])
        val = DataLoader(
            subset,
            batch_size=1,
            pin_memory=False,
            collate_fn=custom_collate_fn,
            num_workers=1,
            shuffle=False,
        )
    else:
        val = DataLoader(
            val_set,
            batch_size=1,
            pin_memory=False,
            collate_fn=custom_collate_fn,
            num_workers=1,
            shuffle=False,
        )
    return val

#########################################
#                ConvPoint              #
#########################################
def get_CP_eval_output(
    val_loader,
    net,
    val_index,
):
    for i, batch in enumerate(val_loader):
        if val_index:
            if i == val_index:
                x_pts, x_feats, y_feats = batch
                out = net.forward(x_feats, x_pts)
                return out, y_feats
            else:
                continue

        else:
            raise Exception("bad logic")

def get_CP_eval_output_for_single_sample(
    val_loader,
    net,
):
    # will only have one sample
    for i, batch in enumerate(val_loader):
        x_pts, x_feats, y_feats = batch
        out = net.forward(x_feats.to('cuda:0'), x_pts.to('cuda:0'))
        return out.detach().cpu().numpy().reshape(-1,4), x_pts.detach().cpu().numpy().reshape(-1,3), y_feats.detach().cpu().numpy().reshape(-1,4)

def get_CP_net(ckpt_path):
    net = PLConvPointUNet.load_from_checkpoint(ckpt_path)
    net = net.eval()
    net.cuda(0)
    return net

def get_CP_val_dataloader(
    val_path,
    bbox_scaler,
    velo_scaler,
    pressure_scaler,
    fparams_scaler,
    context_values_included,
    sample_qty,
    rotate,
    surface_nodes,
    add_normals,
    VP_all_inputs,
    non_dimensionalize,
    sample_index=None,
):
    
    val_set = CPDataset(
        root=val_path,
        bbox_scaler=bbox_scaler,
        velo_scaler=velo_scaler,
        pressure_scaler=pressure_scaler,
        fparams_scaler=fparams_scaler,
        sample_qty=sample_qty,
        non_dimensionalize=non_dimensionalize,
        rotate=rotate,
        surface_nodes=surface_nodes,
        context_values_included=context_values_included,
        add_normals=add_normals,
        VP_all_inputs=VP_all_inputs,
        
    )

    if sample_index is not None:
        subset = torch.utils.data.Subset(val_set, [sample_index])
        val = DataLoader(
            subset,
            batch_size=1,
            pin_memory=False,
            num_workers=1,
            shuffle=False,
        )
    else:
        val = DataLoader(
            val_set,
            batch_size=1,
            pin_memory=False,
            num_workers=1,
            shuffle=False,
        )
    return val

########################################
#               HELPERS                #
########################################

def plot_discrete_bars(x, y):
    value_dict = {}
    for i in zip(x, y):
        if x not in value_dict:
            value_dict[x] = [y]
        else:
            value_dict[x].append(y)

    mean_list = []
    std_list = []
    key_list = [x for x in value_dict]
    for x in value_dict:
        mean_list.append(np.mean(value_dict[x]))
        std_list.append(np.std(value_dict[x]))

    fig = go.Figure(
        data=go.Scatter(
            x=key_list,
            y=mean_list,
            error_y=dict(
                type="data",  # value of error bar given in data coordinates
                array=std_list,
                visible=True,
            ),
        )
    )
    fig.show()

def get_scalers(scaler_path):
    scaler_dict = {}
    with open(os.path.join(scaler_path, "bbox_scaler.pkl"), "rb") as f:
        bbox_scaler = pickle.load(f)
        scaler_dict["bbox"] = bbox_scaler

    with open(os.path.join(scaler_path, "velo_scaler.pkl"), "rb") as f:
        velo_scaler = pickle.load(f)
        scaler_dict["velo"] = velo_scaler

    with open(os.path.join(scaler_path, "P_scaler.pkl"), "rb") as f:
        pressure_scaler = pickle.load(f)
        scaler_dict["pressure"] = pressure_scaler

    with open(os.path.join(scaler_path, "fluid_prop_scaler.pkl"), "rb") as f:
        fluid_prop_scaler = pickle.load(f)
        scaler_dict["fparams"] = fluid_prop_scaler

    with open(os.path.join(scaler_path, "props_included.pkl"), "rb") as f:
        context_values_included = pickle.load(f)
        scaler_dict["context"] = context_values_included

    return scaler_dict

def get_normalization_values(val_path):
    vp_values = []
    for folder in os.listdir(val_path):
        sim_type_folder = os.path.join(val_path, folder)
        for sim_folder in os.listdir(sim_type_folder):
            sim_folder_path = os.path.join(sim_type_folder, sim_folder)
            fluid_prop_path = os.path.join(sim_folder_path, "fluid_props.pkl")
            fluid_prop_dict = DU.load_fluid_prop_data_as_dict(fluid_prop_path)
            v, p = fluid_prop_dict["V"], fluid_prop_dict["P"]
            vp_values.append((v, p))

    return vp_values

def convert_ME_feats_and_coords_to_df(
    output_features,
    output_coords,
    target_features,
    max_location_scale=1,
    surface_only=False,
):
    max_location_val = np.max(output_coords)
    location_scaler = max_location_scale / max_location_val

    df_pred_list = []
    df_y_list = []
    pts = [output_coords[i, 1:] for i in range(output_coords.shape[0])]
    for i, xyz in enumerate(pts):
        if surface_only:
            pass
            # # TODO: update for surface restricted
            # # if (target_F[i,-1] > 0):
            # bin_locations[xyz[0], xyz[1], xyz[2]] = 1
            # bin_list.append([xyz[0], xyz[1], xyz[2]])

            # pred_feat_list.extend([out_F[i, :]])
            # pred_values[xyz[0], xyz[1], xyz[2], 0] = out_F[i, 0]  # pressure
            # pred_values[xyz[0], xyz[1], xyz[2], 1] = out_F[i, 1]  # x_velo
            # pred_values[xyz[0], xyz[1], xyz[2], 2] = out_F[i, 2]  # y_velo
            # pred_values[xyz[0], xyz[1], xyz[2], 3] = out_F[i, 3]  # z_velo

            # y_feat_list.append(target_F[i, :4].tolist())
            # y_values[xyz[0], xyz[1], xyz[2], 0] = target_F[i, 0]
            # y_values[xyz[0], xyz[1], xyz[2], 1] = target_F[i, 1]
            # y_values[xyz[0], xyz[1], xyz[2], 2] = target_F[i, 2]
            # y_values[xyz[0], xyz[1], xyz[2], 3] = target_F[i, 3]
        else:
            xyz = xyz * location_scaler
            x, y, z = xyz[0], xyz[1], xyz[2]
            p, vx, vy, vz = (
                output_features[i, 0],
                output_features[i, 1],
                output_features[i, 2],
                output_features[i, 3],
            )
            p_gt, vx_gt, vy_gt, vz_gt = (
                target_features[i, 0],
                target_features[i, 1],
                target_features[i, 2],
                target_features[i, 3],
            )
            df_pred_list.append([x, y, z, p, vx, vy, vz])
            df_y_list.append([x, y, z, p_gt, vx_gt, vy_gt, vz_gt])

    header = ["X", "Y", "Z", "Pressure", "X-Velo", "Y-Velo", "Z-Velo"]
    df_pred = pd.DataFrame(df_pred_list, columns=header)
    df_y = pd.DataFrame(df_y_list, columns=header)
    return df_pred, df_y

def convert_CP_feats_and_coords_to_df(
    output_features,
    output_coords,
    target_features,
    max_location_scale=1,
    surface_only=False,
):
    max_location_val = np.max(output_coords)
    location_scaler = max_location_scale / max_location_val

    df_pred_list = []
    df_y_list = []
    pts = [output_coords[i,:] for i in range(output_coords.shape[0])]
    for i, xyz in enumerate(pts):
        if surface_only:
            pass
            # # TODO: update for surface restricted
            # # if (target_F[i,-1] > 0):
            # bin_locations[xyz[0], xyz[1], xyz[2]] = 1
            # bin_list.append([xyz[0], xyz[1], xyz[2]])

            # pred_feat_list.extend([out_F[i, :]])
            # pred_values[xyz[0], xyz[1], xyz[2], 0] = out_F[i, 0]  # pressure
            # pred_values[xyz[0], xyz[1], xyz[2], 1] = out_F[i, 1]  # x_velo
            # pred_values[xyz[0], xyz[1], xyz[2], 2] = out_F[i, 2]  # y_velo
            # pred_values[xyz[0], xyz[1], xyz[2], 3] = out_F[i, 3]  # z_velo

            # y_feat_list.append(target_F[i, :4].tolist())
            # y_values[xyz[0], xyz[1], xyz[2], 0] = target_F[i, 0]
            # y_values[xyz[0], xyz[1], xyz[2], 1] = target_F[i, 1]
            # y_values[xyz[0], xyz[1], xyz[2], 2] = target_F[i, 2]
            # y_values[xyz[0], xyz[1], xyz[2], 3] = target_F[i, 3]
        else:
            xyz = xyz * location_scaler
            x, y, z = xyz[0], xyz[1], xyz[2]
            p, vx, vy, vz = (
                output_features[i, 0],
                output_features[i, 1],
                output_features[i, 2],
                output_features[i, 3],
            )
            p_gt, vx_gt, vy_gt, vz_gt = (
                target_features[i, 0],
                target_features[i, 1],
                target_features[i, 2],
                target_features[i, 3],
            )
            df_pred_list.append([x, y, z, p, vx, vy, vz])
            df_y_list.append([x, y, z, p_gt, vx_gt, vy_gt, vz_gt])

    header = ["X", "Y", "Z", "Pressure", "X-Velo", "Y-Velo", "Z-Velo"]
    df_pred = pd.DataFrame(df_pred_list, columns=header)
    df_y = pd.DataFrame(df_y_list, columns=header)
    return df_pred, df_y

def inverse_scale(scaler_dict, df_pred, df_y, ndim=False, V_in=None, P_out=None):
    df_pred_scaled = df_pred.copy()
    df_y_scaled = df_y.copy()

    df_pred_scaled[["X-Velo", "Y-Velo", "Z-Velo"]] = scaler_dict[
        "velo"
    ].inverse_transform(df_pred_scaled[["X-Velo", "Y-Velo", "Z-Velo"]])

    df_pred_scaled[["Pressure"]] = scaler_dict["pressure"].inverse_transform(
        df_pred_scaled[["Pressure"]]
    )

    df_y_scaled[["X-Velo", "Y-Velo", "Z-Velo"]] = scaler_dict["velo"].inverse_transform(
        df_y_scaled[["X-Velo", "Y-Velo", "Z-Velo"]]
    )
    df_y_scaled[["Pressure"]] = scaler_dict["pressure"].inverse_transform(
        df_y_scaled[["Pressure"]]
    )

    return df_pred_scaled, df_y_scaled

def inverse_scale_numpy_array(
    scaler_dict, out_f, target_f, ndim=False, V_in=None, P_out=None
):
    """
    Inverse scales and also undoes non-dimensionalization if the model
    had non-dimensional values

    """
    out_P = out_f[:, 0].reshape(-1, 1)
    out_V = out_f[:, 1:]
    target_P = target_f[:, 0].reshape(-1, 1)
    target_V = target_f[:, 1:]

    out_V = scaler_dict["velo"].inverse_transform(out_V)
    out_P = scaler_dict["pressure"].inverse_transform(out_P)
    target_V = scaler_dict["velo"].inverse_transform(target_V)
    target_P = scaler_dict["pressure"].inverse_transform(target_P)

    # reversing the non-dimensionalization
    if ndim:
        if V_in and P_out:
            out_V = out_V * V_in
            target_V = target_V * V_in

            out_P = out_P * P_out + P_out
            target_P = target_P * P_out + P_out
        else:
            raise Exception("V and P must be present")

    out = np.hstack((out_P, out_V))
    target = np.hstack((target_P, target_V))
    return out, target


def calculate_delta(df_pred, df_y):
    delta_df_list = []
    for index, row in df_y.iterrows():
        pred_row = df_pred.loc[index]
        delta_row = [
            row["X"],
            row["Y"],
            row["Z"],
            pred_row["Pressure"] - row["Pressure"],
            pred_row["X-Velo"] - row["X-Velo"],
            pred_row["Y-Velo"] - row["Y-Velo"],
            pred_row["Z-Velo"] - row["Z-Velo"],
        ]
        delta_df_list.append(delta_row)
    header = ["X", "Y", "Z", "Pressure", "X-Velo", "Y-Velo", "Z-Velo"]
    delta_df = pd.DataFrame(delta_df_list, columns=header)
    return delta_df


def calculate_percent_error(df_pred, df_y):
    perc_df_list = []
    for index, row in df_y.iterrows():
        if row["X-Velo"] != 0.0 and row["Y-Velo"] != 0.0 and row["Z-Velo"] != 0.0:
            pred_row = df_pred.loc[index]
            perc_row = [
                (pred_row["Pressure"] - row["Pressure"]) / row["Pressure"],
                (pred_row["X-Velo"] - row["X-Velo"]) / (row["X-Velo"]),
                (pred_row["Y-Velo"] - row["Y-Velo"]) / (row["Y-Velo"]),
                (pred_row["Z-Velo"] - row["Z-Velo"]) / (row["Z-Velo"]),
            ]
            perc_df_list.append(perc_row)
    header = ["Pressure", "X-Velo", "Y-Velo", "Z-Velo"]
    perc_df = pd.DataFrame(perc_df_list, columns=header)
    return perc_df


def calculate_mean_errors(perc_error_df):
    av_x_err = perc_error_df["X-Velo"].mean(skipna=True)
    av_y_err = perc_error_df["Y-Velo"].mean(skipna=True)
    av_z_err = perc_error_df["Z-Velo"].mean(skipna=True)
    av_p_err = perc_error_df["Pressure"].mean(skipna=True)
    return dict(x=av_x_err, y=av_y_err, z=av_z_err, p=av_p_err)


def calculate_bulk_metrics(error_list):
    best_results = {
        "clean_single_pipe_sims": 1000000,
        "clean_simple_manifold_sims": 100000,
        "clean_split_pipe_sims": 100000,
        "clean_single_to_many_sims": 100000,
    }

    best_single_pipe = {}
    best_split_pipe = {}
    best_one_to_many_pipe = {}
    best_simple_manifold = {}

    x_velo_error = []
    y_velo_error = []
    z_velo_error = []
    p_error = []

    for metric in error_list:
        av_error = (
            abs(metric["x"]) + abs(metric["y"]) + abs(metric["z"]) + abs(metric["p"])
        ) / 4
        x_velo_error.append(metric["x"])
        y_velo_error.append(metric["y"])
        z_velo_error.append(metric["z"])
        p_error.append(metric["p"])

        pipe_type = metric["file_path"].split("/")[-2]
        if av_error < best_results[pipe_type]:
            best_results[pipe_type] = av_error

    fig, axs = plt.subplots(1, 4, sharey=False)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(x_velo_error, rwidth=0.95)
    axs[0].set_title("X Velocity Error Ratio")
    axs[1].hist(y_velo_error, rwidth=0.95)
    axs[1].set_title("Y Velocity Error Ratio")
    axs[2].hist(z_velo_error, rwidth=0.95)
    axs[2].set_title("Z Velocity Error Ratio")
    axs[3].hist(p_error, rwidth=0.95)
    axs[3].set_title("Pressure Error Ratio")
    plt.show()
    return best_results


def calculate_plot_legend_scale(df_pred, df_y, df_delta):
    x_scale_min = min([min(df_pred["X-Velo"]), min(df_y["X-Velo"])])
    x_scale_max = max([max(df_pred["X-Velo"]), max(df_y["X-Velo"])])
    x_scale_tuple = (x_scale_min, x_scale_max)

    y_scale_min = min([min(df_pred["Y-Velo"]), min(df_y["Y-Velo"])])
    y_scale_max = max([max(df_pred["Y-Velo"]), max(df_y["Y-Velo"])])
    y_scale_tuple = (y_scale_min, y_scale_max)

    z_scale_min = min([min(df_pred["Z-Velo"]), min(df_y["Z-Velo"])])
    z_scale_max = max([max(df_pred["Z-Velo"]), max(df_y["Z-Velo"])])
    z_scale_tuple = (z_scale_min, z_scale_max)

    p_scale_min = min([min(df_pred["Pressure"]), min(df_y["Pressure"])])
    p_scale_max = max([max(df_pred["Pressure"]), max(df_y["Pressure"])])
    p_scale_tuple = (p_scale_min, p_scale_max)

    return x_scale_tuple, y_scale_tuple, z_scale_tuple, p_scale_tuple


def calculate_plot_magnitude_scale(df_pred, df_y=None):
    if df_y is not None:
        v_scale_min = min([min(df_pred["velo-magnitude"]), min(df_y["velo-magnitude"])])
        v_scale_max = max([max(df_pred["velo-magnitude"]), max(df_y["velo-magnitude"])])
        v_scale_tuple = (v_scale_min, v_scale_max)
        p_scale_min = min([min(df_pred["Pressure"]), min(df_y["Pressure"])])
        p_scale_max = max([max(df_pred["Pressure"]), max(df_y["Pressure"])])
        p_scale_tuple = (p_scale_min, p_scale_max)

    else:
        v_scale_min = min(df_pred["velo-magnitude"])
        v_scale_max = max(df_pred["velo-magnitude"])
        v_scale_tuple = (v_scale_min, v_scale_max)
        p_scale_min = min(df_pred["Pressure"])
        p_scale_max = max(df_pred["Pressure"])
        p_scale_tuple = (p_scale_min, p_scale_max)

    scale_factor = abs(max(df_pred["X"]) - min(df_pred["X"]))

    return v_scale_tuple, p_scale_tuple, scale_factor


@DeprecationWarning
def create_plot(
    df_pred,
    df_y,
    df_delta,
    x_scale_tuple,
    y_scale_tuple,
    z_scale_tuple,
    pressure_scale_tuple,
    scale_factor,
    voxel_size,
):
    subplot_titles = ("Output", "Ground Truth", "Delta")
    pred_metric_dict = {"X-Velo": [], "Y-Velo": [], "Z-Velo": [], "Pressure": []}
    y_metric_dict = {"X-Velo": [], "Y-Velo": [], "Z-Velo": [], "Pressure": []}
    delta_metric_dict = {"X-Velo": [], "Y-Velo": [], "Z-Velo": [], "Pressure": []}

    for metric in ["X-Velo", "Y-Velo", "Z-Velo", "Pressure"]:
        if metric == "X-Velo":
            scale_min = x_scale_tuple[0]
            scale_max = x_scale_tuple[1]

        if metric == "Y-Velo":
            scale_min = y_scale_tuple[0]
            scale_max = y_scale_tuple[1]

        if metric == "Z-Velo":
            scale_min = z_scale_tuple[0]
            scale_max = z_scale_tuple[1]

        if metric == "Pressure":
            scale_min = pressure_scale_tuple[0]
            scale_max = pressure_scale_tuple[1]

        for step in np.arange(0, scale_factor, scale_factor / 20):
            df_pred_filt = df_pred[(df_pred["X"] < step)]
            df_y_filt = df_y[(df_y["X"] < step)]
            df_delta_filt = df_delta[(df_delta["X"] < step)]

            pred_metric_dict[metric].append(
                go.Scatter3d(
                    x=df_pred_filt["X"],
                    y=df_pred_filt["Y"],
                    z=df_pred_filt["Z"],
                    mode="markers",
                    visible=False,
                    scene="scene1",
                    marker=dict(
                        size=voxel_size,
                        showscale=True,
                        symbol="square",
                        cmin=scale_min,
                        cmax=scale_max,
                        color=df_pred_filt[
                            metric
                        ],  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.5,
                    ),
                    text=[
                        "pred " + metric + ": {}".format(x)
                        for x in df_pred_filt[metric]
                    ],
                )
            )

            y_metric_dict[metric].append(
                go.Scatter3d(
                    x=df_y_filt["X"],
                    y=df_y_filt["Y"],
                    z=df_y_filt["Z"],
                    mode="markers",
                    visible=False,
                    scene="scene2",
                    marker=dict(
                        size=voxel_size,
                        showscale=True,
                        symbol="square",
                        cmin=scale_min,
                        cmax=scale_max,
                        color=df_y_filt[
                            metric
                        ],  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.5,
                    ),
                    text=["y " + metric + ": {}".format(x) for x in df_y_filt[metric]],
                )
            )

            delta_metric_dict[metric].append(
                go.Scatter3d(
                    x=df_delta_filt["X"],
                    y=df_delta_filt["Y"],
                    z=df_delta_filt["Z"],
                    mode="markers",
                    visible=False,
                    scene="scene3",
                    marker=dict(
                        size=voxel_size,
                        showscale=True,
                        cmin=scale_min,
                        cmax=scale_max,
                        symbol="square",
                        color=df_delta_filt[
                            metric
                        ],  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.5,
                    ),
                    text=[
                        "delta " + metric + ": {}".format(x)
                        for x in df_delta_filt[metric]
                    ],
                )
            )
            # fig.update_traces(showlegend=False)

    data = (
        pred_metric_dict["X-Velo"]
        + y_metric_dict["X-Velo"]
        + delta_metric_dict["X-Velo"]
        + pred_metric_dict["Y-Velo"]
        + y_metric_dict["Y-Velo"]
        + delta_metric_dict["Y-Velo"]
        + pred_metric_dict["Z-Velo"]
        + y_metric_dict["Z-Velo"]
        + delta_metric_dict["Z-Velo"]
        + pred_metric_dict["Pressure"]
        + y_metric_dict["Pressure"]
        + delta_metric_dict["Pressure"]
    )
    step_len = len(pred_metric_dict["X-Velo"])
    steps = {"X": [], "Y": [], "Z": [], "P": []}

    # Building up the dictionary for the sliders
    for step in range(step_len):
        x_velo_false, y_velo_false, z_velo_false, pressure_false = (
            [False] * (step_len),
            [False] * (step_len),
            [False] * (step_len),
            [False] * (step_len),
        )
        x_velo_true, y_velo_true, z_velo_true, pressure_true = (
            copy.copy(x_velo_false),
            copy.copy(y_velo_false),
            copy.copy(z_velo_false),
            copy.copy(pressure_false),
        )
        x_velo_true[step], y_velo_true[step], z_velo_true[step], pressure_true[step] = (
            True,
            True,
            True,
            True,
        )

        step_x = dict(
            method="update",
            args=[
                {
                    "visible": x_velo_true
                    + x_velo_true
                    + x_velo_true
                    + y_velo_false
                    + y_velo_false
                    + y_velo_false
                    + z_velo_false
                    + z_velo_false
                    + z_velo_false
                    + pressure_false
                    + pressure_false
                    + pressure_false
                },
                {"title": "X step: {}".format(step)},
            ],
        )
        step_y = dict(
            method="update",
            args=[
                {
                    "visible": x_velo_false
                    + x_velo_false
                    + x_velo_false
                    + y_velo_true
                    + y_velo_true
                    + y_velo_true
                    + z_velo_false
                    + z_velo_false
                    + z_velo_false
                    + pressure_false
                    + pressure_false
                    + pressure_false
                },
                {"title": "Y step: {}".format(step)},
            ],
        )

        step_z = dict(
            method="update",
            args=[
                {
                    "visible": x_velo_false
                    + x_velo_false
                    + x_velo_false
                    + y_velo_false
                    + y_velo_false
                    + y_velo_false
                    + z_velo_true
                    + z_velo_true
                    + z_velo_true
                    + pressure_false
                    + pressure_false
                    + pressure_false
                },
                {"title": "Z step: {}".format(step)},
            ],
        )

        step_p = dict(
            method="update",
            args=[
                {
                    "visible": x_velo_false
                    + x_velo_false
                    + x_velo_false
                    + y_velo_false
                    + y_velo_false
                    + y_velo_false
                    + z_velo_false
                    + z_velo_false
                    + z_velo_false
                    + pressure_true
                    + pressure_true
                    + pressure_true
                },
                {"title": "P step: {}".format(step)},
            ],
        )

        steps["X"].append(step_x)
        steps["Y"].append(step_y)
        steps["Z"].append(step_z)
        steps["P"].append(step_p)

    sliders = {}
    for key, traces in steps.items():
        slider = [
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=50),
                steps=traces,
            )
        ]
        sliders[key] = slider

    # building up the button list
    buttons = []
    for key, slider in sliders.items():
        slider_active = slider[0]["active"]
        slider_visible = slider[0]["steps"][slider_active]["args"][0]["visible"]
        button = dict(
            label=key,
            method="update",
            args=[
                dict(visible=slider_visible),
                dict(
                    title="{} step: {}".format(key, slider_active),
                    yaxis=dict(title="y {}".format(key)),
                    sliders=slider,
                ),
            ],
        )
        # animate_button = dict(label="Animate" + key,
        #                       frame_dur
        #                       method="animate",
        #                       args=[dict(data=steps[key])])

        buttons.append(button)
        # buttons.append(animate_button)

    updatemenus = [dict(active=0, type="buttons", buttons=buttons)]

    layout = go.Layout(
        title="X step: 0",
        xaxis=dict(domain=[0, 0.3], anchor="x1"),
        xaxis2=dict(domain=[0.35, 0.65], anchor="x2"),
        xaxis3=dict(domain=[0.7, 1.0], anchor="x3"),
        yaxis=dict(domain=[0.0, 1.0], anchor="y1"),
        yaxis2=dict(domain=[0.0, 1.0], anchor="y2"),
        yaxis3=dict(domain=[0.0, 1.0], anchor="y3"),
        font=dict(size=16),
        hovermode="x unified",
        hoverlabel=dict(font_size=16),
        sliders=sliders["X"],
        updatemenus=updatemenus,
        showlegend=False,
    )

    fig = go.Figure(dict(data=data, layout=layout))
    fig.set_subplots(
        1,
        3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Output", "Ground Truth", "Delta"),
        horizontal_spacing=0.1,
    )
    fig.update_scenes(aspectmode="data")
    fig.show()


# 9/18/23 - New Plotting Functions
def plot_output(value_data_frame, v_scale, p_scale, scale_factor):
    value_data_frame["velo-magnitude"] = np.linalg.norm(
        value_data_frame[["X-Velo", "Y-Velo", "Z-Velo"]].values, axis=1
    )

    pnt_cld_value_dict = {"velo-magnitude": [], "Pressure": []}
    for metric in ["velo-magnitude", "Pressure"]:
        if metric == "velo-magnitude":
            scale_min = v_scale[0]
            scale_max = v_scale[1]

        if metric == "Pressure":
            scale_min = p_scale[0]
            scale_max = p_scale[1]

        for step in np.arange(0, scale_factor * 1.1, scale_factor / 20):
            pnt_cld_at_x = value_data_frame[value_data_frame["X"] < step]
            pnt_cld_value_dict[metric].append(
                go.Scatter3d(
                    x=pnt_cld_at_x["X"],
                    y=pnt_cld_at_x["Y"],
                    z=pnt_cld_at_x["Z"],
                    mode="markers",
                    visible=False,
                    scene="scene1" if metric == "velo-magnitude" else "scene2",
                    marker=dict(
                        size=4,
                        showscale=True,
                        symbol="square",
                        cmin=scale_min,
                        cmax=scale_max,
                        colorbar={"x": 0.5 if metric == "velo-magnitude" else 1},
                        color=pnt_cld_at_x[
                            metric
                        ],  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.5,
                    ),
                    text=[metric + ": {}".format(x) for x in pnt_cld_at_x[metric]],
                )
            )

    # build up the dictionary for the slider
    data = pnt_cld_value_dict["velo-magnitude"] + pnt_cld_value_dict["Pressure"]
    step_len = len(pnt_cld_value_dict["velo-magnitude"])
    steps = {"V": []}

    for step in range(step_len):
        velo_false, pressure_false = [False] * (step_len), [False] * (step_len)
        velo_true, pressure_true = copy.copy(velo_false), copy.copy(pressure_false)
        velo_true[step], pressure_true[step] = True, True

        step_v = dict(
            method="update",
            args=[
                {"visible": velo_true + pressure_true},
                {"title": "step: {}".format(step)},
            ],
        )

        steps["V"].append(step_v)

    sliders = {}
    for key, traces in steps.items():
        slider = [
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=50),
                steps=traces,
            )
        ]
        sliders[key] = slider

    layout = go.Layout(
        title="X step: 0",
        font=dict(size=16),
        hovermode="x unified",
        hoverlabel=dict(font_size=16),
        sliders=sliders["V"],
        showlegend=False,
    )

    fig = go.Figure(dict(data=data, layout=layout))
    fig.set_subplots(
        1,
        2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Velocity", "Pressure"),
        horizontal_spacing=0.1,
    )
    camera = dict(eye=dict(x=0, y=2.5, z=0))

    fig.layout.scene1.camera = camera
    fig.layout.scene2.camera = camera
    fig.update_scenes(aspectmode="data")
    fig.show()
    return fig


def plot_output_metric(pred_data_frame, gt_data_frame, scale, scale_factor, metric, n_slices = 20, eye = dict(x=0, y=0, z=2.5)):
    pnt_cld_value_dict = {"pred": [], "gt": []}
    scale_min = scale[0]
    scale_max = scale[1]

    for i in ["pred", "gt"]:
        for step in np.arange(0, scale_factor * 1.1, scale_factor / n_slices):
            pnt_cld_at_x = (
                pred_data_frame[pred_data_frame["X"] < step]
                if i == "pred"
                else gt_data_frame[gt_data_frame["X"] < step]
            )
            pnt_cld_value_dict[i].append(
                go.Scatter3d(
                    x=pnt_cld_at_x["X"],
                    y=pnt_cld_at_x["Y"],
                    z=pnt_cld_at_x["Z"],
                    mode="markers",
                    visible=False,
                    scene="scene1" if i == "pred" else "scene2",
                    marker=dict(
                        size=8,
                        showscale=True,
                        symbol="circle",
                        cmin=scale_min,
                        cmax=scale_max,
                        colorbar={"x": 1},
                        color=pnt_cld_at_x[
                            metric
                        ],  # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=1.0,
                    ),
                    text=[i + ": {}".format(x) for x in pnt_cld_at_x[metric]],
                )
            )

    # build up the dictionary for the slider
    data = pnt_cld_value_dict["pred"] + pnt_cld_value_dict["gt"]
    step_len = len(pnt_cld_value_dict["pred"])
    steps = {"V": []}

    for step in range(step_len):
        velo_false, pressure_false = [False] * (step_len), [False] * (step_len)
        velo_true, pressure_true = copy.copy(velo_false), copy.copy(pressure_false)
        velo_true[step], pressure_true[step] = True, True

        step_v = dict(
            method="update",
            args=[
                {"visible": velo_true + pressure_true},
                {"title": "step: {}".format(step)},
            ],
        )

        steps["V"].append(step_v)

    sliders = {}
    for key, traces in steps.items():
        slider = [
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=50),
                steps=traces,
            )
        ]
        sliders[key] = slider

    layout = go.Layout(
        title="X step: 0",
        font=dict(size=16),
        hovermode="x unified",
        hoverlabel=dict(font_size=16),
        sliders=sliders["V"],
        showlegend=False,
    )

    fig = go.Figure(dict(data=data, layout=layout))
    fig.set_subplots(
        1,
        2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Prediction", "Ground Truth"),
        horizontal_spacing=0.1,
    )
    camera = dict(eye=eye)

    fig.layout.scene1.camera = camera
    fig.layout.scene2.camera = camera
    fig.update_layout(title=metric)
    fig.update_scenes(aspectmode="data",xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    fig.show()
    return fig


def filter_points(df_pred, df_y, value=0.0000):
    """
    - Remove all of the points along the boundary
    """

    df_y_filt = df_y[
        (abs(df_y["velo-magnitude"]) >= value) & (abs(df_y["Pressure"]) >= value)
    ]
    filter_indices = df_y_filt.index.tolist()
    df_pred_filt = df_pred.loc[filter_indices]

    # df_pred_filt = df_pred[(abs(df_pred['X-Velo']) > 0.0001) & (abs(df_pred['Y-Velo']) > 0.0001) &
    #     (abs(df_pred['Z-Velo']) > 0.0001) & (abs(df_pred['Pressure']) > 0.0001)]
    # df_delta_filt = df_delta[(abs(df_delta['X-Velo']) > 0.0001) & (abs(df_delta['Y-Velo']) > 0.0001) &
    #     (abs(df_delta['Z-Velo']) > 0.0001) & (abs(df_delta['Pressure']) > 0.0001)]

    return df_pred_filt, df_y_filt
