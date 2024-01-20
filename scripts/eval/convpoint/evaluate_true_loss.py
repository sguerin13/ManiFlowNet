import copy
import os
import sys

import neural.evaluation.eval_utils as eval_utils
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import tqdm
from torchmetrics import WeightedMeanAbsolutePercentageError

from scripts.helpers import load_config

if __name__ == "__main__":
    config = load_config(
        os.path.join(
             "scripts", "eval", "convpoint", "evaluate_true_loss.json"
        )
    )

    parent_folder = config.filePaths.experimentParentFolder
    scaler_path = config.filePaths.pathToScalers

    fpaths = config.filePaths
    network = config.networkParams

    # this is hardcoded for now with my project folders, but we can make it more general later
    folder_list = [
        "single_branch_std_ndim",
        "single_branch_std_VP_all_inputs",
        "single_branch_std",
        "single_branch_ndim_all_fps",
        "multi_branch_std_v2",
        "multi_branch_ndim",
        "multi_branch_ndim_all_fps",
        "multi_branch_std_VP_all_inputs",
    ]

    config_list = [
        {"scaler": "ndim", "non_dimensionalize": True, "VP_all": False},
        {"scaler": "std", "non_dimensionalize": False, "VP_all": True},
        {"scaler": "std", "non_dimensionalize": False, "VP_all": False},
        {"scaler": "ndim_all_fps", "non_dimensionalize": True, "VP_all": False},
        {"scaler": "std", "non_dimensionalize": False, "VP_all": False},
        {"scaler": "ndim", "non_dimensionalize": True, "VP_all": False},
        {"scaler": "ndim_all_fps", "non_dimensionalize": True, "VP_all": False},
        {"scaler": "std", "non_dimensionalize": False, "VP_all": True},
    ]

    key_value_pairs = list(
        zip(
            folder_list,
            [{"config": config_list[i], "values": []} for i in range(len(folder_list))],
        )
    )

    dictionary = dict(key_value_pairs)
    for j in tqdm.tqdm(range(1)):
        for key in tqdm.tqdm(dictionary.keys()):
            torch.cuda.empty_cache()
            target_folder = key
            config = dictionary[key]["config"]

            checkpoint_path = os.path.join(
                parent_folder, target_folder, "val_best.ckpt"
            )
            print(checkpoint_path)

            scaler_type = config["scaler"]
            non_dimensionalize = config["non_dimensionalize"]
            vp_all_inputs = config["VP_all"]
            scaler_dict = eval_utils.get_scalers(os.path.join(scaler_path, scaler_type))

            bbox_scaler = scaler_dict["bbox"]
            velo_scaler = scaler_dict["velo"]
            pressure_scaler = scaler_dict["pressure"]
            fparams_scaler = scaler_dict["fparams"]
            context_values_included = scaler_dict["context"]

            sample_qty = network.sampleQty
            rotate = network.rotate
            surface_nodes = network.surfaceNodes
            add_normals = network.addNormals

            val_loader = eval_utils.get_CP_val_dataloader(
                val_path=fpaths.valDataFPath,
                bbox_scaler=bbox_scaler,
                velo_scaler=velo_scaler,
                pressure_scaler=pressure_scaler,
                fparams_scaler=fparams_scaler,
                context_values_included=context_values_included,
                sample_qty=sample_qty,
                rotate=rotate,
                surface_nodes=surface_nodes,
                add_normals=add_normals,
                VP_all_inputs=vp_all_inputs,
                non_dimensionalize=non_dimensionalize,
            )

            net = eval_utils.get_CP_net(checkpoint_path)

            # for non-dimensionalization analysis
            vp_list = eval_utils.get_normalization_values(
                os.path.join(fpaths.valDataFPath)
            )

            wmape_list = []
            for i, batch in tqdm.tqdm(enumerate(val_loader)):
                x_pts, x_feats, y_feats = batch
                out = net.forward(x_feats.to("cuda:0"), x_pts.to("cuda:0"))
                out_numpy = out.detach().cpu().numpy().reshape(-1, 4)
                target_numpy = y_feats.detach().cpu().numpy().reshape(-1, 4)

                (v_in, P_out) = vp_list[i]
                out, target = eval_utils.inverse_scale_numpy_array(
                    scaler_dict,
                    out_numpy,
                    target_numpy,
                    non_dimensionalize,
                    v_in,
                    P_out,
                )
                loss = WeightedMeanAbsolutePercentageError()
                wmape = loss(torch.tensor(out_numpy), torch.tensor(target_numpy))
                wmape_list.append(wmape.detach().numpy())

            dictionary[key]["values"].append(np.mean(wmape_list))

    # create the dataframe
    column_values = list(dictionary.keys())

    output_list = []
    for j in range(1):
        temp_list = []
        for key in dictionary.keys():
            temp_list.append(dictionary[key]["values"][j])
        output_list.append(copy.deepcopy(temp_list))

    df = pd.DataFrame(output_list, columns=column_values)
    df.to_csv(
        os.path.join(
            "outputs",
            "evaluation",
            "evaluation_true_CP_val_loss_metrics_Linux.csv",
        )
    )
