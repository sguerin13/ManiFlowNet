import copy
import os

import MinkowskiEngine as ME
import numpy as np
import pandas as pd
import torch
import tqdm
from torchmetrics import WeightedMeanAbsolutePercentageError
from scripts.helpers import load_config

import neural.evaluation.eval_utils as eval_utils


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
        "d_8_single_branch_ndim",
        "d_8_single_branch_std_vp_all",
        "d_8_single_branch_std",
        "d_8_single_branch_ndim_all_fps",
        "me_h_params_d_8_multi_branch_std",
        "me_h_params_d_8_multi_branch_ndim",
        "me_h_params_d_8_multi_branch_ndim_all_fps",
        "me_h_params_d_8_multi_branch_std_VP_all",
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
        zip(folder_list, [{"config": config_list[i], "values": []} for i in range(8)])
    )

    dictionary = dict(key_value_pairs)
    for j in tqdm.tqdm(range(10)):
        for key in tqdm.tqdm(dictionary.keys()):
            target_folder = key
            config = dictionary[key]["config"]

            checkpoint_path = os.path.join(
                parent_folder, target_folder, "val_best.ckpt"
            )

            scaler_type = config["scaler"]
            non_dimensionalize = config["non_dimensionalize"]
            vp_all_inputs = config["VP_all"]

            scaler_dict = eval_utils.get_scalers(os.path.join(scaler_path, scaler_type))

            bbox_scaler = scaler_dict["bbox"]
            velo_scaler = scaler_dict["velo"]
            pressure_scaler = scaler_dict["pressure"]
            fparams_scaler = scaler_dict["fparams"]
            context_values_included = scaler_dict["context"]

            n_voxels = 256
            sample_qty = None
            rotate = network.rotate
            surface_nodes = network.surfaceNodes
            add_normals = network.addNormals

            val_loader = eval_utils.get_ME_val_dataloader(
                val_path=fpaths.valDataFPath,
                n_voxels=n_voxels,
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

            net = eval_utils.get_ME_net(checkpoint_path)

            # for non-dimensionalization analysis
            vp_list = eval_utils.get_normalization_values(
                os.path.join(fpaths.valDataFPath)
            )

            wmape_list = []
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
                # x_F = x_feats
                out_F = out.F.detach().cpu().numpy()
                out_C = out.C.detach().cpu().numpy()
                target_F = sparse_output.F.cpu().numpy()

                (v_in, P_out) = vp_list[i]
                out, target = eval_utils.inverse_scale_numpy_array(
                    scaler_dict, out_F, target_F, non_dimensionalize, v_in, P_out
                )
                loss = WeightedMeanAbsolutePercentageError()
                wmape = loss(torch.tensor(out_F), torch.tensor(target_F))
                wmape_list.append(wmape.detach().numpy())

            dictionary[key]["values"].append(np.mean(wmape_list))

    # create the dataframe
    column_values = list(dictionary.keys())

    output_list = []
    for j in range(10):
        temp_list = []
        for key in dictionary.keys():
            temp_list.append(dictionary[key]["values"][j])
        output_list.append(copy.deepcopy(temp_list))

    df = pd.DataFrame(output_list, columns=column_values)
    df.to_csv(
        os.path.join(
            
            "outputs",
            "evaluation",
            "evaluation_true_ME_val_loss_metrics.csv",
        )
    )
