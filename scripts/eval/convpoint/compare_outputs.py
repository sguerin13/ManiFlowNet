import os

import numpy as np

import neural.evaluation.eval_utils as eval_utils
from scripts.helpers import load_config

'''

make sure the config (scaler type, non-dimensionalize, VP_all_inputs) matches the model you are evaluating


'''
if __name__ == "__main__":
    config = load_config(os.path.join("scripts","eval","convpoint","compare_outputs.json"))
    
    model_folder = config.modelFolder
    checkpoint_path = os.path.join(model_folder, "val_best.ckpt")

    scaler_type = config.scalerType
    non_dimensionalize = config.nonDimensionalize
    vp_all_inputs = config.VPAllInputs

    fpaths = config.filePaths
    network = config.networkParams

    # grab the scalers
    scaler_dict = eval_utils.get_scalers(os.path.join(fpaths.pathToScalers, scaler_type))
    bbox_scaler = scaler_dict["bbox"]
    velo_scaler = scaler_dict["velo"]
    pressure_scaler = scaler_dict["pressure"]
    fparams_scaler = scaler_dict["fparams"]
    context_values_included = scaler_dict["context"]

    # set some of the network hyperparameters
    rotate = network.rotate  # network.rotate
    surface_nodes = network.surfaceNodes
    add_normals = network.addNormals
    sample_qty = network.sampleQty

    # sample to grab from the validation set
    sample_index = config.sampleIndex

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
                sample_index=sample_index
            )

    net = eval_utils.get_CP_net(checkpoint_path)

    # for non-dimensionalization analysis
    vp_list = eval_utils.get_normalization_values(os.path.join(fpaths.valDataFPath))

    # coords, output feats, target_feats
    output_features, output_coords, target_features = eval_utils.get_CP_eval_output_for_single_sample(
        val_loader, net
    )


    inv_out, inv_target = eval_utils.inverse_scale_numpy_array(
        scaler_dict,
        output_features,
        target_features,
        ndim=non_dimensionalize,
        V_in=vp_list[sample_index][0],
        P_out=vp_list[sample_index][1],
    )


    df_pred, df_y = eval_utils.convert_CP_feats_and_coords_to_df(
        inv_out,
        output_coords,
        inv_target,
        max_location_scale=1,
        surface_only=False,
    )
    df_delta = eval_utils.calculate_delta(df_pred, df_y)

    df_pred["velo-magnitude"] = np.linalg.norm(
        df_pred[["X-Velo", "Y-Velo", "Z-Velo"]].values, axis=1
    )
    df_y["velo-magnitude"] = np.linalg.norm(
        df_y[["X-Velo", "Y-Velo", "Z-Velo"]].values, axis=1
    )
    df_delta["velo-magnitude"] = np.linalg.norm(
        df_delta[["X-Velo", "Y-Velo", "Z-Velo"]].values, axis=1
    )

    v_scale_tuple, p_scale_tuple, scale_factor = eval_utils.calculate_plot_magnitude_scale(
        df_pred, df_y
    ) 

    dv_scale_tuple, dp_scale_tuple, dscale_factor = eval_utils.calculate_plot_magnitude_scale(
        df_delta
    )

    df_pred_filt, df_y_filt = eval_utils.filter_points(df_pred,df_y)

    eval_utils.plot_output_metric(df_pred_filt,df_y_filt,p_scale_tuple,scale_factor,"Pressure", n_slices = 40, eye=dict(x=2.5, y=0, z=0))
