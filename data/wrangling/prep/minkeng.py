import copy
import pickle

import numpy as np
import pandas as pd
import torch

import data.wrangling.data_utils as DU
import data.wrangling.prep.helpers as DPH



def create_sparse_sample(
    fpath,
    normals_path,
    velo_scaler,
    pressure_scaler,
    bbox_scaler,
    voxel_size=0.0078,
    fluid_prop_scaler=None,
    fluid_prop_path=None,
    context_values_included=None,
    sample_qty=None,
    rotate=True,
    surface_nodes=False,
    add_normals=True,
    VP_all_inputs=True,
    non_dimensionalize=False,
):
    pnt_cld = DU.read_ply(fpath)
    fluid_prop_dict = DU.load_fluid_prop_data_as_dict(fluid_prop_path)
    if non_dimensionalize:
        pnt_cld = DU.non_dimensionalize_quantities(pnt_cld, fluid_prop_dict)

    if add_normals:
        pnt_cld_normals = pickle.load(open(normals_path, "rb"))
        pnt_cld = pd.concat(
            [pnt_cld, pnt_cld_normals[["norm-x", "norm-y", "norm-z"]]], axis=1
        )

    if surface_nodes:
        pnt_cld = DPH.get_surface_nodes(pnt_cld)

    if rotate:
        pnt_cld = DPH.random_rotation(pnt_cld, normals_included=add_normals)

    pnt_cld = DU.normalize_pnt_cld_velo_features_w_velo_scaler(pnt_cld, velo_scaler)
    pnt_cld = DU.normalize_pnt_cld_P_features_w_P_scaler(pnt_cld, pressure_scaler)

    pnt_cld = pnt_cld.astype("float32")  # convert to 32 bit instead of 64
    pnt_cld = DU.convert_zone_id_to_zone_names(pnt_cld)
    pnt_cld = DU.dummy_encode_zones(pnt_cld)

    # Create XY Pairs
    if sample_qty is not None:
        if sample_qty < len(pnt_cld):
            sampled_pnt_cld = pnt_cld.copy().sample(sample_qty)
        else:
            sampled_pnt_cld = pnt_cld.copy()
    else:
        sampled_pnt_cld = pnt_cld.copy()  # just to keep variable name the same

    x, y = DPH.create_xy_pairs(
        sampled_pnt_cld,
        normals_included=add_normals,
        surface_only=surface_nodes,
        VP_all_inputs=VP_all_inputs,
        non_dim_inputs=non_dimensionalize,
    )

    # Create Split into points and features
    (x_pts, x_feats) = DPH.create_input_points(
        x,
        normals_included=add_normals,
        surface_only=surface_nodes,
        VP_all_inputs=VP_all_inputs,
        non_dim_inputs=non_dimensionalize,
    )
    (_, y_feats) = DPH.create_output_points(x, y, surface_only=surface_nodes)

    # Normalize point coordinates
    x_pts_norm, max_dimension = DPH.normalize_coordinates_to_0_1_BB(x_pts)
    relative_scale = DU.get_relative_bbox_scale(max_dimension, bbox_scaler)

    # tack on the simulation context
    x_feats = DPH.add_context_to_feats_from_values_list(
        x_feats,
        fluid_prop_path,
        context_values_included=copy.deepcopy(context_values_included),
        fluid_prop_scaler=fluid_prop_scaler,
        relative_scale=relative_scale,
        velo_scaler=velo_scaler,
        pressure_scaler=pressure_scaler,
        include_VP=VP_all_inputs,
    )

    x_pts_T = torch.from_numpy(
        np.ascontiguousarray(x_pts_norm.div(voxel_size).to_numpy())
    ).float()
    x_feats_T = torch.from_numpy(
        np.ascontiguousarray(x_feats.to_numpy(), dtype=np.float32)
    ).float()
    y_feats_T = torch.from_numpy(
        np.ascontiguousarray(y_feats.to_numpy(), dtype=np.float32)
    ).float()

    return x_pts_T, x_feats_T, y_feats_T
