
import torch
import pandas as pd
import numpy as np
import data_utils as DU
import numpy as np
import pickle
import data.wrangling.prep.helpers as DPH


def create_pyg_sample(pnt_cld_fpath, msh_fpath,normals_path, velo_scaler, pressure_scaler, bbox_scaler, scaler_range=1,
                         fluid_prop_scaler=None, fluid_prop_path=None,
                         add_context_to_input=False, sample_qty=None, rotate = True, 
                         surface_nodes = False, add_normals = True, separate_position = True):
    
    pnt_cld = DU.read_ply(pnt_cld_fpath)
    edges_T =  get_edge_array(msh_fpath)


    if add_normals:
        pnt_cld_normals = pickle.load(open(normals_path, "rb" ))
        pnt_cld =  pd.concat([pnt_cld, pnt_cld_normals[['norm-x','norm-y','norm-z']]], axis=1)
    
    if surface_nodes:
        pnt_cld = DPH.get_surface_nodes(pnt_cld)

    if rotate:
        pnt_cld = DPH.random_rotation(pnt_cld,normals_included = add_normals)

    pnt_cld = DU.normalize_pnt_cld_velo_features_w_velo_scaler(
        pnt_cld, velo_scaler)
    pnt_cld = DU.normalize_pnt_cld_P_features_w_P_scaler(
        pnt_cld, pressure_scaler)

    pnt_cld = pnt_cld.astype('float32')  # convert to 32 bit instead of 64
    pnt_cld = DU.convert_zone_id_to_zone_names(pnt_cld)
    pnt_cld = DU.dummy_encode_zones(pnt_cld)
    
    # TODO: Figure out how I want to handle sampling
    # if sample_qty is not None:

    #     if (sample_qty < len(pnt_cld)):
    #         sampled_pnt_cld = pnt_cld.copy().sample(sample_qty)
    #     else:
    #         sampled_pnt_cld = pnt_cld.copy()

    #     x, y = DPH.create_xy_pairs(sampled_pnt_cld, normals_included=add_normals, surface_only=surface_nodes)
    # else:

    x, y = DPH.create_xy_pairs(pnt_cld, normals_included=add_normals,surface_only=surface_nodes)

    (x_pts, x_feats), (y_pts, y_feats) = DPH.create_input_points(
        x,normals_included=add_normals, surface_only=surface_nodes), DPH.create_output_points(x, y, surface_only=surface_nodes)
    
    x_pts_norm, max_dimension = DPH.normalize_coordinates_to_0_1_BB(
        x_pts)
    relative_scale = DU.get_relative_bbox_scale(max_dimension, bbox_scaler)

    # scale factor will be 0-1, 0-10, or 0-100 depending on normalization min max range
    scale_factor = scaler_range*relative_scale

    if add_context_to_input:
        x_feats = DPH.add_context_to_feats( 
            x_feats, fluid_prop_path, fluid_prop_scaler, scale_factor)
    
    x_pts_T = torch.from_numpy(x_pts_norm.to_numpy()).float()
    x_feats_T = torch.from_numpy(x_feats.to_numpy()).float()
    y_feats_T = torch.from_numpy(y_feats.to_numpy()).float()
    return x_pts_T, x_feats_T, y_feats_T, edges_T


def get_edge_array(msh_fpath):
    msh_dict = DU.load_msh_data(msh_fpath)
    edge_array = []
    for ind in msh_dict['points']['point_data'].keys():
        if ind == 1:
            edge_array = np.array([[ind,i] for i in msh_dict['points']['point_data'][ind]['connected_nodes']]).T
        else:
            edge_array = np.hstack((edge_array,np.array(
                [[ind,i] for i in msh_dict['points']['point_data'][ind]['connected_nodes']]
                ).T))
    
    return torch.tensor(edge_array,dtype=torch.long)