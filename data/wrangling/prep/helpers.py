'''

Common data prep shared across the other specialized data prep routines

'''
import numpy as np
import data.wrangling.data_utils as DU
from data.wrangling.data_utils import cosd, sind
import torch
import copy
import pandas as pd

def get_normalized_bounding_box(fpath, bbox_scaler):
    pnt_cld = DU.read_ply(fpath)
    xyz = DU.get_bounding_box_dims(pnt_cld)
    xyz_array_scaled = DU.normalize_bounding_box(
        xyz, bbox_scaler
    )
    return xyz_array_scaled

def normalize_coordinates_to_0_1_BB(x_pts):
    bb = DU.get_bounding_box(x_pts)
    x_bb_at_origin, new_bb = move_bb_to_origin(
        x_pts, bb)
    
    max_dimension = max(new_bb[1])
    x_bb_norm_at_origin = DU.normalize_bounding_box(
        x_bb_at_origin, max_dimension)
    return x_bb_norm_at_origin, max_dimension

def move_bb_to_origin(pts, bbox):
    # min_x, min_y, min_z
    pts_copy = pts.copy()
    new_bb = [(0.0,0.0,0.0),
              (bbox[1][0] - bbox[0][0],bbox[1][1] - bbox[0][1],bbox[1][2] - bbox[0][2])]
    x_shifted = [i-bbox[0][0] for i in pts['x'].tolist()]
    y_shifted = [i-bbox[0][1] for i in pts['y'].tolist()]
    z_shifted = [i-bbox[0][2] for i in pts['z'].tolist()]
    pts_copy.x = x_shifted
    pts_copy.y = y_shifted
    pts_copy.z = z_shifted
    return pts_copy, new_bb

def random_rotation(pnt_cld, normals_included=False):
    R_X_angle = np.random.uniform(0, 360)
    R_Y_angle = np.random.uniform(0, 360)
    R_Z_angle = np.random.uniform(0, 360)

    Rx = np.array([[1,               0,                0],
                   [0, cosd(R_X_angle), -sind(R_X_angle)],
                   [0, sind(R_X_angle),  cosd(R_X_angle)]])

    Ry = np.array([[cosd(R_Y_angle), 0,  sind(R_Y_angle)],
                   [0, 1,                0],
                   [-sind(R_Y_angle), 0,  cosd(R_Y_angle)]])

    Rz = np.array([[cosd(R_Z_angle), -sind(R_Z_angle), 0],
                   [sind(R_Z_angle),  cosd(R_Z_angle), 0],
                   [0,                0, 1]])

    R = Rx@Ry@Rz

    # rotate data
    xyz = pnt_cld[['x', 'y', 'z']].to_numpy()
    xyz = xyz.T
    xyz_r = R@xyz
    pnt_cld[['x', 'y', 'z']] = xyz_r.T

    xyz_v = pnt_cld[['x-velocity', 'y-velocity', 'z-velocity']].to_numpy()
    xyz_v = xyz_v.T
    xyz_v_r = R@xyz_v
    pnt_cld[['x-velocity', 'y-velocity', 'z-velocity']] = xyz_v_r.T

    if normals_included:
        xyz_n = pnt_cld[['norm-x','norm-y','norm-z']].to_numpy()
        xyz_n = xyz_n.T
        xyz_n_r = R@xyz_n
        pnt_cld[['norm-x','norm-y','norm-z']] = xyz_n_r.T

    return pnt_cld

def create_xy_pairs(pnt_cld, normals_included = False, surface_only = False, VP_all_inputs = False, non_dim_inputs = False):
    ''' 
    split data and drop the node number

    '''

    x = pnt_cld.copy()
    y = pnt_cld.copy()

    # If we are going to provide velo and pressure as global context or use non-dimensional target qtys
    # We do not need to include the physical values in the input 
    if VP_all_inputs or non_dim_inputs:
        x.drop(columns = ['nodenumber','x-velocity', 'y-velocity', 'z-velocity','pressure'], inplace=True)

    else:
        x.loc[x['zone_inlet'] == 0, ['x-velocity', 'y-velocity', 'z-velocity']] = 0 # if not inlet, set equal to zero
        x.loc[x['zone_outlet'] == 0, ['pressure']] = 0                              # if not outlet, set pressure = 0
        x.drop(columns=['nodenumber'], inplace=True)
    
    if surface_only:
        y.drop(columns=['nodenumber', 'zone_inlet', 'zone_outlet', 'zone_wall'], inplace=True) # already removed fluid nodes
    else:
        y.drop(columns=['nodenumber', 'zone_inlet', 'zone_outlet',
            'zone_fluid', 'zone_wall'], inplace=True)
    
    if "mesh_node_number" in x.columns: # account for v2 ply file
        x.drop(columns=['mesh_node_number'])
        y.drop(columns=['mesh_node_number'],inplace=True)

    if normals_included:
        y.drop(columns=["norm-x","norm-y","norm-z"], inplace=True)
    
    return x, y

def convert_to_tensor(pnts):
    tensor = torch.tensor(pnts.to_numpy())
    return tensor

def create_input_points(pnts, normals_included=False, surface_only=False, VP_all_inputs = False, non_dim_inputs = False):
    points = pnts[['x', 'y', 'z']].copy()

    if VP_all_inputs or non_dim_inputs:
        features_to_include = ['zone_fluid', 'zone_inlet',
                           'zone_outlet', 'zone_wall','norm-x','norm-y','norm-z']
    else:     
        features_to_include = ['pressure', 'x-velocity', 'y-velocity',
                            'z-velocity', 'zone_fluid', 'zone_inlet',
                            'zone_outlet', 'zone_wall','norm-x','norm-y','norm-z']
    
    if not normals_included:
        features_to_include.remove('norm-x')
        features_to_include.remove('norm-y')
        features_to_include.remove('norm-z')
    
    if surface_only:
        features_to_include.remove('zone_fluid')
    
    features = \
            pnts[features_to_include].copy()
    return (points, features)

def create_output_points(x_pts, y_pts, surface_only=False):
    '''
    If we are only looking at the surface of the manifold, then we are only concerned with the properties at the outlet
    '''
    if surface_only:
        include_in_loss_list = []
        for i, row in x_pts.iterrows():
            if 1 in row[['zone_inlet','zone_outlet']].tolist():
                include_in_loss_list.extend([1])
            else:
                include_in_loss_list.extend([0])
        
        points = y_pts[['x', 'y', 'z']]
        features = \
            y_pts[['pressure', 'x-velocity', 'y-velocity', 'z-velocity']]
        features.loc[:,'include_in_loss'] = copy.deepcopy(include_in_loss_list)
    else:
        points = y_pts[['x', 'y', 'z']]
        features = \
        y_pts[['pressure', 'x-velocity', 'y-velocity', 'z-velocity']]
        
    return (points, features)

def add_context_to_feats_from_values_list(feats_df, fluid_prop_path, context_values_included,
                                         fluid_prop_scaler, relative_scale, velo_scaler,
                                         pressure_scaler, include_VP = False):
    '''
    
    props included is a list of the values that were included in the scaler

    
    '''
    fluid_params_dict = DU.load_fluid_prop_data_as_dict(fluid_prop_path)
    values = context_values_included
    fluid_props_list = [fluid_params_dict[i] for i in values]
    
    fluid_params = DU.normalize_fluid_prop_data_w_scaler(
        fluid_props_list, fluid_prop_scaler)
    
    context_list = fluid_params.tolist()
    if include_VP:
        if velo_scaler and pressure_scaler:
            V = fluid_params_dict['V']
            P = fluid_params_dict['P']
            V_norm = DU.normalize_single_value_w_velo_scaler(np.array([[V,V,V]]),velo_scaler).tolist()[0][0] # hack since velo scaler expects x,y,z - we'll only use 1
            P_norm = DU.normalize_single_value_w_P_scaler(np.array([[P]]), pressure_scaler).tolist()[0][0]
            values.extend(["V","P"])
            context_list.extend(([V_norm,P_norm]))

    context_list.append(relative_scale.tolist()[0][0])
    values.append("scale")
    context_list = [context_list] * len(feats_df.index)
    feats_df[values] = context_list
    return feats_df

@DeprecationWarning
def add_context_to_feats(feats_df, fluid_prop_path, fluid_prop_scaler, relative_scale):
    fluid_params = \
        DU.load_fluid_props_Eps_Visc_Rho(fluid_prop_path)
    fluid_params = DU.normalize_fluid_prop_data_w_scaler(
        fluid_params, fluid_prop_scaler)
    context_list = fluid_params.tolist()
    context_list.append(relative_scale.tolist()[0][0])
    context_list = [context_list] * len(feats_df.index)
    feats_df[["eps", "visc", "rho", "scale"]] = context_list
    return feats_df

@DeprecationWarning
def add_context_to_feats_V_P(feats_df, fluid_prop_path, fluid_prop_scaler, velo_scaler, pressure_scaler, relative_scale):
    [Eps, Visc, Rho, V, P] = \
        DU.load_fluid_props_Eps_Visc_Rho_V_P(fluid_prop_path)
    fluid_params = DU.normalize_fluid_prop_data_w_scaler(
        [Eps, Visc, Rho], fluid_prop_scaler)
    
    V_norm = DU.normalize_single_value_w_velo_scaler(np.array([[V,V,V]]),velo_scaler).tolist()[0][0] # hack since velo scaler expects x,y,z - we'll only use 1
    P_norm = DU.normalize_single_value_w_P_scaler(np.array([[P]]), pressure_scaler).tolist()[0][0]
    context_list = fluid_params.tolist()
    context_list.append(relative_scale.tolist()[0][0])
    context_list.extend(([V_norm,P_norm]))
    context_list = [context_list] * len(feats_df.index)
    feats_df[["eps", "visc", "rho", "scale","V","P"]] = context_list
    return feats_df

def add_context_to_feats_non_dim(feats_df, fluid_prop_path, fluid_prop_scaler, velo_scaler, pressure_scaler, relative_scale):
    [Eps, Visc, Rho, V, P] = \
        DU.load_fluid_props_Eps_Visc_Rho_V_P(fluid_prop_path)
    fluid_params = DU.normalize_fluid_prop_data_w_scaler(
        [Eps, Visc, Rho], fluid_prop_scaler)
    
    V_norm = DU.normalize_single_value_w_velo_scaler(np.array([[V,V,V]]),velo_scaler).tolist()[0][0] # hack since velo scaler expects x,y,z - we'll only use 1
    P_norm = DU.normalize_single_value_w_P_scaler(np.array([[P]]), pressure_scaler).tolist()[0][0]
    context_list = fluid_params.tolist()
    context_list.append(relative_scale.tolist()[0][0])
    context_list.extend(([V_norm,P_norm]))
    context_list = [context_list] * len(feats_df.index)
    feats_df[["eps", "visc", "rho", "scale","V","P"]] = context_list
    return feats_df

def get_surface_nodes(pnt_cld):
    return pnt_cld[pnt_cld['zone'].isin([5,6,7])]