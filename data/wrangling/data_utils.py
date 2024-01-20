"""

Data loading and preparation functionality for prediction of 
open channel fluid flows using Deep Learning.

Functions cover:
    - Data Loading from compressed files
    - Data Parsing into X,Y pairs
    - Normalization
    - DataLoader Creation
"""

import pickle
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler as STD, MinMaxScaler as MinMax
from pickle import dump, load

from pyntcloud import PyntCloud as ply
import open3d as o3d
import os
import pandas as pd
import random
import shutil
import math
from tqdm import tqdm
import copy
import networkx as nx
import matplotlib.pyplot as plt

import data.simulation.data_processing as SDP

############### Fluid Prop Vector ###############


def load_fluid_prop_data(path):
    """[Re, Eps, Visc, Rho, P, V, D]"""
    fp_dict = load(open(path, "rb"))
    fp_vec = []
    for key in fp_dict.keys():
        fp_vec.extend([fp_dict[key]])
    return fp_vec


def load_fluid_prop_data_as_dict(path):
    """[Re, Eps, Visc, Rho, P, V, D]"""
    fp_dict = load(open(path, "rb"))
    return fp_dict


def load_fluid_props_Eps_Visc_Rho(path):
    """[Eps, Visc, Rho]"""
    fp_dict = load(open(path, "rb"))
    fp_vec = [fp_dict["Eps"], fp_dict["Visc"], fp_dict["Rho"]]
    return fp_vec


def load_fluid_props_Eps_Visc_Rho_V_P(path):
    """[Eps, Visc, Rho,V,P]"""
    fp_dict = load(open(path, "rb"))
    fp_vec = [
        fp_dict["Eps"],
        fp_dict["Visc"],
        fp_dict["Rho"],
        fp_dict["V"],
        fp_dict["P"],
    ]
    return fp_vec


def partial_fit_fluid_prop_data(fp_vec, scaler):
    fp_arr = np.array([fp_vec])
    fp_scaled = fp_arr.copy()
    scaler.partial_fit(fp_scaled)
    del fp_scaled
    return scaler


def partial_fit_fluid_prop_data_with_selected_values(fp_dict, scaler, values):
    """
    fp: fluid property dictionary
    scaler: sklearn scaler
    values: the keys we want to include scaler
    """
    fp_list = [fp_dict[i] for i in values]
    fp_arr = np.array([fp_list])
    fp_scaled = fp_arr.copy()
    scaler.partial_fit(fp_scaled)
    del fp_scaled
    return scaler


def normalize_fluid_prop_data(fp_vec, type="MinMax"):
    fp_arr = np.array([fp_vec])
    fp_scaled = fp_arr.copy()
    if type == "std":
        scaler = STD()
    else:
        scaler = MinMax()
    fp_scaled = scaler.fit_transform(fp_scaled)
    return fp_scaled


def normalize_fluid_prop_data_w_scaler(fp_vec, scaler):
    fp_arr = np.array([fp_vec])
    fp_scaled = fp_arr.copy()
    fp_scaled = scaler.transform(fp_scaled)
    return fp_scaled[0]


#############################################

### Point Cloud Data ###


def read_ply(fpath):
    pnt_cld = ply.from_file(fpath)
    pnt_cld = pnt_cld.points
    return pnt_cld


def partial_fit_pnt_cld_data(pnt_cld, scaler, non_dim=False, fluid_prop_dict=None):
    pnt_cld_scaled = pnt_cld.copy()

    # if non-dim and fluid prop - we use P = delta_P/(rho*V^2), V = V/V_in
    if non_dim and fluid_prop_dict:
        pnt_cld_scaled = non_dimensionalize_quantities(pnt_cld_scaled, fluid_prop_dict)

    scaler.partial_fit(
        pnt_cld_scaled[["pressure", "x-velocity", "y-velocity", "z-velocity"]]
    )
    del pnt_cld_scaled
    return scaler


def normalize_pnt_cld_features(pnt_cld, type="MinMax"):
    if type == "std":
        scaler = STD()
    else:
        scaler = MinMax()
    pnt_cld_scaled = pnt_cld.copy()
    pnt_cld_scaled[
        ["pressure", "x-velocity", "y-velocity", "z-velocity"]
    ] = scaler.fit_transform(
        pnt_cld_scaled[["pressure", "x-velocity", "y-velocity", "z-velocity"]]
    )
    return pnt_cld_scaled


def non_dimensionalize_quantities(pnt_cld, fluid_prop_dict):
    P_in = fluid_prop_dict["P"]
    V_in = fluid_prop_dict["V"]
    rho = fluid_prop_dict["Rho"]
    pressure_div = rho * (V_in**2)
    pressure_ndim = [(i - P_in) / (pressure_div) for i in pnt_cld["pressure"].tolist()]
    xv_ndim = [i / V_in for i in pnt_cld["x-velocity"].tolist()]
    yv_ndim = [i / V_in for i in pnt_cld["y-velocity"].tolist()]
    zv_ndim = [i / V_in for i in pnt_cld["z-velocity"].tolist()]

    pnt_cld["pressure"] = pressure_ndim
    pnt_cld["x-velocity"] = xv_ndim
    pnt_cld["y-velocity"] = yv_ndim
    pnt_cld["z-velocity"] = zv_ndim
    return pnt_cld


def get_value_ratios(pnt_cld, fluid_prop_dict, n_dim=False):
    P_in = fluid_prop_dict["P"]
    V_in = fluid_prop_dict["V"]
    rho = fluid_prop_dict["Rho"]
    pressure_div = rho * (V_in**2)

    max_x = copy.deepcopy(max(abs(pnt_cld["x-velocity"])))
    max_y = copy.deepcopy(max(abs(pnt_cld["y-velocity"])))
    max_z = copy.deepcopy(max(abs(pnt_cld["z-velocity"])))
    if n_dim:
        max_p = (copy.deepcopy(max(pnt_cld["pressure"])) - P_in) / pressure_div
    else:
        max_p = copy.deepcopy(max(pnt_cld["pressure"])) / P_in

    return [max_x / V_in, max_y / V_in, max_z / V_in, max_p]


# updated feature scalers to put input on better scale

"""
Because the Simulations are set so that the geometry is aligned along the X-axis,
the velocity scalers will not be uniform along X,Y, and Z, in positive and negative directions
therefore we need to take the calculated scaler and set it so that the maximum and minimum values
are equal. Additionally, we want to set the velocity vectors to the range of [-1,1] so that zero is 
a true zero, and directionality of the velocity is maintained. 

Pressure, however, should be on [0,1] scale since it is a scaler value and not a vector. therefore the
point cloud features will need to be processed by two separate scalers.

"""


def create_uniform_pnt_cld_velo_scaler_from_feature_scaler(
    pnt_cld_scaler, min_max_range=(0, 1)
):
    """

    reads the features of the standard pnt_cld_scaler generated on the dataset
    and changes the range for x,y, and z velocities so that the scale is equivalent for
    positive and negative velocity vectors as well as having a true 0 == no velocity.

    """
    # [Presure, X Velocity, Y Velocity, Z Velocity]
    scaler_mins = list(pnt_cld_scaler.data_min_)
    scaler_maxs = list(pnt_cld_scaler.data_max_)

    largest_val = 0
    for i in range(1, 4):  # skipping pressure
        max_val = max(abs(scaler_mins[i]), abs(scaler_maxs[i]))
        if max_val > largest_val:
            largest_val = max_val

    # create new scaler with desired range
    syn_data = np.array(
        [
            [largest_val, largest_val, largest_val],
            [-largest_val, -largest_val, -largest_val],
        ]
    )
    scaler = MinMax(feature_range=min_max_range)
    scaler.fit(syn_data)
    return scaler

def normalize_pnt_cld_velo_features_w_velo_scaler(pnt_cld, velo_scaler):
    pnt_cld_scaled = pnt_cld.copy()
    pnt_cld_scaled[["x-velocity", "y-velocity", "z-velocity"]] = velo_scaler.transform(
        pnt_cld_scaled[["x-velocity", "y-velocity", "z-velocity"]]
    )
    return pnt_cld_scaled


def normalize_single_value_w_velo_scaler(value, velo_scaler):
    value2 = value.copy()
    return velo_scaler.transform(value2)


def create_pnt_cld_pressure_scaler_from_feat_scaler(pnt_cld_scaler, max_scale_value=1):
    # [Presure, X Velocity, Y Velocity, Z Velocity]
    p_max = pnt_cld_scaler.data_max_[0]

    # create new scaler with desired range
    syn_data = np.array([[0], [p_max]])
    scaler = MinMax(feature_range=(0, max_scale_value))
    scaler.fit(syn_data)
    return scaler

def normalize_pnt_cld_P_features_w_P_scaler(pnt_cld, P_scaler):
    pnt_cld_scaled = pnt_cld.copy()
    pnt_cld_scaled[["pressure"]] = P_scaler.transform(pnt_cld_scaled[["pressure"]])
    return pnt_cld_scaled


def normalize_single_value_w_P_scaler(value, P_scaler):
    value2 = value.copy()
    return P_scaler.transform(value2)


def normalize_pnt_cld_features_w_scaler(pnt_cld, feat_scaler):
    pnt_cld_scaled = pnt_cld.copy()
    pnt_cld_scaled[
        ["pressure", "x-velocity", "y-velocity", "z-velocity"]
    ] = feat_scaler.transform(
        pnt_cld_scaled[["pressure", "x-velocity", "y-velocity", "z-velocity"]]
    )
    return pnt_cld_scaled

def convert_zone_id_to_zone_names(pnt_cld):
    zone_list = []
    zones = pnt_cld["zone"].copy(deep=True)
    for zone_id in zones:
        if zone_id == 1:
            zone_list.append("fluid")
        elif zone_id == 5:
            zone_list.append("inlet")
        elif zone_id == 6:
            zone_list.append("outlet")
        elif zone_id == 7:
            zone_list.append("wall")

    pnt_cld.drop(columns=["zone"], inplace=True)
    pnt_cld["zone"] = zone_list
    return pnt_cld


def dummy_encode_zones(pnt_cld):
    pnt_cld = pd.get_dummies(pnt_cld, columns=["zone"])
    return pnt_cld


def calculate_normals(fpath):
    mesh_dict = load_msh_data(os.path.join(fpath, "msh_dict.pkl"))
    surface_mesh_dict = get_surface_nodes(mesh_dict)
    o3d_mesh, zero_index_node_mapping = generate_triangle_surface_mesh(
        surface_mesh_dict
    )
    norms_df = map_node_normals_to_point_normal_df(
        o3d_mesh, mesh_dict, zero_index_node_mapping
    )
    pickle.dump(norms_df, open(os.path.join(fpath, "normals_df.pkl"), "wb"))


def get_surface_nodes(mesh_dict):
    mesh_dict_copy = copy.deepcopy(mesh_dict)
    surface_nodes = [5, 6, 7]
    # for nodes that belong to multiple zones, consolidate:
    for point in mesh_dict_copy["points"]["point_data"]:
        mesh_point = mesh_dict_copy["points"]["point_data"][point]
        if len(mesh_point["zone_id"]) > 1:
            if 5 in mesh_point["zone_id"]:  # inlet
                mesh_point["zone_id"] = [5]

            elif 6 in mesh_point["zone_id"]:  # outlet
                mesh_point["zone_id"] = [6]

            elif 7 in mesh_point["zone_id"]:  # wall
                mesh_point["zone_id"] = [7]

    # update the connectivity to only include surface nodes
    for point in mesh_dict_copy["points"]["point_data"]:
        if mesh_dict_copy["points"]["point_data"][point]["zone_id"][0] in surface_nodes:
            connected_nodes_to_keep = []
            for con_node in mesh_dict_copy["points"]["point_data"][point][
                "connected_nodes"
            ]:
                if (
                    mesh_dict_copy["points"]["point_data"][con_node]["zone_id"][0]
                    in surface_nodes
                ):
                    connected_nodes_to_keep.append(con_node)
            mesh_dict_copy["points"]["point_data"][point][
                "connected_nodes"
            ] = connected_nodes_to_keep

    # remove non-surface nodes
    for point in list(mesh_dict_copy["points"]["point_data"].keys()):
        if (
            mesh_dict_copy["points"]["point_data"][point]["zone_id"][0]
            not in surface_nodes
        ):
            del mesh_dict_copy["points"]["point_data"][point]

    return mesh_dict_copy


def generate_triangle_surface_mesh(surface_mesh_dict):
    face_dict = {}
    # get faces from nodes:
    adj_list = SDP.build_adjacency_list_from_dict(surface_mesh_dict)
    G = nx.Graph(adj_list)
    isP, GP = nx.check_planarity(G)  # GP returns
    clockwise_adj_dict = GP.get_data()
    max_deg = 0

    for i in clockwise_adj_dict.keys():
        for j in clockwise_adj_dict[i]:
            face = GP.traverse_face(i, j)
            if tuple(sorted(face)) not in face_dict:
                face_dict[tuple(sorted(face))] = face
                if len(face) > max_deg:
                    max_deg = len(face)
    # V-E+F = 2
    assert (
        len(surface_mesh_dict["points"]["point_data"].keys())
        - len(G.edges)
        + len(face_dict.keys())
        == 2
    )

    face_dict = gen_triangles_from_polygons(face_dict)
    zero_index_node_mapping = create_node_map(surface_mesh_dict["points"]["point_data"])
    o3d_mesh = create_o3d_mesh(face_dict, surface_mesh_dict, zero_index_node_mapping)
    return o3d_mesh, zero_index_node_mapping



def gen_triangles_from_polygons(face_dict):
    for face in list(face_dict.keys()):
        if len(face_dict[face]) > 3:
            if len(face_dict[face]) > 4:
                raise Exception("Too many nodes on face")
            nodes = face_dict[face].copy()
            new_face_1 = [nodes[0], nodes[1], nodes[3]]
            new_face_2 = [nodes[1], nodes[2], nodes[3]]
            face_dict[tuple(sorted(new_face_1))] = new_face_1
            face_dict[tuple(sorted(new_face_2))] = new_face_2
            del face_dict[face]
    return face_dict


def create_o3d_mesh(face_dict, mesh_dict, zero_index_node_mapping_dict):
    """
    open3d mesh keeps the ordering of the vertices and faces
    """
    vertices = [
        list(mesh_dict["points"]["point_data"][i]["xyz"])
        for i in mesh_dict["points"]["point_data"].keys()
    ]
    vertices = o3d.utility.Vector3dVector(vertices)
    faces = [face_dict[face] for face in face_dict.keys()]
    zero_indexed_faces = []
    for face in faces:
        zero_indexed_face = [zero_index_node_mapping_dict["original"][i] for i in face]
        zero_indexed_faces.append(zero_indexed_face)

    faces = o3d.utility.Vector3iVector(zero_indexed_faces)
    mesh_np = o3d.geometry.TriangleMesh(vertices, faces)
    mesh_np.compute_vertex_normals()
    return mesh_np


def map_node_normals_to_point_normal_df(o3d_mesh, mesh_dict, zero_index_mapping):
    headers = ["nodenumber", "norm-x", "norm-y", "norm-z"]
    df_list = [[i, 0, 0, 0] for i in list(mesh_dict["points"]["point_data"].keys())]
    node_normals = list(np.asarray(o3d_mesh.vertex_normals))
    for i, node in enumerate(node_normals):
        target_node = zero_index_mapping["zero_sorted"][i]
        assert df_list[target_node - 1][0] == target_node, "wrong node"
        # xyz of the normals
        df_list[target_node - 1][1] = node[0]
        df_list[target_node - 1][2] = node[1]
        df_list[target_node - 1][3] = node[2]

    df = pd.DataFrame(df_list, columns=headers)
    return df


def create_node_map(point_dict):
    """
    To use open3d mesh from numpy we need to map the vertex index
    from their point cloud ordering to a 0 index sorted order so that vertex indices in the
    face tuple match their order in the list

    nodes are already in sorted order, so we just need to map slam it to 0 and remove any gaps in the indices
    """
    zero_index_mapping = {"original": {}, "zero_sorted": {}}
    key_list = list(point_dict.keys())
    for i, key in enumerate(key_list):
        zero_index_mapping["original"][key] = i
        zero_index_mapping["zero_sorted"][i] = key

    return zero_index_mapping

def reorder_pnt_cld(sim_path):
    pnt_cld = ply.from_file(os.path.join(sim_path, "pnt_cld.ply"))
    pnt_cld.points.sort_values(by=["mesh_node_number"], inplace=True)
    pnt_cld.points.reset_index(drop=True, inplace=True)
    pnt_cld.to_file(
        os.path.join(sim_path, "pnt_cld.ply"), as_text=True
    )

    ########## Bounding Box Data ##############

def get_bounding_scale(pnt_cld):
    xyz = pnt_cld[["x", "y", "z"]]
    x_scale = np.max(xyz["x"]) - np.min(xyz["x"])
    y_scale = np.max(xyz["y"]) - np.min(xyz["y"])
    z_scale = np.max(xyz["z"]) - np.min(xyz["z"])
    return [x_scale, y_scale, z_scale]


def get_bounding_box(pts):
    return [
        (np.min(pts["x"]), np.min(pts["y"]), np.min(pts["z"])),
        (np.max(pts["x"]), np.max(pts["y"]), np.max(pts["z"])),
    ]

def partial_fit_bounding_box_data(xyz, scaler):
    xyz_array = np.array([xyz])
    xyz_array_copy = xyz_array.copy()
    xyz_array_copy = xyz_array_copy.reshape(-1, 1)
    scaler.partial_fit(xyz_array_copy)
    return scaler

def get_relative_bbox_scale(max_bbox_dim_value, scaler):
    # figure out how large the bounding box max dimension is relative to the
    # max dim scale factor of the dataset
    np_max_bbox_dim_value = np.array([max_bbox_dim_value])
    np_max_bbox_dim_value_copy = np_max_bbox_dim_value.copy()
    np_max_bbox_dim_value_copy = np_max_bbox_dim_value_copy.reshape(-1, 1)
    max_bbox_dim_value_rel_scale = scaler.transform(np_max_bbox_dim_value_copy)
    return max_bbox_dim_value_rel_scale.reshape(1, -1)

def normalize_bounding_box(pts, max_dimension):
    # pts are a df with cols ['x','y','z']
    pts_copy = pts.copy()
    # add the value at the end to keep the number of voxels correct
    x_div = [i / (max_dimension + 0.0000001) for i in pts_copy["x"]]
    y_div = [i / (max_dimension + 0.0000001) for i in pts_copy["y"]]
    z_div = [i / (max_dimension + 0.0000001) for i in pts_copy["z"]]
    pts_copy["x"] = x_div
    pts_copy["y"] = y_div
    pts_copy["z"] = z_div
    return pts_copy


###########################################


################ MESH DATA ################
def load_msh_data(path):
    mesh_dict = load(open(path, "rb"))
    return mesh_dict


############################################


############ Boundary Conditions ###########


def load_bc_data(path):
    bc_tup = load(open(path, "rb"))
    vmag, P = bc_tup[0], bc_tup[1]
    return [vmag, P]


def normalize_bc_data(bc, type="std"):
    """normalize the boundary condition data"""
    bc_arr = np.array([[bc]])
    bc_scaled = bc_arr.copy()
    if type == "std":
        scaler = STD()
    else:
        scaler = MinMax()

    bc_scaled = scaler.fit_transform(bc_scaled)
    return bc_scaled


def normalize_bc_data_w_scaler(bc, scaler):
    bc_arr = np.array([[bc]])
    bc_scaled = bc_arr.copy()
    bc_scaled = scaler.transform(bc_scaled)
    return bc_scaled[0]


def partial_fit_bc_data(bc, scaler):
    bc_arr = np.array([[bc]])
    bc_scaled = bc_arr.copy()
    scaler.partial_fit(bc_scaled)
    del bc_scaled
    return scaler


###############################################


################### Train Val Splits ##################


def split_dataset(train_percentage, root_path):
    # create train and test folders within each geometry group
    if os.path.join(root_path, "train") in os.listdir(root_path):
        raise Exception("dataset already split into train and val sets")

    os.mkdir(os.path.join(root_path, "train"))
    os.mkdir(os.path.join(root_path, "val"))

    for geo_group in os.listdir(root_path):
        # skip these folders
        if (geo_group == "train") or (geo_group == "val"):
            continue

        geo_group_path = os.path.join(root_path, geo_group)
        for sim_no in os.listdir(geo_group_path):
            put_in_train = random.random() * 100 <= train_percentage
            if put_in_train:
                sim_path = os.path.join(geo_group_path, sim_no)
                train_sim_path = os.path.join(root_path, "train", geo_group, sim_no)
                shutil.move(sim_path, train_sim_path)

            else:
                sim_path = os.path.join(geo_group_path, sim_no)
                val_sim_path = os.path.join(root_path, "val", geo_group, sim_no)
                shutil.move(sim_path, val_sim_path)
    return


########################################################


################### Scalers ##################


def save_scaler(scaler, file_name):
    dump(scaler, open(file_name, "wb"))


def load_scaler(file_name):
    scaler = load(open(file_name, "rb"))
    return scaler


def generate_scalers(
    root_fpath,
    fluid_prop_values,
    bbox_scaler_fpath=None,
    fluid_props_scaler_fpath=None,
    velo_scaler_path=None,
    pressure_scaler_path=None,
    fluid_props_included_path=None,
    mode="test",
    values_path=None,
    non_dim=False,
):
    sim_folders = []
    for folder in os.listdir(root_fpath):
        # simple manifold, multiple_manifold, etc...
        sim_type_folder = os.path.join(root_fpath, folder)
        for sim_folder in os.listdir(sim_type_folder):
            sim_folder_path = os.path.join(sim_type_folder, sim_folder)
            sim_folders.append(sim_folder_path)

    pnt_cld_scaler = MinMax()
    fluid_props_scaler = MinMax(feature_range=(0, 1))
    # We don't adjust bbox scaler range because scale factor is determining during dataloading
    bbox_scaler = MinMax()

    xv = []
    yv = []
    zv = []
    del_p = []
    for i, folder in enumerate(tqdm(sim_folders)):
        try:
            ply_path = os.path.join(sim_folders[i], "pnt_cld.ply")
            fluid_props_path = os.path.join(sim_folders[i], "fluid_props.pkl")

            pnt_cld = read_ply(ply_path)
            fluid_prop_dict = load_fluid_prop_data_as_dict(fluid_props_path)
            bbox = get_bounding_scale(pnt_cld)

            """
            If we are non-dimensionalizing the input, we don't need to worry about non-dimensionalizing the fluid props, currently...
            because we are only going to take the Reynolds number and the Epsilon, and the scale factor, the rest of the values - V_in, Rho, Visc, Pout
            don't matter because of the non-dimensionalization.
            """
            if non_dim:
                pnt_cld_scaler = partial_fit_pnt_cld_data(
                    pnt_cld, pnt_cld_scaler, non_dim, fluid_prop_dict=fluid_prop_dict
                )
                ratios = get_value_ratios(pnt_cld, fluid_prop_dict, n_dim=True)
                xv.append(ratios[0])
                yv.append(ratios[1])
                zv.append(ratios[2])
                del_p.append(ratios[3])

            else:
                pnt_cld_scaler = partial_fit_pnt_cld_data(pnt_cld, pnt_cld_scaler)
                ratios = get_value_ratios(pnt_cld, fluid_prop_dict)
                xv.append(ratios[0])
                yv.append(ratios[1])
                zv.append(ratios[2])
                del_p.append(ratios[3])

            fluid_props_scaler = partial_fit_fluid_prop_data_with_selected_values(
                fluid_prop_dict, fluid_props_scaler, fluid_prop_values
            )

            bbox_scaler = partial_fit_bounding_box_data(bbox, bbox_scaler)
        except Exception as e:
            print("Error in: ", folder, " Error : ", e)

    # set bbox scaler min = 0 and reset range
    bbox_scaler.data_min_ = 0
    bbox_scaler.data_range_ = bbox_scaler.data_max_

    # create the Velo and Pressure Scalers
    velo_scaler = create_uniform_pnt_cld_velo_scaler_from_feature_scaler(
        pnt_cld_scaler, min_max_range=(-1, 1)
    )

    pressure_scaler = create_pnt_cld_pressure_scaler_from_feat_scaler(pnt_cld_scaler)

    if mode == "test":
        return (
            pnt_cld_scaler,
            fluid_props_scaler,
            bbox_scaler,
            velo_scaler,
            pressure_scaler,
        )
    else:  # mode == save
        # save the scalers
        values_dict = {"xv": xv, "yv": yv, "zv": zv, "del_p": del_p}
        save_scaler(values_dict, values_path)
        save_scaler(fluid_props_scaler, fluid_props_scaler_fpath)
        save_scaler(bbox_scaler, bbox_scaler_fpath)
        save_scaler(velo_scaler, velo_scaler_path)
        save_scaler(pressure_scaler, pressure_scaler_path)
        save_scaler(fluid_prop_values, fluid_props_included_path)
        return


################################################


################### MISC Helpers #############################


def sind(x):
    return math.sin(np.deg2rad(x))


def cosd(x):
    return math.cos(np.deg2rad(x))
