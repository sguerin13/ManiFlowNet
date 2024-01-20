import os
import warnings

import torch

import data.wrangling.prep.gnn as GDP

warnings.filterwarnings("ignore")


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        bbox_scaler=None,
        velo_scaler=None,
        pressure_scaler=None,
        fparams_scaler=None,
        transform=None,
        scaler_range=1,
        sample_qty=None,
        rotate=True,
        surface_nodes=False,
        add_normals=True,
        separate_position=True,
    ):
        super(Dataset, self).__init__()
        self.root = root
        sim_folders = []
        for folder in os.listdir(self.root):
            # simple manifold, multiple_manifold, etc...
            sim_type_folder = os.path.join(self.root, folder)

            for sim_folder in os.listdir(sim_type_folder):
                sim_folder_path = os.path.join(sim_type_folder, sim_folder)
                sim_folders.append(sim_folder_path)

        self.sample_qty = sample_qty
        self.sim_folders = sim_folders
        self.transform = transform
        self.velo_scaler = velo_scaler
        self.pressure_scaler = pressure_scaler
        self.fparams_scaler = fparams_scaler
        self.bbox_scaler = bbox_scaler
        self.scaler_range = scaler_range
        self.rotate = rotate
        self.surface_nodes = surface_nodes
        self.add_normals = add_normals
        self.separate_position = separate_position

    def __len__(self):
        return len(self.sim_folders)

    def __getitem__(self, idx):
        ply_path = os.path.join(self.sim_folders[idx], "pnt_cld.ply")
        msh_path = os.path.join(self.sim_folders[idx], "msh_dict.pkl")
        normals_path = os.path.join(self.sim_folders[idx], "normals_df.pkl")
        fluid_prop_path = os.path.join(self.sim_folders[idx], "fluid_props.pkl")
        x_coords, x_feats, y_feats, edges = GDP.create_pyg_sample(
            ply_path,
            msh_path,
            normals_path,
            velo_scaler=self.velo_scaler,
            pressure_scaler=self.pressure_scaler,
            bbox_scaler=self.bbox_scaler,
            fluid_prop_scaler=self.fparams_scaler,
            fluid_prop_path=fluid_prop_path,
            scaler_range=int(self.scaler_range),
            add_context_to_input=True,
            sample_qty=self.sample_qty,
            rotate=self.rotate,
            surface_nodes=self.surface_nodes,
            add_normals=self.add_normals,
            separate_position=True,  # TODO: see if we want to include this
        )

        # TODO: add pytorch geometric piece here
        return x_coords, x_feats, y_feats, edges
