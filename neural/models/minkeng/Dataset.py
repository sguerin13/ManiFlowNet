import os
import warnings

import MinkowskiEngine as ME
import torch

import data.wrangling.prep.minkeng as MEP

warnings.filterwarnings("ignore")


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        n_voxels,
        bbox_scaler=None,
        velo_scaler=None,
        pressure_scaler=None,
        fparams_scaler=None,
        context_values_included=None,
        sample_qty=None,
        rotate=True,
        surface_nodes=False,
        add_normals=True,
        VP_all_inputs=True,
        non_dimensionalize=False,
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

        self.n_voxels = n_voxels
        self.sample_qty = sample_qty
        self.sim_folders = sim_folders
        self.velo_scaler = velo_scaler
        self.pressure_scaler = pressure_scaler
        self.fparams_scaler = fparams_scaler
        self.bbox_scaler = bbox_scaler
        self.context_values_included = context_values_included
        self.rotate = rotate
        self.surface_nodes = surface_nodes
        self.add_normals = add_normals
        self.VP_all_inputs = VP_all_inputs
        self.non_dimensionalize = non_dimensionalize

    def __len__(self):
        return len(self.sim_folders)

    def __getitem__(self, idx):
        ply_path = os.path.join(self.sim_folders[idx], "pnt_cld.ply")
        normals_path = os.path.join(self.sim_folders[idx], "normals_df.pkl")
        fluid_prop_path = os.path.join(self.sim_folders[idx], "fluid_props.pkl")
        x_coords, x_feats, y_feats = MEP.create_sparse_sample(
            ply_path,
            normals_path,
            voxel_size=1 / self.n_voxels,
            velo_scaler=self.velo_scaler,
            pressure_scaler=self.pressure_scaler,
            bbox_scaler=self.bbox_scaler,
            fluid_prop_scaler=self.fparams_scaler,
            fluid_prop_path=fluid_prop_path,
            sample_qty=self.sample_qty,
            rotate=self.rotate,
            surface_nodes=self.surface_nodes,
            add_normals=self.add_normals,
            VP_all_inputs=self.VP_all_inputs,
            non_dimensionalize=self.non_dimensionalize,
            context_values_included=self.context_values_included,
        )
        return x_coords, x_feats, y_feats  # scale_factor


def custom_collate_fn(sample):
    coords, x_feats, y_feats = list(zip(*sample))
    batch_coords = ME.utils.batched_coordinates(coords)
    batch_x_feats = torch.cat(x_feats, 0)
    batch_y_feats = torch.cat(y_feats, 0)
    return batch_coords, batch_x_feats, batch_y_feats
