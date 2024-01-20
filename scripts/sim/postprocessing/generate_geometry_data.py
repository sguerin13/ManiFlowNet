from data.simulation.data_processing import SimulationDataProcessing as SDP
import os
from tqdm import tqdm
from joblib import Parallel, delayed

import data.wrangling.data_utils as DU
import traceback

from scripts.helpers import load_config

if __name__ == "__main__":
    config = load_config(os.path.join("scripts","sim","postprocessing","generate_geometry_data.json"))
    root_path = config.simDataPath
    sim_path_list = []

    for geo_group in os.listdir(root_path):
        geo_group_path = os.path.join(root_path,geo_group)
        for sim_no in os.listdir(geo_group_path):
            sim_path_list.extend([os.path.join(geo_group_path,sim_no)])

    def generate_geo_data(sim_path):
        print(sim_path)
        try:
            cfd_mesh_file = "CFD_mesh.msh"
            solution_file = "sol_file.txt"
            log_file = "log.txt"
            mesh_dict_file = "msh_dict.pkl"
            pnt_cld_csv = "pnt_cld.csv"
            pnt_cld_file = "pnt_cld.ply"
            bc_file = "bc.txt"
            fluid_prop_file = "fluid_props.pkl"

            if os.path.exists(sim_path.joinpath(mesh_dict_file)):
                    # already processed file
                    return

            sim_data = SDP(
                sim_path=sim_path,
                cfd_mesh_file=cfd_mesh_file,
                mesh_dict_file=mesh_dict_file,
                solution_file=solution_file,
                point_cloud_csv=pnt_cld_csv,
                point_cloud_file=pnt_cld_file,
                bc_file=bc_file,
                log_file=log_file,
                fluid_prop_file=fluid_prop_file,
            )
            sim_data.create_data_structures()
            # TODO: integrate these into the class
            DU.reorder_pnt_cld(sim_path)
            DU.calculate_normals(sim_path) 
        # except:
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(sim_path)

    Parallel(n_jobs=1)(delayed(generate_geo_data)(i) for i in tqdm(sim_path_list))
