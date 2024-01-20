import data.wrangling.data_utils as DU
from scripts.helpers import load_config
import os

"""
Script to generate scalers from the training dataset

  - There are two types of scalers that can be generated:
    - std normalization of values between 0 & +-1
    - scalers based on non-dimensionalization of the data in a 0 to +-1 range

    
  - You control which types of scalers to generate based on the mode parameter
    in the json file

        
  - some finer points:

    - If you are using the standard normalization, you want to include
        - ["Re","Eps","Visc","Rho","D"]

    - If you are using the non-dimensionalization, you want to include
        - ["Re","Eps"]

        - or optionally ["Re","Eps","Visc","Rho","D"], but theoretically
          you shouldn't need to include these extra fluid params
          
          - If want to test both, create a different folder for each set of non-dimensionalized
            scalers

            - i.e. : .../ndim/...  or .../ndim_all_fps/... 


"""


if __name__ == "__main__":

    config = load_config(os.path.join("scripts", "sim", "generate_scalers.json"))

    mode = config.mode
    train_path = config.trainPath  # this was root
    scaler_folder_path = config.scalerSavePath
    velo_scaler_path = os.path.join(scaler_folder_path, "velo_scaler.pkl")
    pressure_scaler_path = os.path.join(scaler_folder_path, "P_scaler.pkl")
    fluid_prop_scaler_path = os.path.join(scaler_folder_path, "fluid_prop_scaler.pkl")
    bbox_scaler_path = os.path.join(scaler_folder_path, "bbox_scaler.pkl")
    props_included_path = os.path.join(scaler_folder_path, "props_included.pkl")
    props_to_include = config.propsToInclude

    ndim_values_path = os.path.join( "outputs", config.ndimScalerFileName)
    norm_values_path = os.path.join( "outputs", config.normScalerFileName)

    ndim_values_path = "ndim_scaler_ratios.pkl"
    reg_values_path = "norm_scaler_ratios.pkl"

    scaler_ranges_path = ndim_values_path if mode == "ndim" else reg_values_path

    DU.generate_scalers(
        root_fpath=train_path,
        fluid_prop_values=props_to_include,
        mode="save",
        fluid_props_scaler_fpath=fluid_prop_scaler_path,
        bbox_scaler_fpath=bbox_scaler_path,
        velo_scaler_path=velo_scaler_path,
        pressure_scaler_path=pressure_scaler_path,
        fluid_props_included_path=props_included_path,
        non_dim=mode == "ndim",
        values_path=scaler_ranges_path,
    )
