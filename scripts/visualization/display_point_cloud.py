import os

from data.visualization.vis_functions import display_pnt_cloud
from scripts.helpers import load_config

if __name__ == "__main__":
    config = load_config(
        os.path.join("scripts", "visualization", "display_pnt_cloud.json")
    )

    fpath = config.samplePntCld
    display_pnt_cloud(fpath,size=3,opacity=.6,n_points=4096) #TODO: add more config control