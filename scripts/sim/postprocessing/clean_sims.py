import json
import os
from types import SimpleNamespace

from scripts.helpers import load_config

if __name__ == "__main__":
    config = load_config(os.path.join("scripts", "visualizations", "config.json"))

    sim_dir = config.simDirectoryToClean
    cleaned_sim_dir = config.locationForCleanSims
    failed_sim_dir = config.locationForFailedSims

    if not os.path.exists(cleaned_sim_dir):
        os.mkdir(cleaned_sim_dir)

    if not os.path.exists(failed_sim_dir):
        os.mkdir(failed_sim_dir)

    def move_failed_file(sim_dir, failed_dir, file_name):
        try:
            os.rename(
                os.path.join(sim_dir, file_name), os.path.join(failed_dir, file_name)
            )
        except:
            os.rename(
                os.path.join(sim_dir, file_name),
                os.path.join(failed_dir, file_name + "_1"),
            )

    # purge routine
    for sim in os.listdir(sim_dir):
        if "_failed" in sim:
            try:
                move_failed_file(sim_dir, failed_sim_dir, sim)
            except:
                pass

        else:
            # check to see that it has all of the requisite data
            files = os.listdir(os.path.join(sim_dir, sim))
            if "CFD_mesh.msh" not in files:
                try:
                    move_failed_file(sim_dir, failed_sim_dir, sim)
                except:
                    pass

            if "log.txt" not in files:
                try:
                    move_failed_file(sim_dir, failed_sim_dir, sim)
                except:
                    pass

            if "D_file.txt" not in files:
                try:
                    move_failed_file(sim_dir, failed_sim_dir, sim)
                except:
                    pass

            if "res_file.txt" not in files:
                try:
                    move_failed_file(sim_dir, failed_sim_dir, sim)
                except:
                    pass

            if "sim_status.txt" not in files:
                try:
                    move_failed_file(sim_dir, failed_sim_dir, sim)
                except:
                    pass

            if "sol_file.txt" not in files:
                try:
                    move_failed_file(sim_dir, failed_sim_dir, sim)
                except:
                    pass

    # rename clean sim files in order and move to new directory
    counter = len(os.listdir(cleaned_sim_dir))
    for sim in os.listdir(sim_dir):
        os.rename(
            os.path.join(sim_dir, sim), os.path.join(cleaned_sim_dir, str(counter))
        )
        counter += 1
