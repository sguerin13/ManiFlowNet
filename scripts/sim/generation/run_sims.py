import subprocess
import os
import glob
import re
import json
from types import SimpleNamespace

from scripts.helpers import load_config


def check_processes():
    # this routine pulled from:
    # https://stackoverflow.com/questions/13525882/tasklist-output
    tasks = subprocess.check_output(["tasklist"]).decode().split("\r\n")
    p = []
    for task in tasks:
        m = re.match("(.+?) +(\d+) (.+?) +(\d+) +(\d+.* K).*", task)
        if m is not None:
            p.append(
                {
                    "image": m.group(1),
                    "pid": m.group(2),
                    "session_name": m.group(3),
                    "session_num": m.group(4),
                    "mem_usage": m.group(5),
                }
            )

    # grab the files of interest
    #                   # ansys exe                     workbench      mechanical   fluent_thread     meshing server
    processes_to_kill = [
        "RunWB2.exe",
        "SpaceClaim.exe",
        "AnsysFW.exe",
        "AnsysWBU.exe",
        "fl_mpi2220.exe",
        "AnsMeshingServer.exe",
    ]
    terminate_list = []
    for processes in p:
        if processes["image"] in processes_to_kill:
            terminate_list.append(processes["image"])

    return terminate_list


def cleanup_processes(processes_to_kill):
    # cleanup Fluent processes:
    bat_files = glob.glob("*.bat")
    if len(bat_files) > 0:
        for bf in bat_files:
            subprocess.run(bf)

    for proc in processes_to_kill:
        subprocess.run(["taskkill", "/IM", proc, "/F"])


if __name__ == "__main__":
    config = load_config(os.path.join("scripts", "sim", "generation","run_sims.json"))
    successful_sims = 0
    fail_count = 0

    """assuming that we are running at the top of the project"""

    path_to_routine = os.path.join("sim", "data_generation", "sim_routine.py")

    sim_output_path = config.simOutputPath
    n_sims_to_generate = config.nSimsToGenerate
    max_fails_allowed = config.maxFailsAllowed
    max_sim_runtime = config.maxSimRunTime

    while successful_sims < n_sims_to_generate:
        try:
            result = subprocess.run(
                ["runwb2.exe", "-B", "-R", path_to_routine], timeout=max_sim_runtime
            )

            # check if simulation was successful
            sims = sorted(
                os.listdir(sim_output_path),
                key=lambda x: os.path.getctime(os.path.join(sim_output_path, x)),
            )
            latest_sim = sims[-1]

            if "_failed" in latest_sim:
                fail_count += 1

            else:
                successful_sims += 1

            # run exe cleanup
            cleanup_processes(check_processes())

        except:
            files = sorted(
                os.listdir(sim_output_path),
                key=lambda x: os.path.getctime(os.path.join(sim_output_path, x)),
            )
            if len(files) > 0:
                failed_file = files[-1]
                try:
                    os.rename(
                        os.path.join(sim_output_path, failed_file),
                        os.path.join(sim_output_path, failed_file + "_failed"),
                    )
                except:
                    print("failing to overwrite file name")

            # run exe cleanup
            cleanup_processes(check_processes())

            if fail_count > max_fails_allowed:
                break
