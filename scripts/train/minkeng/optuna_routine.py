import os
from scripts.helpers import load_config

import optuna
from optuna.samplers import RandomSampler
import torch
import warnings

from scripts.train.minkeng.search import MEObjective

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    config = load_config(
        os.path.join("scripts", "train", "convpoint", "config", "optuna_routine.json")
    )

    VP_all_inputs = config.VPAllInputs
    non_dimensionalize = config.nonDimensionalize
    selected_scaler_config = config.scalerType
    cpu = config.useCPU

    if VP_all_inputs and non_dimensionalize:
        raise Exception("Can't do that")

    if VP_all_inputs and (selected_scaler_config != "std"):
        raise Exception("Can't do that")

    if non_dimensionalize and selected_scaler_config not in ["ndim", "ndim_all_fps"]:
        raise Exception("Cant do that")

    n_voxels = config.optunaConfig.nVoxels
    pruner_patience = config.optunaConfig.prunerPatience
    early_stopping_patience = config.optunaConfig.earlyStoppingPatience
    lr_scheduler_patience = config.optunaConfig.learnRateSchedulerPatience
    max_epochs = config.optunaConfig.maxEpochs
    overfit = config.optunaConfig.overfit

    surface_only = config.networkConfig.surfaceOnly
    normals = config.networkConfig.normals
    rotate = config.networkConfig.rotate
    multiple_branches = config.networkConfig.multipleBranches

    home_path = config.filePaths.dataPath
    exp_path = config.filePaths.experimentNamePath
    exp_db_path = config.filePaths.experimentDBPath

    train_path = os.path.join(home_path, "train")
    val_path = os.path.join(home_path, "val")
    scaler_root_path = os.path.join(home_path, "scalers", selected_scaler_config)
    overfit_path = os.path.join(home_path, "overfit")

    study_name = config.studyName
    
    try:
        torch.cuda.empty_cache()
        objective = MEObjective(
            train_path=train_path,
            val_path=val_path,
            current_exp_folder=exp_path,
            scaler_root_path=scaler_root_path,
            n_voxels=n_voxels,
            surface_only=surface_only,
            normals=normals,
            rotate=rotate,
            overfit=overfit,
            max_epochs=max_epochs,
            VP_all_inputs=VP_all_inputs,
            non_dimensionalize=non_dimensionalize,
            multiple_branches=multiple_branches,
            early_stopping_patience=early_stopping_patience,
            lr_scheduler_patience=lr_scheduler_patience,
        )

        pruner = optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(n_startup_trials=10, n_min_trials=20),
            patience=pruner_patience,
        )
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=pruner,
            sampler=RandomSampler(),
            storage="sqlite:///" + exp_db_path,
            load_if_exists=True,
        )
        study.optimize(
            objective,
            n_trials=1,
            timeout=360 * 60,
            show_progress_bar=False,
            gc_after_trial=True,
        )
        torch.cuda.empty_cache()

    except Exception as e:
        torch.cuda.empty_cache()
        print("Exception", e)
