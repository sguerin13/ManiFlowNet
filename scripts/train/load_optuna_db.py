import os
import optuna
from optuna.visualization import (plot_contour, plot_edf,
                                  plot_intermediate_values,
                                  plot_optimization_history,
                                  plot_parallel_coordinate,
                                  plot_param_importances, plot_slice)

from scripts.helpers import load_config

if __name__ == "__main__":
    config = load_config(
        os.path.join("scripts", "train", "load_optuna_db.json")
    )
    storage_name = config.experimentDBPath
    storage_name = "sqlite:///{}.db".format(storage_name)
    study_summaries = optuna.study.get_all_study_summaries(storage=storage_name)
    study_name = study_summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    df = study.trials_dataframe()

    # purge outliers
    index_to_remove = []
    for i, trial in enumerate(study.trials):
        if trial.values and trial.values[0] > 5: # this can be changed to set outlier threshold
            index_to_remove.append(i)

    new_trials = [st for i, st in enumerate(study.trials) if i not in index_to_remove]

    study = optuna.create_study()
    study.add_trials(new_trials)
    fig = plot_parallel_coordinate(study)
    fig.show()
    fig = plot_param_importances(study)
    fig.show()
    print(study.best_params)
    print(study.best_value)

    fig = plot_slice(study, ["train_loss", "n_centers"])
    fig.show()

