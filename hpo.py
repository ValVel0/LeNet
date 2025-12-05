import optuna
import logging
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.samplers import RandomSampler, NSGAIIISampler, TPESampler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # delete line to enable GPU-based training
from search_space import HPOptimization


def main():
    exp_name = "mnist_lenet5"
    search_strategy = "tpe"
    num_trials = 50
    lr = 0.01
    epochs = 10

    # create experiment directory with `exp_name`, otherwise use default
    exp_dir = os.path.join(
        os.getcwd(),
        "output",
        exp_name,
    )
    os.makedirs(exp_dir, exist_ok=True)

    # Setup the logger for easy parsing of results
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(
        logging.FileHandler(os.path.join(exp_dir, "exp_log.log"), mode="w")
    )
    optuna.logging.enable_propagation()

    # Define search strategy
    if search_strategy.lower() == "rs":
        sampler = RandomSampler()
    elif search_strategy.lower() == "ea":
        sampler = NSGAIIISampler()
    elif search_strategy.lower() == "tpe":
        sampler = TPESampler()
    else:
        raise RuntimeError(
            f"unsuported search strategy: {search_strategy}"
        )

    # create optuna study for optimization
    ot_study = optuna.create_study(
        sampler=sampler,
        study_name=exp_name,
        storage=JournalStorage(
            JournalFileBackend(os.path.join(exp_dir, "journal.log")),
        ),
        load_if_exists=True,
        directions=["maximize"],
    )

    # perform HPO
    ot_study.optimize(
        HPOptimization(
            lr=lr,
            epochs=epochs,
            exp_dir=exp_dir,
        ),
        n_trials=num_trials,
        callbacks=[
            MaxTrialsCallback(
                num_trials,
                states=(TrialState.COMPLETE,)
            )
        ],
    )
    logger.log(
        level=logging.INFO,
        msg="Optimization completed SUCCESSFULLY!"
    )


if __name__ == "__main__":
    main()
