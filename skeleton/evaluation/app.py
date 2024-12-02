from common import MODEL_NAME, MODEL_VERSION
from common.mlflow_api import list_runs, load_single_model, update_active_runs
from common.model_status import ModelStatus
from common.transformations import infer
from evaluation.load_data import load_evaluation_data

from pandas import concat, DataFrame
from sklearn.metrics import accuracy_score


def load_live_runs() -> DataFrame:
    new_runs = list_runs(MODEL_VERSION, experiment_name=MODEL_NAME, active_state=ModelStatus.New)
    live_runs = list_runs(MODEL_VERSION, experiment_name=MODEL_NAME, active_state=ModelStatus.Active)
    canary_runs = list_runs(MODEL_VERSION, experiment_name=MODEL_NAME, active_state=ModelStatus.Canary)
    runs = concat([x for x in [new_runs, live_runs, canary_runs] if not x.empty])
    return runs


def evaluate_model_on_data(data: DataFrame, model) -> float:
    predictions = infer(data, model)
    target_column = 'target'
    actuals = data[target_column]
    return accuracy_score(actuals, predictions)


def main():
    new_valid_runs = dict()

    data = load_evaluation_data()

    runs = load_live_runs()
    for _, run in runs.iterrows():
        model = load_single_model(run)
        if run['metrics.active_state'] == ModelStatus.Active.value:
            new_valid_runs[run.run_id] = 2.
        elif evaluate_model_on_data(data, model) > 0.95:
            new_valid_runs[run.run_id] = 1.
        else:
            new_valid_runs[run.run_id] = 0.

    update_active_runs(new_valid_runs, MODEL_VERSION, experiment_name=MODEL_NAME)


if __name__ == "__main__":
    main()
