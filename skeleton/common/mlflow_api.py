from typing import Tuple, Optional, Dict, Union, Sequence, Any, Set

import mlflow.pyfunc
from mlflow.entities import Run, Experiment
from pandas import DataFrame, Series
from pkg_resources import packaging

from mlflow_utilities.model_status import ModelStatus


def _parse_semver(version: str) -> Tuple[str, str, str]:
    x = packaging.version.parse(version)
    return x.major, x.minor, x.micro


def _build_filter_string(is_immutable: bool, var_name: str, value: Any) -> str:
    if is_immutable:
        return "params." + var_name + ' = "' + str(value) + '"'
    else:
        return "metrics." + var_name + ' = ' + str(value)  # Expects a double


def save_model(model,
               model_version: Union[str, Tuple[str, str, str]],
               mlflow_subpackage=None,
               experiment_id: Optional[str] = None,
               experiment_name: Optional[str] = None,
               submodel_name: Optional[str] = None,
               immutable_metadata: Dict[str, str] = {},
               mutable_metadata: Dict[str, float] = {}):
    """
    Saves a model in MLFlow.

    Args:
        model: Model object to save
        model_version (Union[str, Tuple[str, str, str]]): Semantic Version of the model. Can be handled as either a string or as a tuple of 3 numbers (major, minor, and micro version)
        mlflow_subpackage: The mlflow package itself (normally imported) where you call "log_model". If none is provided, uses the general mlflow.pyfunc.
        experiment_id (Optional[str]): Experiment Id if known. Optional with the experiment name.
        experiment_name (Optional[str]): Experiment Name if known. Optional with the experiment id.
        submodel_name (Optional[str]): Submodel name if you have one. If not provided, uses the name of the experiment.
        immutable_metadata (Dict[str, str]): Any additional metadata inherent to the model or model process that you want to keep track of.
        mutable_metadata (Dict[str, float]): Any additional metadata specific to the model that can change over time.
    """
    if experiment_id is None and experiment_name is None:
        raise ValueError("Experiment Id or Experiment Name must be set")
    if experiment_id is not None:
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_name=experiment_name)

    if submodel_name is None:
        submodel_name = experiment_name

    if mlflow_subpackage is None:
        mlflow_subpackage = mlflow.pyfunc

    if type(model_version) == str:
        model_version = _parse_semver(model_version)

    with mlflow.start_run():
        mlflow.log_param("submodel_name", submodel_name)
        mlflow.log_param("major_version", model_version[0])
        mlflow.log_param("minor_version", model_version[1])
        mlflow.log_param("micro_version", model_version[2])

        mlflow.log_metric("active_state", ModelStatus.New.value)
        mlflow.log_metric("test_fraction", 0.0)

        if immutable_metadata:
            mlflow.log_params(immutable_metadata)
        if mutable_metadata:
            mlflow.log_metrics(mutable_metadata)

        mlflow_subpackage.log_model(model, "", registered_model_name=submodel_name)


def change_status(run_id: str, new_active_state: Optional[ModelStatus], new_test_fraction: Optional[float]):
    """
    Changes the production status of a run/model already saved in MLFlow.

    Args:
        run_id (str): The id of the run.
        new_active_state (Optional[ModelStatus]): The new production status.
        new_test_fraction (Optional[float]): The A/B test fraction.
    """
    run = mlflow.get_run(run_id)
    with mlflow.start_run(run.info.run_id):
        if new_active_state:
            mlflow.log_metric("active_state", new_active_state.value)
        if new_test_fraction:
            mlflow.log_metric("test_fraction", new_test_fraction)


def enable_run(run_id: str):
    """
    Sets a run/model to active production status.

    Args:
        run_id (str): The id of the run.
    """
    change_status(run_id, ModelStatus.Active, 1.0)


def disable_run(run_id: str):
    """
    Sets a run/model to disabled production status.

    Args:
        run_id (str): The id of the run.
    """
    change_status(run_id, ModelStatus.Disabled, 0.0)


def canary_run(run_id: str):
    """
    Sets a run/model to canary production status.

    Args:
        run_id (str): The id of the run.
    """
    change_status(run_id, ModelStatus.Canary, 0.0)


def _rebalance_test_fractions(test_fraction_by_run: Dict[str, float]) -> Dict[str, float]:
    total_fraction = sum(test_fraction_by_run.values())
    if total_fraction <= 0:
        raise ValueError("Test fractions must be strictly positive.")
    return {key: value / total_fraction for key, value in test_fraction_by_run.items()}


def update_active_runs(test_fraction_by_run: Dict[str, float],
                       model_version: Union[str, Tuple[str, str, str]],
                       experiment_id: Optional[str] = None,
                       experiment_name: Optional[str] = None,
                       submodel_name: Optional[str] = None,
                       extra_immutable_metadata: Dict[str, str] = {},
                       extra_mutable_metadata: Dict[str, float] = {}) -> Sequence[Run]:
    """
    Updates which runs are active and in which test fraction.

    Args:
        test_fraction_by_run (Dict[str, float]): Dictionary of run id to the new test fraction.
        model_version (Union[str, Tuple[str, str, str]]): Semantic Version of the model. Can be handled as either a string or as a tuple of 3 numbers (major, minor, and micro version)
        experiment_id (Optional[str]): Experiment Id if known. Optional with the experiment name.
        experiment_name (Optional[str]): Experiment Name if known. Optional with the experiment id.
        submodel_name (Optional[str]): Submodel name if you have one. If not provided, uses the name of the experiment.
        extra_immutable_metadata (Dict[str, str]): Any additional metadata inherent to the model or model process that you need to filter your search by.
        extra_mutable_metadata (Dict[str, float]): Any additional metadata specific to the model that you need to filter your search by.
    """
    test_fraction_by_run = _rebalance_test_fractions(test_fraction_by_run)
    new_runs = list_runs(model_version, experiment_id, experiment_name, ModelStatus.New, submodel_name, extra_immutable_metadata, extra_mutable_metadata)
    active_runs = list_runs(model_version, experiment_id, experiment_name, ModelStatus.Active, submodel_name, extra_immutable_metadata, extra_mutable_metadata)
    canary_runs = list_runs(model_version, experiment_id, experiment_name, ModelStatus.Canary, submodel_name, extra_immutable_metadata, extra_mutable_metadata)

    new_run_set: Set[str] = {run.run_id for _, run in new_runs.iterrows()}
    active_run_set: Set[str] = {run.run_id for _, run in active_runs.iterrows()}
    canary_runs_set: Set[str] = {run.run_id for _, run in canary_runs.iterrows()}

    if test_fraction_by_run.keys() - new_run_set - active_run_set - canary_runs_set:
        raise ValueError("Attempting to modify a run that doesn't exist. Exiting to prevent odd behavior.")

    for run_not_present in active_run_set - test_fraction_by_run.keys():
        disable_run(run_not_present)
    for run_not_present in canary_runs_set - test_fraction_by_run.keys():
        disable_run(run_not_present)

    all_runs_to_update: Set[str] = set(new_run_set).union(active_run_set).union(canary_runs_set)

    for run_to_update in all_runs_to_update.intersection(test_fraction_by_run.keys()):
        if test_fraction_by_run[run_to_update] <= 0.0:
            disable_run(run_to_update)
        else:
            change_status(run_to_update, ModelStatus.Active, test_fraction_by_run[run_to_update])


def change_test_fractions(test_fraction_by_run: Dict[str, float],
                          model_version: Union[str, Tuple[str, str, str]],
                          experiment_id: Optional[str] = None,
                          experiment_name: Optional[str] = None,
                          active_state: Optional[ModelStatus] = None,
                          submodel_name: Optional[str] = None,
                          extra_immutable_metadata: Dict[str, str] = {},
                          extra_mutable_metadata: Dict[str, float] = {}):
    """
    Updates which runs are in what test fraction for a specified production state.

    Args:
        test_fraction_by_run (Dict[str, float]): Dictionary of run id to the new test fraction.
        model_version (Union[str, Tuple[str, str, str]]): Semantic Version of the model. Can be handled as either a string or as a tuple of 3 numbers (major, minor, and micro version)
        experiment_id (Optional[str]): Experiment Id if known. Optional with the experiment name.
        experiment_name (Optional[str]): Experiment Name if known. Optional with the experiment id.
        active_state (Optional[ModelStatus]): Optional[ModelStatus]: Which production state to search and update.
        submodel_name (Optional[str]): Submodel name if you have one. If not provided, uses the name of the experiment.
        extra_immutable_metadata (Dict[str, str]): Any additional metadata inherent to the model or model process that you need to filter your search by.
        extra_mutable_metadata (Dict[str, float]): Any additional metadata specific to the model that you need to filter your search by.
    """
    test_fraction_by_run = _rebalance_test_fractions(test_fraction_by_run)
    runs = list_runs(model_version, experiment_id, experiment_name, active_state, submodel_name, extra_immutable_metadata, extra_mutable_metadata)

    run_dict: Dict[str, Series] = {run.run_id: run for _, run in runs.iterrows()}

    if test_fraction_by_run.keys() - run_dict.keys():
        raise ValueError("Attempting to modify a run that doesn't exist. Exiting to prevent odd behavior.")

    for run_not_present in run_dict.keys() - test_fraction_by_run.keys():
        disable_run(run_not_present)

    for run_to_update in set(run_dict.keys()).intersection(test_fraction_by_run.keys()):
        if test_fraction_by_run[run_to_update] <= 0.0:
            disable_run(run_to_update)
        else:
            change_status(run_to_update, ModelStatus(run_dict[run_to_update]['metrics.active_state']), test_fraction_by_run[run_to_update])


def list_runs(model_version: Union[str, Tuple[str, str, str]],
              experiment_id: Optional[str] = None,
              experiment_name: Optional[str] = None,
              active_state: Optional[ModelStatus] = None,
              submodel_name: Optional[str] = None,
              extra_immutable_metadata: Dict[str, str] = {},
              extra_mutable_metadata: Dict[str, float] = {}) -> DataFrame:
    """
    List runs in MLFlow for the semantic versioning framework.

    Args:
        model_version (Union[str, Tuple[str, str, str]]): Semantic Version of the model. Can be handled as either a string or as a tuple of 3 numbers (major, minor, and micro version)
        experiment_id (Optional[str]): Experiment Id if known. Optional with the experiment name.
        experiment_name (Optional[str]): Experiment Name if known. Optional with the experiment id.
        active_state (Optional[ModelStatus]): Optional[ModelStatus]: Which production state to search and update.
        submodel_name (Optional[str]): Submodel name if you have one. If not provided, uses the name of the experiment.
        extra_immutable_metadata (Dict[str, str]): Any additional metadata inherent to the model or model process that you want to keep track of.
        extra_mutable_metadata (Dict[str, float]): Any additional metadata specific to the model that can change over time.
    """
    if experiment_id is None and experiment_name is None:
        raise ValueError("Experiment Id or Experiment Name must be set.")

    experiment: Optional[Experiment] = None
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    else:
        experiment = mlflow.get_experiment_by_name(experiment_name)

    if not experiment:
        raise ValueError("Experiment does not exist.")

    if type(model_version) == str:
        model_version = _parse_semver(model_version)

    filter = ["status = 'FINISHED'"]

    filter.append(_build_filter_string(True, "major_version", model_version[0]))
    filter.append(_build_filter_string(True, "minor_version", model_version[1]))
    filter.append(_build_filter_string(True, "micro_version", model_version[2]))

    if active_state:
        filter.append(_build_filter_string(False, "active_state", active_state.value))
    if submodel_name:
        filter.append(_build_filter_string(True, "submodel_name", submodel_name))

    for immutable_metadata_name, value in extra_immutable_metadata.items():
        filter.append(_build_filter_string(True, immutable_metadata_name, value))
    for mutable_metadata_name, value in extra_mutable_metadata.items():
        filter.append(_build_filter_string(True, mutable_metadata_name, value))

    runs = mlflow.search_runs(experiment_names=[experiment.name], filter_string=" and ".join(filter), order_by=['end_time desc'])
    return runs


def list_models(model_version: Union[str, Tuple[str, str, str]],
                experiment_id: Optional[str] = None,
                experiment_name: Optional[str] = None,
                active_state: Optional[ModelStatus] = None,
                submodel_name: Optional[str] = None,
                extra_immutable_metadata: Dict[str, str] = {},
                extra_mutable_metadata: Dict[str, float] = {}) -> Sequence:
    """
    List models in MLFlow for the semantic versioning framework.

    Args:
        model_version (Union[str, Tuple[str, str, str]]): Semantic Version of the model. Can be handled as either a string or as a tuple of 3 numbers (major, minor, and micro version)
        experiment_id (Optional[str]): Experiment Id if known. Optional with the experiment name.
        experiment_name (Optional[str]): Experiment Name if known. Optional with the experiment id.
        active_state (Optional[ModelStatus]): Optional[ModelStatus]: Which production state to search and update.
        submodel_name (Optional[str]): Submodel name if you have one. If not provided, uses the name of the experiment.
        extra_immutable_metadata (Dict[str, str]): Any additional metadata inherent to the model or model process that you want to keep track of.
        extra_mutable_metadata (Dict[str, float]): Any additional metadata specific to the model that can change over time.
    """
    runs = list_runs(model_version, experiment_id, experiment_name, active_state, submodel_name, extra_immutable_metadata, extra_mutable_metadata)

    models = []

    for _, run in runs.iterrows():
        filepath = run.artifact_uri
        models.append(mlflow.sklearn.load_model(filepath))

    return models
