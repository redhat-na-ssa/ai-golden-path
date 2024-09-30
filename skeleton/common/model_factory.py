from pickle import dump, load
from typing import Dict, Tuple, Sequence

from common.mlflow_api import (
    save_model as save_to_mlflow,
    list_models,
    list_models_with_metadata
)
from common.model_status import ModelStatus
from common import MODEL_VERSION

_model_path = f"../models/model_{MODEL_VERSION}.pkl"
_cached_model = None


def new_model():
    model = None  # Put your default model constructor here.
    return model

def save_model(model):
    save_to_mlflow(model, MODEL_VERSION, experiment_name="{{ cookiecutter.project_name }}")

# TODO: Use a TTL cache to determine when to reload the model, then run load_model() during the healthcheck
def load_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    models = list_models(MODEL_VERSION, experiment_name="{{ cookiecutter.project_name }}", active_state=ModelStatus.Active)
    
    _cached_model = model
    return model


def load_active_models() -> Dict[str, Tuple[Sequence, Sequence]]:
    models = list_models_with_metadata(MODEL_VERSION, experiment_name="{{ cookiecutter.project_name }}", active_state=ModelStatus.Active)
    return models
