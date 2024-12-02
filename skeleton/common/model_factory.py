from cachetools import TTLCache
from typing import Dict, Tuple, Sequence

from common.mlflow_api import (
    save_model as save_to_mlflow,
    list_models,
    list_models_with_metadata
)
from common.model_status import ModelStatus
from common import MODEL_NAME, MODEL_VERSION

# The cache will default to 10 minutes, but you can change this as needed.
_model_cache = TTLCache(maxsize=10, ttl=600)


def invalidate_models():
    if "single_model" in _model_cache:
        _model_cache.pop("single_model")
    if "models" in _model_cache:
        _model_cache.pop("models")

def new_model():
    model = None  # Put your default model constructor here.
    return model

def save_model(model):
    save_to_mlflow(model, MODEL_VERSION, experiment_name=MODEL_NAME)

def load_model():
    if "single_model" in _model_cache:
        return _model_cache["single_model"]

    models = list_models(MODEL_VERSION, experiment_name=MODEL_NAME, active_state=ModelStatus.Active)
    model = models[0]
    
    _model_cache["single_model"] = model
    return model


def load_active_models() -> Dict[str, Tuple[Sequence, Sequence]]:
    if "models" in _model_cache:
        return _model_cache["models"]
    
    models = list_models_with_metadata(MODEL_VERSION, experiment_name=MODEL_NAME, active_state=ModelStatus.Active)
    # Don't cache if no models are found since this is technically unhealthy.
    if models:
        _model_cache["models"] = models

    return models
