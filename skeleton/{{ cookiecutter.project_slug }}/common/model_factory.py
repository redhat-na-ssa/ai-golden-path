from pickle import dump, load

from common import MODEL_VERSION

_model_path = f"../models/model_{MODEL_VERSION}.pkl"
_cached_model = None


def new_model():
    model = None  # Put your default model constructor here.
    return model

def save_model(model):
    with open(_model_path, 'wb') as pickle_file:
        dump(model, pickle_file)

# TODO: Use a TTL cache to determine when to reload the model, then run load_model() during the healthcheck
def load_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    with open(_model_path, 'rb') as pickle_file:
        model = load(pickle_file)
    _cached_model = model
    return model
