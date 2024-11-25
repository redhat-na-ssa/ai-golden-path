from enum import Enum


class ModelStatus(Enum):
    """
    Controls model production status.
    "Disabled" is when a model is completely disabled - no longer in use.
    "New" is a transitory state that indicates that the model was trained but has not been evaluated for production use yet.
    "Active" are models that are in use in production making predictions.
    "Canary" are models that you want to make predictions with but don't want to use the results in production yet.
    """
    Disabled = 0
    New = 1
    Active = 2
    Canary = 3
