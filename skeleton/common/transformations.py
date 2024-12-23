from pandas import DataFrame
from typing import Optional, Any

from common.serving_runtime import predict


# Be very careful - the data contracts for these functions are overly flexible.
#  Keep in mind that the inputted data MUST MATCH between the various components for this to work.
#  If you are interested in putting input validation on this function, I would recommend you look at the pandera library.
def preprocess(data: DataFrame) -> DataFrame:
    transformed_data = data
    
    # Any preprocessing that needs to happen when both training and serving goes here

    return transformed_data

def postprocess(data: DataFrame) -> DataFrame:
    transformed_data = data
    # Any postprocessing that needs to happen when both training and serving goes here

    return transformed_data


def infer(data: DataFrame, model: Optional[Any], unique_id: Optional[str] = None):
    preprocessed_data = preprocess(data)
    if model:
        predictions = model.predict(preprocessed_data)
    else:
        predictions = predict(data, unique_id)
    results = postprocess(predictions)
    return results
