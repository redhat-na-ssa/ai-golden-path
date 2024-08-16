from pandas import DataFrame


def preprocess(data: DataFrame) -> DataFrame:
    transformed_data = data
    
    # Any preprocessing that needs to happen when both training and serving goes here

    return transformed_data

def postprocess(data: DataFrame) -> DataFrame:
    transformed_data
    transformed_data = data
    # Any postprocessing that needs to happen when both training and serving goes here

    return transformed_data
