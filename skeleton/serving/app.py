from typing import Tuple, Dict, Sequence
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status
from pandas import DataFrame
from random import seed, random
from socket import gethostbyname, gethostname
from time import time_ns
from uvicorn import run

from contract import Contract, ResponseContract, ModelMetadata
from common.model_factory import load_active_models
from common.transformations import infer

def seed_by_time():
    # Seed the RNG at the start of the process by a combination of host IP and time to be unique across multiple instances.
    seed(time_ns() + hash(gethostbyname(gethostname())))

@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_by_time()
    load_active_models()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def healthcheck(response: Response) -> str:
    if not load_active_models():
        response.status_code = status.HTTP_404_NOT_FOUND
        return "Unhealthy"
        
    return "Ok"


def choose_request_model(models: Dict[str, Tuple[Sequence, Sequence]]) -> str:
    """
    Randomly select a live model to predict with based on the test fractions provided.

    Args:
        models: The models and metadata as loaded from MLflow

    Returns:
        The selected run id.
    """
    rng_value = random()
    total_value = 0
    for run_id, this_model_data in models.items():
        total_value += this_model_data[1]['metrics.test_fraction']
        if rng_value < total_value:
            return run_id


@app.post("/predict")
def predict(request: Contract) -> ResponseContract:
    data = DataFrame([request.model_dump()])

    models = load_active_models()
    model_id = choose_request_model(models)
    this_model = models[model_id]
    model = this_model[0]
    this_model_version = f"{this_model[1]['params.major_version']}.{this_model[1]['params.minor_version']}.{this_model[1]['params.micro_version']}"

    results = infer(data, model)

    metadata = ModelMetadata(model_id=model_id,
                             model_version=this_model_version,
                             submodel_name=this_model[1]['params.submodel_name'])
    return ResponseContract(value=results[0],
                            metadata=[metadata])


if __name__ == '__main__':
    run(app, host="0.0.0.0", port=8000)
