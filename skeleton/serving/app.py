from typing import Tuple, Dict, Sequence, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pandas import DataFrame
from random import seed, random
from socket import gethostbyname, gethostname
from time import time_ns
from uvicorn import run

from contract import Contract, ResponseContract, ModelMetadata
from common import USE_SERVING_RUNTIME
from common.model_factory import invalidate_models, load_active_models, get_model_metadata, get_model
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
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )

@app.get("/health")
def healthcheck(response: Response) -> str:
    if not load_active_models():
        response.status_code = status.HTTP_404_NOT_FOUND
        return "Unhealthy"
        
    return "Ok"


def choose_request_model(models: Dict[str, Union[Sequence, Tuple[Sequence, Sequence]]]) -> str:
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
        metadata = get_model_metadata(this_model_data)
        total_value += metadata['metrics.test_fraction']
        if rng_value < total_value:
            return run_id


@app.put("/reload_models")
def reload_models() -> bool:
    invalidate_models()
    return True


@app.post("/predict")
def predict(request: Contract) -> ResponseContract:
    data = DataFrame([request.model_dump()])

    models = load_active_models()
    model_id = choose_request_model(models)
    this_model = models[model_id]
    metadata = get_model_metadata(this_model)
    this_model_version = f"{metadata['params.major_version']}.{metadata['params.minor_version']}.{metadata['params.micro_version']}"

    model = get_model(this_model)
    results = infer(data, model, metadata.run_id)

    metadata = ModelMetadata(model_id=model_id,
                             model_version=this_model_version,
                             submodel_name=metadata['params.submodel_name'])
    return ResponseContract(value=results[0],
                            metadata=[metadata])


if __name__ == '__main__':
    run(app, host="0.0.0.0", port=8000)
