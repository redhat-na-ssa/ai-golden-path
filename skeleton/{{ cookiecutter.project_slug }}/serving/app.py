from fastapi import FastAPI
from pandas import DataFrame
from uvicorn import run

from contract import Contract, Response
from common.model_factory import load_model
from common.transformations import preprocess, postprocess

app = FastAPI()

@app.post("/predict")
def predict(request: Contract) -> Response:
    data = DataFrame([request.model_dump()])

    preprocessed_data = preprocess(data)

    model = load_model()
    predictions = model.predict(preprocessed_data)

    results = postprocess(predictions)

    return Response(value=results[0])


if __name__ == '__main__':
    run(app, host="0.0.0.0", port=8000)
