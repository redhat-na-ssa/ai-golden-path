from typing import Sequence
from kubernetes import client, config, dynamic
from kubernetes.client.rest import ApiException
from os import getenv
from pandas import DataFrame
from re import sub
from requests import post
from traceback import print_exc
from yaml import safe_load

from common import MODEL_NAME, CACHE_TTL


_base_config_str = f"""apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    serving.kserve.io/deploymentMode: ModelMesh
  labels:
    opendatahub.io/dashboard: 'true'
spec:
  predictor:
    model:
      args: []
      env: []
      modelFormat:
        name: onnx
        version: '1'
      runtime: {{ cookiecutter.project_name }}
      storage:
        key: aws-connection-my-storage"""


def get_inference_service_name(unique_id: str) -> str:
    return f"{MODEL_NAME}-{unique_id}"


def _get_removal_job_name(unique_id: str) -> str:
    return f"remove-model-{MODEL_NAME}-{unique_id}"


def _get_namespace() -> str:
    return f"{{ cookiecutter.project_name }}-{getenv('ENVIRONMENT')}"


def _title_to_kebab_case(name: str) -> str:
    new_name = sub(r'(?<!^)(?=[A-Z])', '-', name).lower()
    return "".join(new_name.split())


def _build_model_info(name: str, path: str) -> dict:
    base_config = safe_load(_base_config_str)
    base_config["metadata"]["annotations"]["openshift.io/display-name"] = name
    base_config["metadata"]["name"] = _title_to_kebab_case(name)
    base_config["metadata"]["namespace"] = _get_namespace()
    base_config["spec"]["predictor"]["model"]["storage"]["path"] = path
    return base_config


def add_model(unique_id: str, path: str):
    config.load_incluster_config()
    name = get_inference_service_name(unique_id)
    default_api_client = client.ApiClient()
    model_dict = _build_model_info(name, path)
    dynamic_client = dynamic.DynamicClient(default_api_client)
    crd_api = dynamic_client.resources.get(api_version=model_dict["apiVersion"], kind=model_dict["kind"])

    try:
        batch = client.BatchV1Api()
        batch.delete_namespaced_job("remove-model", "rhods-notebooks")
    except ApiException as e:
        # If the job isn't found, we don't need to remove it.
        if e.status != 404:
            print("Model removal job not canceled.")
            print_exc()

    try:
        crd_api.get(namespace=_get_namespace(), name=name)
        crd_api.patch(body=model_dict, content_type="application/merge-patch+json")
    except dynamic.exceptions.NotFoundError:
        crd_api.create(body=model_dict, namespace=_get_namespace())


def remove_model(unique_id: str):
    """
    # Runs the following commands on a delay.

    config.load_incluster_config()
    name = get_inference_service_name(unique_id)
    default_api_client = client.ApiClient()
    api_client = client.CustomObjectsApi(default_api_client)
    try:
        api_client.delete_namespaced_custom_object(group="serving.kserve.io",
                                                version="v1beta1",
                                                namespace=_get_namespace(),
                                                plural="inferenceservices",
                                                name=_title_to_kebab_case(name)
                                                )
    except Exception as e:
        print("Model not removed from the inference server. It may have already been removed.")
    """
    config.load_incluster_config()
    delay_seconds = CACHE_TTL * 1.5

    name = get_inference_service_name(unique_id)

    container = client.V1Container(
        name=_get_removal_job_name(unique_id),
        image="registry.access.redhat.com/ubi9/python-311",
        command=["/bin/bash"],
        args=["-ec",
              f"echo \"from kubernetes import client, config\" > run.py; echo \"config.load_incluster_config()\" >> run.py; echo \"name = '{name}'\" >> run.py; echo \"default_api_client = client.ApiClient()\" >> run.py; echo \"api_client = client.CustomObjectsApi(default_api_client)\" >> run.py; echo \"try:\" >> run.py; echo \"    api_client.delete_namespaced_custom_object(group='serving.kserve.io',\" >> run.py; echo \"    version='v1beta1',\" >> run.py; echo \"    namespace='{_get_namespace()}',\" >> run.py; echo \"    plural='inferenceservices',\" >> run.py; echo \"    name='{_title_to_kebab_case(name)}'\" >> run.py; echo \"    )\" >> run.py; echo \"except Exception as e:\" >> run.py; echo \"    print('Model {_title_to_kebab_case(name)} not removed from the inference server. It may have already been removed.')\" >> run.py; pip install kubernetes; python run.py"]
    )

    init = client.V1Container(
        name="delay",
        image="registry.access.redhat.com/ubi9/python-311",
        command=["/bin/bash"],
        args=["-ec", f"echo sleeping for {delay_seconds} seconds; sleep {delay_seconds}"]
    )

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(name=_get_removal_job_name(unique_id)),
        spec=client.V1PodSpec(restart_policy="Never",
                            containers=[container],
                            init_containers=[init],
                            service_account_name="model-controller")
    )

    spec = client.V1JobSpec(
        template=template,
        backoff_limit=3,
        ttl_seconds_after_finished=1200  # 20 minutes
    )

    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=_get_removal_job_name(unique_id)),
        spec=spec
    )

    batch = client.BatchV1Api()
    batch.create_namespaced_job(body=job, namespace=_get_namespace())


def predict(data: DataFrame, unique_id: str) -> Sequence:
    model_name = get_inference_service_name(unique_id)
    inference_url = f"http://modelmesh-serving:8008/v2/models/{model_name}/infer"
    json_data = {
        "inputs": [
            {
                "name": "dense_input",
                "shape": list(data.values.shape),
                "datatype": "FP32",
                "data": data.values.tolist()
            }
        ]
        }
    response = post(inference_url, json=json_data)
    response_dict = response.json()
    return response_dict['outputs'][0]['data']
