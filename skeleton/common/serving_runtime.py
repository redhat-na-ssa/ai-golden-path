import re
from yaml import safe_load
from kubernetes import client, config, dynamic

_base_config_str = """apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    serving.kserve.io/deploymentMode: ModelMesh
  namespace: {{ cookiecutter.project_name }}
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


def _title_to_kebab_case(name: str):
    new_name = re.sub(r'(?<!^)(?=[A-Z])', '-', name).lower()
    return "".join(new_name.split())


def _build_model_info(name: str, path: str):
    base_config = safe_load(_base_config_str)
    base_config["metadata"]["annotations"]["openshift.io/display-name"] = name
    base_config["metadata"]["name"] = _title_to_kebab_case(name)
    base_config["spec"]["predictor"]["model"]["storage"]["path"] = path
    return base_config


def add_model(name: str, path: str):
    config.load_incluster_config()
    default_api_client = client.ApiClient()

    model_dict = _build_model_info(name, path)
    dynamic_client = dynamic.DynamicClient(default_api_client)
    crd_api = dynamic_client.resources.get(api_version=model_dict["apiVersion"], kind=model_dict["kind"])

    try:
        crd_api.get(namespace="{{ cookiecutter.project_name }}", name=name)
        crd_api.patch(body=model_dict, content_type="application/merge-patch+json")
    except dynamic.exceptions.NotFoundError:
        crd_api.create(body=model_dict, namespace="{{ cookiecutter.project_name }}")


def remove_model(name: str):
    config.load_incluster_config()
    default_api_client = client.ApiClient()
    api_client = client.CustomObjectsApi(default_api_client)
    try:
        api_client.delete_namespaced_custom_object(group="serving.kserve.io",
                                                version="v1beta1",
                                                namespace="{{ cookiecutter.project_name }}",
                                                plural="inferenceservices",
                                                name=title_to_kebab_case(name)
                                                )
    except Exception as e:
        print("Model not removed from the inference server. It may have already been removed.")
