from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List


class Contract(BaseModel):
    pass

class ModelMetadata(BaseModel):
    model_id: str
    model_version: str
    submodel_name: str
    extra_metadata: List[ModelMetadata] = Field(default_factory=list)

class ResponseContract(BaseModel):
    value: float
    metadata: List[ModelMetadata]
