from pydantic import BaseModel

class Contract(BaseModel):
    pass

class ResponseContract(BaseModel):
    value: float
