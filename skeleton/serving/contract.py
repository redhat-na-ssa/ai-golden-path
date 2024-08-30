from pydantic import BaseModel

class Contract(BaseModel):
    pass

class Response(BaseModel):
    value: float
