from pydantic import BaseModel

class LiftoverRequest(BaseModel):
    source_build: str
    target_build: str
    chromosome: str
    start: int
    end: int

class AnnotationCompareRequest(BaseModel):
    gene: str
    version_a: str
    version_b: str
