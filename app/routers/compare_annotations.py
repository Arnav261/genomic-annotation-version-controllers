from fastapi import APIRouter
from ..schemas import AnnotationCompareRequest
from ..services import ensembl


from pydantic import BaseModel

class CompareAnnotationsRequest(BaseModel):
    gene_symbol: str   
    version_a: str
    version_b: str



router = APIRouter()

@router.post("/")
def compare_annotations(req: AnnotationCompareRequest):
    a = ensembl.get_gene_annotation(req.gene, req.version_a)
    b = ensembl.get_gene_annotation(req.gene, req.version_b)
    return {"gene_symbol": req.gene, "version_a": a, "version_b": b}
