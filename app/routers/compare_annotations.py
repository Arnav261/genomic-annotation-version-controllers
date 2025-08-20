from fastapi import APIRouter
from pydantic import BaseModel
from ..services import ensembl

class CompareAnnotationsRequest(BaseModel):
    gene_symbol: str   
    version_a: str
    version_b: str

router = APIRouter()

@router.post("/")
def compare_annotations(req: CompareAnnotationsRequest):
    a = ensembl.parse_gene_annotation(req.gene_symbol, req.version_a)
    b = ensembl.parse_gene_annotation(req.gene_symbol, req.version_b)

    return {
        "gene_symbol": req.gene_symbol,
        "version_a": a,
        "version_b": b
    }
