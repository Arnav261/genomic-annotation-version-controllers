from fastapi import APIRouter
from ..schemas import LiftoverRequest
from ..services import liftover_service

router = APIRouter()

@router.post("/")
def liftover_coords(req: LiftoverRequest):
    return liftover_service.liftover_coords(
        req.source_build, req.target_build, req.chromosome, req.start, req.end
    )
