from fastapi import APIRouter, HTTPException
from typing import List, Dict

router = APIRouter()
LIFTOVER_CHAINS = {
    "hg19Tohg38": "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
}
@router.get("/")
async def liftover_info():
    """Liftover router information"""
    return {
        "router": "liftover",
        "description": "Coordinate liftover services",
        "status": "active"
    }