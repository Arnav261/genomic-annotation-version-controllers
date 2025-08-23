from fastapi import APIRouter, HTTPException
from typing import List, Dict

router = APIRouter()

@router.get("/")
async def liftover_info():
    """Liftover router information"""
    return {
        "router": "liftover",
        "description": "Coordinate liftover services",
        "status": "active"
    }