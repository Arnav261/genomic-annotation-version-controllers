from fastapi import APIRouter, HTTPException
from typing import List, Dict

router = APIRouter()

@router.get("/")
async def compare_annotations_info():
    """Compare annotations router information"""
    return {
        "router": "compare_annotations",
        "description": "Annotation comparison services",
        "status": "active"
    }