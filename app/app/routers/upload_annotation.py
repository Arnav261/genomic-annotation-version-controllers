from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict

router = APIRouter()

@router.get("/")
async def upload_annotation_info():
    """Upload annotation router information"""
    return {
        "router": "upload_annotation", 
        "description": "Annotation upload services",
        "status": "active"
    }