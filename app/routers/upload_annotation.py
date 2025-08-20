
import os
from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import models

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_annotation(
    file: UploadFile = File(...),
    version_tag: str = Form("v99"),
    genome_build_name: str = Form("GRCh38"),
    file_format: str = Form("GTF"),
    db: Session = Depends(get_db)
):
    dest = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    genome_build = db.query(models.GenomeBuild).filter_by(name=genome_build_name).first()
    if not genome_build:
        genome_build = models.GenomeBuild(name=genome_build_name, species="Homo sapiens")
        db.add(genome_build)
        db.commit()
        db.refresh(genome_build)

    new_file = models.AnnotationFile(
        filename=file.filename,
        stored_path=dest,
        file_format=file_format,
        version_tag=version_tag,
        genome_build_id=genome_build.id
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    return {
        "message": "uploaded",
        "db_id": new_file.id,
        "stored_path": dest,
        "version_tag": version_tag,
        "genome_build": genome_build_name,
        "file_format": file_format
    }
