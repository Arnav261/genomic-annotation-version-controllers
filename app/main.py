from fastapi import FastAPI
from app.routers import liftover, compare_annotations, upload_annotation

app = FastAPI()

app.include_router(liftover.router, prefix="/liftover", tags=["liftover"])
app.include_router(compare_annotations.router, prefix="/compare-annotations", tags=["compare"])
app.include_router(upload_annotation.router, prefix="/upload-annotation", tags=["uploads"])

@app.get("/")
def root():
    return {"message": "API is up!"}


