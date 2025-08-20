from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean,
    ForeignKey, Float, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..database import Base

class GenomeBuild(Base):
    __tablename__ = "genome_builds"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)  # e.g., GRCh38
    species = Column(String(100), nullable=False, default="Homo sapiens")
    assembly_accession = Column(String(50))  # e.g., GCF_000001405.39
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

    annotation_files = relationship("AnnotationFile", back_populates="genome_build")
    coordinate_mappings_source = relationship(
        "CoordinateMapping",
        foreign_keys="CoordinateMapping.source_build_id",
        back_populates="source_build"
    )
    coordinate_mappings_target = relationship(
        "CoordinateMapping",
        foreign_keys="CoordinateMapping.target_build_id",
        back_populates="target_build"
    )
    ensembl_cache = relationship("EnsemblGeneCache", back_populates="genome_build")

class AnnotationFile(Base):
    __tablename__ = "annotation_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(512), nullable=False)
    stored_path = Column(String(1024), nullable=False, unique=True)
    file_format = Column(String(10), nullable=False)  # GTF or GFF3
    version_tag = Column(String(64), nullable=False)  # e.g., v99
    genome_build_id = Column(Integer, ForeignKey("genome_builds.id"))
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)

    genome_build = relationship("GenomeBuild", back_populates="annotation_files")

class CoordinateMapping(Base):
    __tablename__ = "coordinate_mappings"

    id = Column(Integer, primary_key=True)
    source_build_id = Column(Integer, ForeignKey("genome_builds.id"), nullable=False)
    target_build_id = Column(Integer, ForeignKey("genome_builds.id"), nullable=False)
    source_chromosome = Column(String(50), nullable=False)
    source_start = Column(Integer, nullable=False)
    source_end = Column(Integer, nullable=False)
    target_chromosome = Column(String(50), nullable=False)
    target_start = Column(Integer, nullable=False)
    target_end = Column(Integer, nullable=False)
    chain_score = Column(Float)
    liftover_tool = Column(String(100))  # e.g., UCSC liftOver, Ensembl REST
    created_at = Column(DateTime, server_default=func.now())

    source_build = relationship("GenomeBuild", foreign_keys=[source_build_id], back_populates="coordinate_mappings_source")
    target_build = relationship("GenomeBuild", foreign_keys=[target_build_id], back_populates="coordinate_mappings_target")

    __table_args__ = (
        UniqueConstraint('source_build_id', 'target_build_id', 'source_chromosome', 'source_start', 'source_end', name='uq_coordinate_mapping'),
    )

class EnsemblGeneCache(Base):
    __tablename__ = "ensembl_gene_cache"

    id = Column(Integer, primary_key=True)
    gene_id = Column(String(100))        # ENSG0000...
    gene_symbol = Column(String(200))
    genome_build_id = Column(Integer, ForeignKey("genome_builds.id"), nullable=False)
    chromosome = Column(String(50))
    start_pos = Column(Integer)
    end_pos = Column(Integer)
    strand = Column(Integer)
    biotype = Column(String(100))
    description = Column(Text)
    ensembl_version = Column(String(20))
    last_updated = Column(DateTime, server_default=func.now())

    genome_build = relationship("GenomeBuild", back_populates="ensembl_cache")
