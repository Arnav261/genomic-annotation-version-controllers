from .celery_app import celery_app
import os
import json
import logging
from typing import List, Dict, Any

from .semantic_context import ingest_annotation_embeddings
from .rl_agent import train_agent

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def ingest_annotations_task(self, annotations: List[Dict[str, Any]], version: str, model_key: str = "sbert", seq_type: str = "text"):
    """
    Celery task to ingest annotations in background.
    Returns the list of ingested ids on success.
    """
    try:
        ids = ingest_annotation_embeddings(annotations, version, model_key=model_key, seq_type=seq_type)
        return {"status": "ok", "ingested_ids": ids}
    except Exception as e:
        logger.exception("Background ingestion failed")
        raise

@celery_app.task(bind=True)
def train_rl_task(self, training_payload_path: str, total_timesteps: int = 10000, save_path: str = "app/model_data/rl_agent.zip"):
    """
    training_payload_path: path to JSONL or other preprocessed training data for RL.
    This is a scaffold that calls train_agent with an env_creator you should implement for your dataset.
    """
    # This is a scaffold: operator must write env_creator based on saved feedback
    try:
        # Example: implement env_creator using data at training_payload_path
        def env_creator():
            raise NotImplementedError("Create env_creator for RL training using your curated data.")
        # Call trainer (will raise until env_creator implemented)
        path = train_agent(env_creator, total_timesteps=total_timesteps, save_path=save_path)
        return {"status": "ok", "model_path": path}
    except Exception as e:
        logger.exception("RL training failed")
        raise