import os
import numpy as np
import tempfile
from app.vector_store import FaissVectorStore

def test_faiss_add_search(tmp_path):
    idx_path = tmp_path / "faiss.index"
    meta_db = tmp_path / "faiss_meta.db"
    store = FaissVectorStore(index_path=str(idx_path), meta_db_path=str(meta_db), dim=4)
    v1 = np.array([1.0,0.0,0.0,0.0], dtype='float32')
    v2 = np.array([0.9,0.1,0.0,0.0], dtype='float32')
    store.add_vector("id1", v1, version="v1", metadata={"value":"A"})
    store.add_vector("id2", v2, version="v1", metadata={"value":"B"})
    res = store.search(v1, top_k=2, version="v1")
    assert len(res) >= 1
    assert any(r["id"] == "id1" for r in res)
    # persist and reload
    store.persist()
    store2 = FaissVectorStore(index_path=str(idx_path), meta_db_path=str(meta_db), dim=4)
    res2 = store2.search(v1, top_k=2, version="v1")
    assert len(res2) >= 1